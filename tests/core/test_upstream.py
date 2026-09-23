import httpx
import pytest

from app.config.settings import ProviderConfig, Settings
from app.core.upstream import UpstreamPool

SETTINGS = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://openrouter.ai/api/v1", protocol="openai", api_key=None
        ),
        "local": ProviderConfig(
            base_url="http://localhost:8080/v1", protocol="openai", api_key=None
        ),
    },
    models={},
    routes=[],
    default_model=None,
)


async def test_same_provider_reuses_the_same_client():
    pool = UpstreamPool(SETTINGS)
    try:
        assert pool.get("openrouter") is pool.get("openrouter")
        assert pool.get("openrouter") is not pool.get("local")
    finally:
        await pool.aclose()


async def test_client_carries_the_provider_base_url_and_split_timeouts():
    pool = UpstreamPool(SETTINGS)
    try:
        client = pool.get("local")
        assert str(client.base_url).rstrip("/") == "http://localhost:8080/v1"
        assert client.timeout.connect == 10.0
        assert client.timeout.read == 60.0
        # `write` and `read` were the only two ever asserted here; a mutation
        # to either the `write` or `pool` bound in `TIMEOUT` would pass every
        # other test silently since nothing exercises them.
        assert client.timeout.write == 30.0
        assert client.timeout.pool == 10.0
    finally:
        await pool.aclose()


async def test_aclose_closes_the_clients_and_empties_the_pool():
    pool = UpstreamPool(SETTINGS)
    first = pool.get("openrouter")
    await pool.aclose()

    assert first.is_closed
    # A fresh get() after aclose must build a brand-new client: proof the
    # internal map was actually emptied, not just left with closed entries.
    second = pool.get("openrouter")
    assert second is not first
    await pool.aclose()


async def test_aclose_closes_every_client_not_just_the_first():
    # With a single client in the pool, a mutation that only closes the first
    # entry of `self._clients.values()` (e.g. breaking out of the loop after
    # one iteration) would still pass `test_aclose_closes_the_clients_and_
    # empties_the_pool` above. Two providers make that mutation observable.
    pool = UpstreamPool(SETTINGS)
    first = pool.get("openrouter")
    second = pool.get("local")
    await pool.aclose()
    assert first.is_closed
    assert second.is_closed


async def test_injected_transport_is_actually_used_by_the_client():
    # The base_url is deliberately unroutable (port 9 is the standard
    # "discard" port, nothing answers there): if the transport wiring ever
    # regressed and the client fell through to a real transport, this test
    # must fail fast and locally instead of depending on egress to the real
    # internet to prove the regression.
    unroutable = Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="http://127.0.0.1:9", protocol="openai", api_key=None
            )
        },
        models={},
        routes=[],
        default_model=None,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"via": "fake-transport"})

    fake_transport = httpx.MockTransport(handler)
    pool = UpstreamPool(unroutable, transport=fake_transport)
    try:
        client = pool.get("openrouter")
        response = await client.get("/")
        assert response.status_code == 200
        assert response.json() == {"via": "fake-transport"}
    finally:
        await pool.aclose()


# ---------------------------------------------------------------------------
# Slots de concorrencia por provedor
# ---------------------------------------------------------------------------

LIMITED = Settings(
    providers={
        "local": ProviderConfig(
            base_url="http://localhost:8080/v1", protocol="openai", max_concurrency=2
        ),
        "cloud": ProviderConfig(base_url="https://cloud.test/v1", protocol="openai"),
    },
    models={},
    routes=[],
    default_model=None,
)


async def test_a_limited_provider_hands_out_slots_up_to_its_limit():
    pool = UpstreamPool(LIMITED)
    assert pool.limit("local") == 2
    first = pool.try_slot("local")
    second = pool.try_slot("local")
    assert first is not None and second is not None
    assert pool.in_use("local") == 2
    # Cheio: a terceira nao espera, volta None na hora.
    assert pool.try_slot("local") is None
    assert pool.in_use("local") == 2
    first.release()
    assert pool.in_use("local") == 1
    third = pool.try_slot("local")
    assert third is not None
    assert pool.in_use("local") == 2
    second.release()
    third.release()
    assert pool.in_use("local") == 0


async def test_releasing_a_slot_twice_frees_only_one_place():
    pool = UpstreamPool(LIMITED)
    slot = pool.try_slot("local")
    other = pool.try_slot("local")
    assert slot is not None and other is not None
    slot.release()
    slot.release()
    assert pool.in_use("local") == 1
    # Se a segunda liberacao tivesse devolvido lugar, caberiam dois agora.
    assert pool.try_slot("local") is not None
    assert pool.try_slot("local") is None


async def test_an_unlimited_provider_never_runs_out_of_slots():
    pool = UpstreamPool(LIMITED)
    assert pool.limit("cloud") is None
    slots = [pool.try_slot("cloud") for _ in range(50)]
    assert all(slot is not None for slot in slots)
    for slot in slots:
        assert slot is not None
        slot.release()
    assert pool.in_use("cloud") == 0


async def test_a_slot_is_a_context_manager_that_releases_on_exit():
    pool = UpstreamPool(LIMITED)
    slot = pool.try_slot("local")
    assert slot is not None
    try:
        with slot:
            assert pool.in_use("local") == 1
            raise RuntimeError("falhou no meio")
    except RuntimeError:
        pass
    assert pool.in_use("local") == 0


async def test_wait_slot_gets_the_place_that_frees_before_the_timeout():
    import asyncio

    pool = UpstreamPool(LIMITED)
    held = [pool.try_slot("local"), pool.try_slot("local")]
    asyncio.get_running_loop().call_later(0.02, held[0].release)
    slot = await pool.wait_slot("local", 2.0)
    assert slot is not None
    assert pool.in_use("local") == 2
    slot.release()
    held[1].release()
    assert pool.in_use("local") == 0


async def test_wait_slot_gives_up_at_the_timeout_without_taking_a_place():
    pool = UpstreamPool(LIMITED)
    held = [pool.try_slot("local"), pool.try_slot("local")]
    assert await pool.wait_slot("local", 0.02) is None
    assert pool.in_use("local") == 2
    for slot in held:
        slot.release()
    # A espera que desistiu nao ficou na fila: se ficasse, uma das duas
    # liberacoes iria para ela e um lugar sumiria para sempre.
    assert pool.in_use("local") == 0
    assert pool.try_slot("local") is not None


async def test_wait_slot_on_an_unlimited_provider_returns_at_once():
    pool = UpstreamPool(LIMITED)
    assert await pool.wait_slot("cloud", 0.0) is not None


async def test_a_waiter_in_line_is_served_before_a_newcomer():
    """Quem esperou pelo ultimo recurso nao perde o lugar para um pedido novo
    que chega no mesmo instante em que o slot libera."""
    import asyncio

    pool = UpstreamPool(LIMITED)
    held = [pool.try_slot("local"), pool.try_slot("local")]
    waiter = asyncio.ensure_future(pool.wait_slot("local", 2.0))
    await asyncio.sleep(0)
    held[0].release()
    assert pool.try_slot("local") is None
    slot = await waiter
    assert slot is not None
    slot.release()
    held[1].release()


async def test_wait_slot_takes_a_free_place_at_once():
    # O lugar pode vagar entre o "cheio" do dispatcher e o inicio da espera.
    pool = UpstreamPool(LIMITED)
    slot = await pool.wait_slot("local", 0.0)
    assert slot is not None
    assert pool.in_use("local") == 1
    slot.release()


async def test_a_waiter_cancelled_after_the_place_reached_it_gives_the_place_back():
    """O cliente desconecta no mesmo tique em que o slot chegou: a espera sai
    com CancelledError, e o lugar que ja era dela nao pode ficar preso."""
    import asyncio

    pool = UpstreamPool(LIMITED)
    held = [pool.try_slot("local"), pool.try_slot("local")]
    waiter = asyncio.ensure_future(pool.wait_slot("local", 5.0))
    await asyncio.sleep(0)
    held[0].release()  # o lugar e transferido para a espera...
    waiter.cancel()  # ...que e cancelada antes de rodar de novo
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert pool.in_use("local") == 1
    held[1].release()
    assert pool.in_use("local") == 0


async def test_a_waiter_cancelled_before_any_place_frees_does_not_swallow_the_next_release():
    import asyncio

    pool = UpstreamPool(LIMITED)
    held = [pool.try_slot("local"), pool.try_slot("local")]
    waiter = asyncio.ensure_future(pool.wait_slot("local", 5.0))
    await asyncio.sleep(0)
    waiter.cancel()
    # Liberado antes de a espera cancelada rodar o `finally`: o lugar tem de
    # voltar ao pool, nao ir para uma espera que ja desistiu.
    held[0].release()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert pool.in_use("local") == 1
    held[1].release()
    assert pool.in_use("local") == 0


async def test_a_place_granted_at_once_returns_if_the_request_is_abandoned_unused():
    # O lugar veio no construtor (provedor tinha vaga) e o `with` saiu sem
    # `wait()` -- cliente foi embora antes: o lugar tem de voltar.
    pool = UpstreamPool(LIMITED)
    with pool.request_slot("local"):
        assert pool.in_use("local") == 1
    assert pool.in_use("local") == 0


async def test_a_request_hands_out_its_place_only_once():
    pool = UpstreamPool(LIMITED)
    with pool.request_slot("local") as request:
        first = await request.wait(0.0)
        second = await request.wait(0.0)
    assert first is not None
    assert second is None
    assert pool.in_use("local") == 1
    first.release()
    assert pool.in_use("local") == 0


async def test_a_queued_request_hands_out_its_place_only_once():
    pool = UpstreamPool(LIMITED)
    held = [pool.try_slot("local"), pool.try_slot("local")]
    with pool.request_slot("local") as request:
        held[0].release()
        first = await request.wait(1.0)
        second = await request.wait(0.0)
    assert first is not None
    assert second is None
    assert pool.in_use("local") == 2
    first.release()
    held[1].release()
    assert pool.in_use("local") == 0


async def test_cancelling_an_unused_request_twice_returns_its_place_once():
    pool = UpstreamPool(LIMITED)
    other = pool.try_slot("local")
    request = pool.request_slot("local")
    assert pool.in_use("local") == 2
    request.cancel()
    request.cancel()
    assert pool.in_use("local") == 1
    assert other is not None
    other.release()


# ---------------------------------------------------------------------------
# update(): gates reconciliados no lugar, sem zerar in_use
# ---------------------------------------------------------------------------


def _limited(**limits: int | None) -> Settings:
    return Settings(
        providers={
            name: ProviderConfig(
                base_url=f"http://{name}.test/v1", protocol="openai", max_concurrency=limit
            )
            for name, limit in limits.items()
        },
        models={},
        routes=[],
        default_model=None,
    )


async def test_update_raising_the_limit_wakes_waiters_up_to_the_new_capacity():
    pool = UpstreamPool(_limited(local=1))
    held = pool.try_slot("local")
    assert held is not None
    first = pool.request_slot("local")
    second = pool.request_slot("local")
    third = pool.request_slot("local")
    await pool.update(_limited(local=3))
    assert pool.limit("local") == 3
    assert await first.wait(0) is not None
    assert await second.wait(0) is not None
    # Capacidade nova (3) ja ocupada: o terceiro continua na fila.
    assert await third.wait(0) is None
    assert pool.in_use("local") == 3
    third.cancel()
    held.release()
    await pool.aclose()


async def test_update_raising_the_limit_with_no_waiters_does_not_raise():
    # `resize` so deve tentar tirar da fila enquanto ela tiver gente: sem
    # ninguem esperando, subir o limite nao pode estourar a deque vazia.
    pool = UpstreamPool(_limited(local=1))
    await pool.update(_limited(local=3))
    assert pool.limit("local") == 3
    assert pool.in_use("local") == 0
    await pool.aclose()


async def test_update_raising_the_limit_with_fewer_waiters_than_new_capacity_serves_them_without_raising():
    # Capacidade nova (3) maior que o total de candidatos a receber lugar (1
    # em uso + 1 esperando = 2): o loop tem de parar quando a fila esvazia,
    # nao quando a capacidade enche.
    pool = UpstreamPool(_limited(local=1))
    held = pool.try_slot("local")
    assert held is not None
    waiting = pool.request_slot("local")
    await pool.update(_limited(local=3))
    assert pool.limit("local") == 3
    assert pool.in_use("local") == 2
    got = await waiting.wait(0)
    assert got is not None
    got.release()
    held.release()
    assert pool.in_use("local") == 0
    await pool.aclose()


async def test_update_lowering_the_limit_keeps_in_flight_and_holds_new_requests():
    pool = UpstreamPool(_limited(local=2))
    a = pool.try_slot("local")
    b = pool.try_slot("local")
    assert a is not None and b is not None
    await pool.update(_limited(local=1))
    assert pool.in_use("local") == 2  # preservado, nao zerado
    assert pool.try_slot("local") is None
    waiting = pool.request_slot("local")
    a.release()  # 2 -> 1: ainda nao cabe entregar acima do limite novo
    assert pool.in_use("local") == 1
    assert await waiting.wait(0) is None
    b.release()  # 1 <= 1: agora o lugar passa direto a quem espera
    got = await waiting.wait(0)
    assert got is not None
    assert pool.in_use("local") == 1
    got.release()
    await pool.aclose()


async def test_update_removing_the_limit_releases_waiters_at_once():
    pool = UpstreamPool(_limited(local=1))
    held = pool.try_slot("local")
    assert held is not None
    waiting = pool.request_slot("local")
    await pool.update(_limited(local=None))
    assert pool.limit("local") is None
    got = await waiting.wait(0)
    assert got is not None
    got.release()  # devolve num gate orfao: sem efeito no pool
    held.release()
    assert pool.in_use("local") == 0
    assert pool.try_slot("local") is not None
    await pool.aclose()


async def test_update_adding_a_limit_to_a_provider_creates_its_gate():
    pool = UpstreamPool(_limited(local=None))
    assert pool.limit("local") is None
    await pool.update(_limited(local=1))
    assert pool.limit("local") == 1
    assert pool.try_slot("local") is not None
    assert pool.try_slot("local") is None
    await pool.aclose()


async def test_update_removing_the_provider_drops_its_gate_and_frees_waiters():
    pool = UpstreamPool(_limited(local=1, cloud=None))
    held = pool.try_slot("local")
    assert held is not None
    waiting = pool.request_slot("local")
    await pool.update(_limited(cloud=None))
    assert pool.limit("local") is None
    assert await waiting.wait(0) is not None
    held.release()
    await pool.aclose()


async def test_update_removing_the_limit_releases_every_waiter_not_just_the_first():
    # Com uma unica espera, um `release_all` que so libera a primeira (e nao
    # esvazia a fila toda) passaria despercebido: duas esperas tornam isso
    # observavel.
    pool = UpstreamPool(_limited(local=1))
    held = pool.try_slot("local")
    assert held is not None
    first = pool.request_slot("local")
    second = pool.request_slot("local")
    await pool.update(_limited(local=None))
    assert pool.limit("local") is None
    got_first = await first.wait(0)
    got_second = await second.wait(0)
    assert got_first is not None
    assert got_second is not None
    got_first.release()
    got_second.release()
    held.release()
    assert pool.in_use("local") == 0
    await pool.aclose()


async def test_in_use_survives_an_update_that_changes_nothing_relevant():
    pool = UpstreamPool(_limited(local=2))
    held = pool.try_slot("local")
    assert held is not None
    await pool.update(_limited(local=2))
    assert pool.in_use("local") == 1
    assert pool.limit("local") == 2
    held.release()
    assert pool.in_use("local") == 0
    await pool.aclose()


# ---------------------------------------------------------------------------
# update(): clientes aposentados fecham quando o ultimo uso termina
# ---------------------------------------------------------------------------


def _moved() -> Settings:
    moved = SETTINGS.model_copy(deep=True)
    moved.providers["local"].base_url = "http://localhost:9090/v1"
    return moved


async def test_update_with_a_new_base_url_serves_a_new_client_and_closes_the_idle_old_one():
    pool = UpstreamPool(SETTINGS)
    old = pool.get("local")
    await pool.update(_moved())
    new = pool.get("local")
    assert new is not old
    assert str(new.base_url).rstrip("/") == "http://localhost:9090/v1"
    assert old.is_closed
    await pool.aclose()


async def test_a_retired_client_stays_open_while_a_request_uses_it_and_closes_after():
    pool = UpstreamPool(SETTINGS)
    async with pool.client("local") as lent:
        await pool.update(_moved())
        assert not lent.is_closed  # em uso: o update nao fecha
        async with pool.client("local") as fresh:
            assert fresh is not lent  # requisicao nova ja recebe o endereco novo
    assert lent.is_closed  # ultimo uso terminou: fechou
    assert not pool.get("local").is_closed
    await pool.aclose()


async def test_a_retired_client_shared_by_two_requests_closes_only_after_both():
    pool = UpstreamPool(SETTINGS)
    first = pool.client("local")
    second = pool.client("local")
    a = await first.__aenter__()
    b = await second.__aenter__()
    assert a is b
    await pool.update(_moved())
    await first.__aexit__(None, None, None)
    assert not a.is_closed  # o segundo ainda usa
    await second.__aexit__(None, None, None)
    assert a.is_closed
    await pool.aclose()


async def test_update_keeps_the_client_when_only_key_or_limit_changes():
    pool = UpstreamPool(SETTINGS)
    same = pool.get("local")
    changed = SETTINGS.model_copy(deep=True)
    changed.providers["local"].api_key = "nova"
    changed.providers["local"].max_concurrency = 4
    await pool.update(changed)
    assert pool.get("local") is same
    assert not same.is_closed
    assert pool.limit("local") == 4
    await pool.aclose()


async def test_update_removing_a_provider_retires_its_client():
    pool = UpstreamPool(SETTINGS)
    old = pool.get("local")
    only_openrouter = Settings(
        providers={"openrouter": SETTINGS.providers["openrouter"]},
        models={},
        routes=[],
        default_model=None,
    )
    await pool.update(only_openrouter)
    assert old.is_closed
    with pytest.raises(KeyError):
        pool.get("local")
    await pool.aclose()


async def test_aclose_closes_retired_clients_still_in_use():
    pool = UpstreamPool(SETTINGS)
    lease = pool.client("local")
    lent = await lease.__aenter__()
    await pool.update(_moved())
    await pool.aclose()
    assert lent.is_closed
    await lease.__aexit__(None, None, None)  # nao levanta com o cliente ja fechado
