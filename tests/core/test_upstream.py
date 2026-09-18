import httpx

from app.config.settings import ProviderConfig, Settings
from app.core.upstream import UpstreamPool

SETTINGS = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://openrouter.ai/api/v1", protocol="openai", api_key_env=None
        ),
        "local": ProviderConfig(
            base_url="http://localhost:8080/v1", protocol="openai", api_key_env=None
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
                base_url="http://127.0.0.1:9", protocol="openai", api_key_env=None
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
