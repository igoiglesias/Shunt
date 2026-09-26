"""Limite de concorrencia por provedor: cheio e pulado na hora, nao enfileirado.

Medido: um llama.cpp com 2 slots, com 4-8 requisicoes simultaneas, segurava as
excedentes na fila dele ate o ReadTimeout. Com `max_concurrency`, o proxy sabe
que o provedor esta cheio e passa ao proximo candidato sem esperar -- e so
espera um slot quando nao sobrou nenhum outro candidato (ultimo recurso).
"""

import asyncio

import httpx
import pytest
import respx

from app.config.settings import ModelConfig, ProviderConfig, Settings
from app.core import dispatcher
from app.core.dispatcher import ShuntRequest, dispatch
from app.core.upstream import UpstreamPool
from tests.core.test_dispatcher import BODY, ok_payload

LOCAL_URL = "http://local.test/v1/chat/completions"
LOCAL2_URL = "http://local2.test/v1/chat/completions"
CLOUD_URL = "https://api.test/v1/chat/completions"


def _settings(*aliases: str, cloud_key: str | None = "sk-teste") -> Settings:
    return Settings(
        providers={
            "local": ProviderConfig(
                base_url="http://local.test/v1", protocol="openai", api_key="k", max_concurrency=1
            ),
            "local2": ProviderConfig(
                base_url="http://local2.test/v1", protocol="openai", api_key="k", max_concurrency=1
            ),
            "cloud": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key=cloud_key
            ),
        },
        models={
            "loc": ModelConfig(
                provider="local", model="vendor/local", context_window=64000, max_output_tokens=8192
            ),
            "loc2": ModelConfig(
                provider="local2", model="vendor/local2", context_window=64000, max_output_tokens=8192
            ),
            "cld": ModelConfig(
                provider="cloud", model="vendor/cloud", context_window=64000, max_output_tokens=8192
            ),
        },
        routes=[("opus", list(aliases))],
        default_model=None,
    )


LOCAL_THEN_CLOUD = _settings("loc", "cld")


def _occupy(pool: UpstreamPool, provider: str):
    slot = pool.try_slot(provider)
    assert slot is not None
    return slot


# ---------------------------------------------------------------------------
# Caminho bufferizado
# ---------------------------------------------------------------------------


@respx.mock
async def test_a_full_provider_is_skipped_at_once_and_the_next_candidate_serves():
    local = respx.post(LOCAL_URL).mock(return_value=httpx.Response(200, json=ok_payload("x")))
    respx.post(CLOUD_URL).mock(return_value=httpx.Response(200, json=ok_payload("vendor/cloud")))
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    held = _occupy(pool, "local")
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), LOCAL_THEN_CLOUD, pool)
    finally:
        held.release()
        await pool.aclose()
    assert result.status == 200
    assert result.real_model == "vendor/cloud"
    assert local.call_count == 0
    assert result.trace == ["vendor/local: busy (1/1 in use)"]


@respx.mock
async def test_the_slot_is_held_across_every_retry_and_released_on_success(monkeypatch):
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    seen: list[int] = []

    def answer(request):
        seen.append(pool.in_use("local"))
        if len(seen) == 1:
            return httpx.Response(503, json={"error": {"message": "ocupado"}})
        return httpx.Response(200, json=ok_payload("vendor/local"))

    respx.post(LOCAL_URL).mock(side_effect=answer)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), LOCAL_THEN_CLOUD, pool)
        assert pool.in_use("local") == 0
    finally:
        await pool.aclose()
    assert result.real_model == "vendor/local"
    assert seen == [1, 1]


@respx.mock
async def test_the_slot_is_released_when_the_candidate_fails_and_the_chain_moves_on():
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    seen: list[int] = []

    def cloud(request):
        seen.append(pool.in_use("local"))
        return httpx.Response(200, json=ok_payload("vendor/cloud"))

    respx.post(LOCAL_URL).mock(return_value=httpx.Response(400, json={"error": {"message": "nao"}}))
    respx.post(CLOUD_URL).mock(side_effect=cloud)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), LOCAL_THEN_CLOUD, pool)
    finally:
        await pool.aclose()
    assert result.real_model == "vendor/cloud"
    # Liberado ANTES do proximo candidato, nao so no fim da requisicao.
    assert seen == [0]


@respx.mock
async def test_the_slot_is_released_when_an_unexpected_exception_escapes():
    respx.post(LOCAL_URL).mock(side_effect=RuntimeError("bug no meio"))
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    try:
        with pytest.raises(RuntimeError):
            await dispatch(ShuntRequest("anthropic", BODY, {}), LOCAL_THEN_CLOUD, pool)
        assert pool.in_use("local") == 0
    finally:
        await pool.aclose()


@respx.mock
async def test_the_slot_is_released_when_the_2xx_body_is_unreadable():
    respx.post(LOCAL_URL).mock(return_value=httpx.Response(200, text="<html>"))
    respx.post(CLOUD_URL).mock(return_value=httpx.Response(200, json=ok_payload("vendor/cloud")))
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), LOCAL_THEN_CLOUD, pool)
        assert pool.in_use("local") == 0
    finally:
        await pool.aclose()
    assert result.real_model == "vendor/cloud"


@respx.mock
async def test_last_resort_waits_for_the_busy_candidate_when_nothing_else_can_serve():
    # `cld` sem chave e inelegivel; `loc` esta cheio. Sem o ultimo recurso a
    # requisicao falharia com um slot vagando logo depois.
    settings = _settings("loc", "cld", cloud_key=None)
    respx.post(LOCAL_URL).mock(return_value=httpx.Response(200, json=ok_payload("vendor/local")))
    pool = UpstreamPool(settings)
    held = _occupy(pool, "local")
    asyncio.get_running_loop().call_later(0.05, held.release)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), settings, pool)
        assert pool.in_use("local") == 0
    finally:
        await pool.aclose()
    assert result.status == 200
    assert result.real_model == "vendor/local"
    assert result.trace[0] == "vendor/local: busy (1/1 in use)"


@respx.mock
async def test_last_resort_gives_up_at_the_total_deadline(monkeypatch):
    monkeypatch.setattr(dispatcher, "TOTAL_DEADLINE", 0.05)
    settings = _settings("loc")
    local = respx.post(LOCAL_URL).mock(return_value=httpx.Response(200, json=ok_payload("x")))
    pool = UpstreamPool(settings)
    held = _occupy(pool, "local")
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), settings, pool)
        assert pool.in_use("local") == 1
    finally:
        held.release()
        await pool.aclose()
    assert local.call_count == 0
    assert result.status == 502
    assert result.trace == [
        "vendor/local: busy (1/1 in use)",
        "vendor/local: still busy at the deadline",
    ]
    assert result.body["error"]["message"].startswith("no candidate answered - tried: ")
    assert "still busy at the deadline" in result.body["error"]["message"]


@respx.mock
async def test_last_resort_waits_on_the_first_busy_candidate():
    settings = _settings("loc", "loc2")
    respx.post(LOCAL_URL).mock(return_value=httpx.Response(200, json=ok_payload("vendor/local")))
    respx.post(LOCAL2_URL).mock(return_value=httpx.Response(200, json=ok_payload("x")))
    pool = UpstreamPool(settings)
    first = _occupy(pool, "local")
    second = _occupy(pool, "local2")
    asyncio.get_running_loop().call_later(0.05, first.release)
    try:
        result = await asyncio.wait_for(
            dispatch(ShuntRequest("anthropic", BODY, {}), settings, pool), 5.0
        )
    finally:
        second.release()
        await pool.aclose()
    assert result.real_model == "vendor/local"
    assert result.trace == [
        "vendor/local: busy (1/1 in use)",
        "vendor/local2: busy (1/1 in use)",
    ]


@respx.mock
async def test_no_last_resort_once_some_candidate_was_actually_tried(monkeypatch):
    monkeypatch.setattr(dispatcher, "TOTAL_DEADLINE", 5.0)
    respx.post(CLOUD_URL).mock(return_value=httpx.Response(400, json={"error": {"message": "nao"}}))
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    held = _occupy(pool, "local")
    try:
        result = await asyncio.wait_for(
            dispatch(ShuntRequest("anthropic", BODY, {}), LOCAL_THEN_CLOUD, pool), 1.0
        )
    finally:
        held.release()
        await pool.aclose()
    assert result.status == 400
    assert result.trace == [
        "vendor/local: busy (1/1 in use)",
        "vendor/cloud: 400 nao (attempt 1)",
    ]


@respx.mock
async def test_no_last_resort_when_nothing_was_busy():
    # Cadeia toda inelegivel e ninguem cheio: o erro de sempre, sem espera.
    settings = _settings("cld", cloud_key=None)
    pool = UpstreamPool(settings)
    try:
        result = await asyncio.wait_for(
            dispatch(ShuntRequest("anthropic", BODY, {}), settings, pool), 1.0
        )
    finally:
        await pool.aclose()
    assert result.status == 502
    assert not any("busy" in line for line in result.trace)


# ---------------------------------------------------------------------------
# Streaming: o slot vale pelo stream INTEIRO
# ---------------------------------------------------------------------------

SSE_HEADERS = {"content-type": "text/event-stream"}
STREAM_BODY = {**BODY, "stream": True}


def _sse(*texts: str) -> str:
    chunks = "".join(
        f'data: {{"choices": [{{"delta": {{"content": "{t}"}}}}]}}\n\n' for t in texts
    )
    return chunks + "data: [DONE]\n\n"


async def _collect(settings, pool) -> bytes:
    return b"".join(
        [
            chunk
            async for chunk in dispatcher.dispatch_stream(
                ShuntRequest("anthropic", STREAM_BODY, {}), settings, pool
            )
        ]
    )


@respx.mock
async def test_a_full_provider_is_skipped_in_streaming_too():
    local = respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(200, text=_sse("x"), headers=SSE_HEADERS)
    )
    respx.post(CLOUD_URL).mock(
        return_value=httpx.Response(200, text=_sse("da-nuvem"), headers=SSE_HEADERS)
    )
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    held = _occupy(pool, "local")
    try:
        body = await _collect(LOCAL_THEN_CLOUD, pool)
    finally:
        held.release()
        await pool.aclose()
    assert local.call_count == 0
    assert b"da-nuvem" in body


@respx.mock
async def test_the_busy_line_reaches_the_streaming_trace():
    respx.post(CLOUD_URL).mock(return_value=httpx.Response(400, json={"error": {"message": "n"}}))
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    held = _occupy(pool, "local")
    try:
        body = await _collect(LOCAL_THEN_CLOUD, pool)
    finally:
        held.release()
        await pool.aclose()
    assert b"vendor/local: busy (1/1 in use)" in body


class _Watched(httpx.AsyncByteStream):
    """Stream que anota quantos slots do `local` estao ocupados a cada chunk."""

    def __init__(self, pool: UpstreamPool, *chunks: bytes) -> None:
        self._pool = pool
        self._chunks = chunks
        self.seen: list[int] = []

    async def __aiter__(self):
        for chunk in self._chunks:
            self.seen.append(self._pool.in_use("local"))
            yield chunk

    async def aclose(self) -> None:
        pass


@respx.mock
async def test_the_slot_is_held_while_streaming_and_freed_after_full_consumption():
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    stream = _Watched(
        pool,
        b'data: {"choices": [{"delta": {"content": "a"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "b"}}]}\n\n',
    )
    respx.post(LOCAL_URL).mock(return_value=httpx.Response(200, headers=SSE_HEADERS, stream=stream))
    seen_by_client: list[int] = []
    try:
        async for _ in dispatcher.dispatch_stream(
            ShuntRequest("anthropic", STREAM_BODY, {}), LOCAL_THEN_CLOUD, pool
        ):
            seen_by_client.append(pool.in_use("local"))
        assert pool.in_use("local") == 0
    finally:
        await pool.aclose()
    assert stream.seen == [1, 1]
    # Ja entregue ao cliente, o stream ainda segura o slot: o provedor continua
    # gerando enquanto o cliente le.
    assert seen_by_client[0] == 1


@respx.mock
async def test_the_slot_is_held_during_a_raw_passthrough_and_freed_after():
    settings = Settings(
        providers={
            "local": ProviderConfig(
                base_url="http://local.test/v1", protocol="openai", api_key="k", max_concurrency=1
            )
        },
        models={
            "loc": ModelConfig(
                provider="local", model="vendor/local", context_window=64000, max_output_tokens=8192
            )
        },
        routes=[("gpt-4o", ["loc"])],
        default_model=None,
    )
    pool = UpstreamPool(settings)
    stream = _Watched(pool, b'data: {"choices": [{"delta": {"content": "a"}}]}\n\n', b"data: [DONE]\n\n")
    respx.post(LOCAL_URL).mock(return_value=httpx.Response(200, headers=SSE_HEADERS, stream=stream))
    body = {"model": "gpt-4o", "max_tokens": 64, "stream": True, "messages": [{"role": "user", "content": "oi"}]}
    try:
        chunks = [
            c
            async for c in dispatcher.dispatch_stream(ShuntRequest("openai", body, {}), settings, pool)
        ]
        assert pool.in_use("local") == 0
    finally:
        await pool.aclose()
    assert stream.seen == [1, 1]
    assert b"".join(chunks).count(b"[DONE]") == 1


@respx.mock
async def test_the_slot_is_freed_after_an_upstream_error(monkeypatch):
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    respx.post(LOCAL_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "caiu"}}))
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    seen: list[int] = []

    def cloud(request):
        seen.append(pool.in_use("local"))
        return httpx.Response(200, text=_sse("ok"), headers=SSE_HEADERS)

    respx.post(CLOUD_URL).mock(side_effect=cloud)
    try:
        await _collect(LOCAL_THEN_CLOUD, pool)
        assert pool.in_use("local") == 0
    finally:
        await pool.aclose()
    assert seen == [0]


@respx.mock
async def test_the_slot_is_freed_after_a_mid_stream_failure():
    class Dies(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices": [{"delta": {"content": "a"}}]}\n\n'
            raise httpx.ReadTimeout("silencio")

        async def aclose(self) -> None:
            pass

    respx.post(LOCAL_URL).mock(return_value=httpx.Response(200, headers=SSE_HEADERS, stream=Dies()))
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    try:
        body = await _collect(LOCAL_THEN_CLOUD, pool)
        assert pool.in_use("local") == 0
    finally:
        await pool.aclose()
    assert b"ReadTimeout" in body


@respx.mock
async def test_the_slot_is_freed_when_the_client_disconnects_early():
    respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(200, text=_sse("a", "b", "c"), headers=SSE_HEADERS)
    )
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    generator = dispatcher.dispatch_stream(
        ShuntRequest("anthropic", STREAM_BODY, {}), LOCAL_THEN_CLOUD, pool
    )
    try:
        await generator.__anext__()
        assert pool.in_use("local") == 1
        await generator.aclose()
        assert pool.in_use("local") == 0
    finally:
        await pool.aclose()


async def _enter_last_resort(monkeypatch, settings, pool):
    """Leva o stream ate a espera do ultimo recurso e devolve o gerador parado
    no primeiro keep-alive: prova de que ele esta esperando, sem relogio real.
    O `_now()` fica parado em 0, entao o prazo nunca vence sozinho."""
    monkeypatch.setattr(dispatcher, "_now", _FrozenClock(0.0))
    monkeypatch.setattr(dispatcher, "PING_INTERVAL", 0.001)
    stream = dispatcher.dispatch_stream(
        ShuntRequest("anthropic", STREAM_BODY, {}), settings, pool
    )
    assert await stream.__anext__() == b": ping\n\n"
    return stream


@respx.mock
async def test_streaming_last_resort_serves_when_the_slot_frees_in_time(monkeypatch):
    settings = _settings("loc", "cld", cloud_key=None)
    respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(200, text=_sse("do-local"), headers=SSE_HEADERS)
    )
    pool = UpstreamPool(settings)
    held = _occupy(pool, "local")
    stream = await _enter_last_resort(monkeypatch, settings, pool)
    try:
        held.release()
        body = b"".join([chunk async for chunk in stream])
        assert pool.in_use("local") == 0
    finally:
        await stream.aclose()
        await pool.aclose()
    assert b"do-local" in body
    # Servido pelo ultimo recurso, o stream termina ali: nenhum erro depois.
    assert b"event: error" not in body


@respx.mock
async def test_streaming_last_resort_waits_on_the_first_busy_candidate(monkeypatch):
    settings = _settings("loc", "loc2")
    respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(200, text=_sse("do-primeiro"), headers=SSE_HEADERS)
    )
    respx.post(LOCAL2_URL).mock(
        return_value=httpx.Response(200, text=_sse("do-segundo"), headers=SSE_HEADERS)
    )
    pool = UpstreamPool(settings)
    first = _occupy(pool, "local")
    second = _occupy(pool, "local2")
    stream = await _enter_last_resort(monkeypatch, settings, pool)
    try:
        first.release()
        body = b"".join([chunk async for chunk in stream])
    finally:
        await stream.aclose()
        second.release()
        await pool.aclose()
    assert b"do-primeiro" in body
    assert b"do-segundo" not in body


@respx.mock
async def test_streaming_last_resort_reports_why_the_awaited_candidate_failed(monkeypatch):
    settings = _settings("loc")
    respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(400, json={"error": {"message": "contexto estourou"}})
    )
    pool = UpstreamPool(settings)
    held = _occupy(pool, "local")
    stream = await _enter_last_resort(monkeypatch, settings, pool)
    try:
        held.release()
        body = b"".join([chunk async for chunk in stream])
        assert pool.in_use("local") == 0
    finally:
        await stream.aclose()
        await pool.aclose()
    assert b"contexto estourou - tried: vendor/local: busy (1/1 in use)" in body


@respx.mock
async def test_streaming_last_resort_gives_up_at_the_first_event_deadline(monkeypatch):
    # Leituras de `_now()`: prazo montado (0), fatia de espera (0), checagem
    # do laco ja depois do prazo (100) -- desiste sem nenhum keep-alive.
    monkeypatch.setattr(dispatcher, "_now", _FrozenClock(0.0, 0.0, 100.0))
    monkeypatch.setattr(dispatcher, "PING_INTERVAL", 0.001)
    settings = _settings("loc")
    local = respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(200, text=_sse("x"), headers=SSE_HEADERS)
    )
    pool = UpstreamPool(settings)
    held = _occupy(pool, "local")
    try:
        body = await asyncio.wait_for(_collect(settings, pool), 5.0)
        assert pool.in_use("local") == 1
    finally:
        held.release()
        await pool.aclose()
    assert local.call_count == 0
    assert b": ping" not in body
    assert b"vendor/local: busy (1/1 in use); vendor/local: still busy at the deadline" in body


@respx.mock
async def test_streaming_last_resort_keeps_the_client_warm_while_waiting(monkeypatch):
    # prazo (0), fatia (0), checagem (0) -> ping, fatia (0), checagem (0) ->
    # ping, fatia (100, vira espera zero), checagem (100) -> desiste.
    monkeypatch.setattr(dispatcher, "_now", _FrozenClock(0.0, 0.0, 0.0, 0.0, 0.0, 100.0))
    monkeypatch.setattr(dispatcher, "PING_INTERVAL", 0.001)
    settings = _settings("loc")
    pool = UpstreamPool(settings)
    held = _occupy(pool, "local")
    try:
        body = await asyncio.wait_for(_collect(settings, pool), 5.0)
    finally:
        held.release()
        await pool.aclose()
    assert body.startswith(b": ping\n\n: ping\n\n")
    assert body.count(b": ping\n\n") == 2
    assert b"still busy at the deadline" in body


@respx.mock
async def test_streaming_has_no_last_resort_once_a_candidate_was_tried(monkeypatch):
    monkeypatch.setattr(dispatcher, "FIRST_EVENT_DEADLINE", 5.0)
    respx.post(CLOUD_URL).mock(return_value=httpx.Response(400, json={"error": {"message": "n"}}))
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    held = _occupy(pool, "local")
    try:
        body = await asyncio.wait_for(_collect(LOCAL_THEN_CLOUD, pool), 1.0)
    finally:
        held.release()
        await pool.aclose()
    assert b"still busy" not in body
    assert b": ping" not in body


class _FrozenClock:
    """`_now()` controlado: devolve os instantes dados, depois repete o ultimo."""

    def __init__(self, *instants: float) -> None:
        self._instants = list(instants)

    def __call__(self) -> float:
        return self._instants.pop(0) if len(self._instants) > 1 else self._instants[0]


@respx.mock
async def test_a_streaming_last_resort_keeps_its_place_in_line_across_pings(monkeypatch):
    """O revisor mediu: a espera refeita a cada ping entrava no FIM da fila, e
    durante o `yield` do keep-alive nem estava na fila -- um pedido que chegou
    depois levava o slot. A espera tem de ficar na fila entre os pings."""
    monkeypatch.setattr(dispatcher, "_now", _FrozenClock(0.0))
    monkeypatch.setattr(dispatcher, "PING_INTERVAL", 0.001)
    settings = _settings("loc")
    respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(200, text=_sse("do-primeiro"), headers=SSE_HEADERS)
    )
    pool = UpstreamPool(settings)
    held = _occupy(pool, "local")
    first = dispatcher.dispatch_stream(ShuntRequest("anthropic", STREAM_BODY, {}), settings, pool)
    later = None
    try:
        # Primeiro chunk = keep-alive: a espera do ultimo recurso ja comecou e
        # o gerador esta parado no `yield`.
        assert await first.__anext__() == b": ping\n\n"
        later = asyncio.ensure_future(pool.wait_slot("local", 5.0))
        for _ in range(3):
            await asyncio.sleep(0)
        held.release()
        for _ in range(3):
            await asyncio.sleep(0)
        # O slot foi para quem esperava ANTES, nao para quem chegou depois.
        assert not later.done()
        rest = b"".join([chunk async for chunk in first])
        assert b"do-primeiro" in rest
        slot = await asyncio.wait_for(later, 1.0)
        assert slot is not None
        slot.release()
    finally:
        await first.aclose()
        if later is not None and not later.done():
            later.cancel()
        await pool.aclose()


def test_the_busy_line_names_slots_in_use_and_the_limit_separately():
    # No pulo real os dois numeros coincidem (cheio = in_use == limite); o
    # pool falso os separa para que trocar um pelo outro nao passe.
    class _Pool:
        def in_use(self, provider):
            return 3

        def limit(self, provider):
            return 5

    assert dispatcher._busy(_Pool(), "local") == "busy (3/5 in use)"  # type: ignore[arg-type]


async def test_the_buffered_last_resort_waits_only_for_what_is_left_of_the_deadline(monkeypatch):
    from tests.core.test_dispatcher import _Clock

    # Leituras: linha de log (0), prazo montado (0 + 120), ultimo recurso ja
    # em 30 -- sobram 90, nao os 120 do prazo inteiro.
    monkeypatch.setattr(dispatcher, "TOTAL_DEADLINE", 120.0)
    monkeypatch.setattr(dispatcher, "time", _Clock(0.0, 0.0, 30.0))
    settings = _settings("loc")
    pool = UpstreamPool(settings)
    held = _occupy(pool, "local")
    asked: list[float] = []

    async def spy(provider, timeout):
        asked.append(timeout)

    monkeypatch.setattr(pool, "wait_slot", spy)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), settings, pool)
    finally:
        held.release()
        await pool.aclose()
    assert asked == [90.0]
    assert result.trace[-1] == "vendor/local: still busy at the deadline"


@respx.mock
async def test_on_disconnect_the_upstream_response_closes_before_the_slot_returns(monkeypatch):
    """Devolver o slot com a resposta ainda aberta deixa o provedor gerando
    para ninguem enquanto outro pedido ja ocupa o lugar: limite + 1."""
    order: list[str] = []

    class Recorded(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices": [{"delta": {"content": "a"}}]}\n\n'
            yield b'data: {"choices": [{"delta": {"content": "b"}}]}\n\n'

        async def aclose(self) -> None:
            order.append("upstream closed")

    respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, stream=Recorded())
    )
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    real_try_slot = pool.try_slot

    def recording_try_slot(provider):
        slot = real_try_slot(provider)
        if slot is not None:
            release = slot.release

            def recorded_release():
                order.append("slot released")
                release()

            slot.release = recorded_release  # type: ignore[method-assign]
        return slot

    monkeypatch.setattr(pool, "try_slot", recording_try_slot)
    stream = dispatcher.dispatch_stream(
        ShuntRequest("anthropic", STREAM_BODY, {}), LOCAL_THEN_CLOUD, pool
    )
    try:
        await stream.__anext__()
        await stream.aclose()
    finally:
        await pool.aclose()
    assert order[:2] == ["upstream closed", "slot released"]


def _track_clients(monkeypatch) -> list[httpx.AsyncClient]:
    created: list[httpx.AsyncClient] = []
    real = httpx.AsyncClient

    class Tracked(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(httpx, "AsyncClient", Tracked)
    return created


@respx.mock
async def test_a_buffered_last_resort_across_a_reload_leaks_no_client(monkeypatch):
    """O shutdown (`aclose`) chega enquanto um pedido ainda espera o slot. O
    pedido segue sendo servido, mas o cliente httpx que ele usa nao pode
    ficar vivo dentro de um pool que ninguem mais vai fechar."""
    created = _track_clients(monkeypatch)
    settings = _settings("loc")
    respx.post(LOCAL_URL).mock(return_value=httpx.Response(200, json=ok_payload("vendor/local")))
    pool = UpstreamPool(settings)
    held = _occupy(pool, "local")
    task = asyncio.ensure_future(dispatch(ShuntRequest("anthropic", BODY, {}), settings, pool))
    for _ in range(5):
        await asyncio.sleep(0)
    await pool.aclose()  # o reload troca o pool com o pedido na fila
    held.release()
    result = await asyncio.wait_for(task, 5.0)
    assert result.real_model == "vendor/local"
    assert created
    assert all(client.is_closed for client in created)


@respx.mock
async def test_a_streaming_last_resort_across_a_reload_leaks_no_client(monkeypatch):
    created = _track_clients(monkeypatch)
    settings = _settings("loc")
    respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(200, text=_sse("do-local"), headers=SSE_HEADERS)
    )
    pool = UpstreamPool(settings)
    held = _occupy(pool, "local")
    stream = await _enter_last_resort(monkeypatch, settings, pool)
    try:
        await pool.aclose()
        held.release()
        body = b"".join([chunk async for chunk in stream])
    finally:
        await stream.aclose()
    assert b"do-local" in body
    assert created
    assert all(client.is_closed for client in created)


async def test_a_client_lent_by_an_open_pool_is_shared_and_stays_open():
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    try:
        async with pool.client("local", LOCAL_THEN_CLOUD.providers["local"]) as lent:
            assert lent is pool.get("local")
        assert not lent.is_closed
    finally:
        await pool.aclose()


@respx.mock
async def test_a_request_still_on_a_closed_pool_leaks_no_client(monkeypatch):
    # Pedido em voo que chega ao proximo candidato depois do reload: o pool
    # que ele carrega ja foi fechado, e o caminho normal (sem ultimo recurso)
    # tambem nao pode criar cliente orfao nele.
    created = _track_clients(monkeypatch)
    respx.post(LOCAL_URL).mock(return_value=httpx.Response(200, json=ok_payload("vendor/local")))
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    await pool.aclose()
    result = await dispatch(ShuntRequest("anthropic", BODY, {}), LOCAL_THEN_CLOUD, pool)
    assert result.real_model == "vendor/local"
    assert created
    assert all(client.is_closed for client in created)


@respx.mock
async def test_a_stream_still_on_a_closed_pool_leaks_no_client(monkeypatch):
    created = _track_clients(monkeypatch)
    respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(200, text=_sse("do-local"), headers=SSE_HEADERS)
    )
    pool = UpstreamPool(LOCAL_THEN_CLOUD)
    await pool.aclose()
    body = await _collect(LOCAL_THEN_CLOUD, pool)
    assert b"do-local" in body
    assert created
    assert all(client.is_closed for client in created)


@respx.mock
async def test_the_streaming_last_resort_stops_exactly_at_the_deadline(monkeypatch):
    # prazo 0 + 20; fatia (0); checagem exatamente em 20 -> ja venceu, sem ping.
    monkeypatch.setattr(dispatcher, "FIRST_EVENT_DEADLINE", 20.0)
    monkeypatch.setattr(dispatcher, "_now", _FrozenClock(0.0, 0.0, 20.0))
    monkeypatch.setattr(dispatcher, "PING_INTERVAL", 0.001)
    settings = _settings("loc")
    pool = UpstreamPool(settings)
    held = _occupy(pool, "local")
    try:
        body = await asyncio.wait_for(_collect(settings, pool), 5.0)
    finally:
        held.release()
        await pool.aclose()
    assert b": ping" not in body
    assert b"still busy at the deadline" in body


# ---------------------------------------------------------------------------
# Catalogo trocado com o pedido em voo: o pedido segue o snapshot dele
# ---------------------------------------------------------------------------
# O revisor mediu no ramo: o waiter do ultimo recurso liberado por
# `release_all` (provedor removido) levantava `KeyError 'local'`; e um pedido
# com o snapshot antigo mandava a chave antiga ("k") ao host NOVO.

NEW_LOCAL_URL = "http://new-host.test/v1/chat/completions"


def _without_local() -> Settings:
    full = _settings("loc2")
    return Settings(
        providers={k: v for k, v in full.providers.items() if k != "local"},
        models={k: v for k, v in full.models.items() if k != "loc"},
        routes=[("opus", ["loc2"])],
        default_model=None,
    )


def _local_moved() -> Settings:
    moved = _settings("loc").model_copy(deep=True)
    moved.providers["local"].base_url = "http://new-host.test/v1"
    moved.providers["local"].api_key = "NEW-KEY"
    return moved


async def _until_waiting(pool: UpstreamPool, provider: str) -> None:
    for _ in range(50):
        if pool._gates[provider]._waiters:
            return
        await asyncio.sleep(0)
    raise AssertionError("o pedido nao chegou a fila do ultimo recurso")


@respx.mock
async def test_a_last_resort_waiter_released_by_removing_its_provider_is_still_served():
    settings = _settings("loc")
    respx.post(LOCAL_URL).mock(return_value=httpx.Response(200, json=ok_payload("vendor/local")))
    pool = UpstreamPool(settings)
    held = _occupy(pool, "local")
    task = asyncio.ensure_future(dispatch(ShuntRequest("anthropic", BODY, {}), settings, pool))
    try:
        await _until_waiting(pool, "local")
        await pool.update(_without_local())  # release_all entrega o lugar ao waiter
        result = await asyncio.wait_for(task, 5.0)
    finally:
        held.release()
        await pool.aclose()
    assert result.status == 200
    assert result.real_model == "vendor/local"


@respx.mock
async def test_a_request_in_flight_across_a_base_url_and_key_change_keeps_host_and_key_together():
    settings = _settings("loc")
    old = respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(200, json=ok_payload("vendor/local"))
    )
    new = respx.post(NEW_LOCAL_URL).mock(
        return_value=httpx.Response(200, json=ok_payload("vendor/local"))
    )
    pool = UpstreamPool(settings)
    held = _occupy(pool, "local")
    task = asyncio.ensure_future(dispatch(ShuntRequest("anthropic", BODY, {}), settings, pool))
    try:
        await _until_waiting(pool, "local")
        moved = _local_moved()
        await pool.update(moved)
        held.release()  # o lugar passa ao pedido que carrega o snapshot antigo
        result = await asyncio.wait_for(task, 5.0)
        assert result.status == 200
        assert old.call_count == 1
        assert new.call_count == 0
        assert old.calls[0].request.headers["authorization"] == "Bearer k"
        # Pedido novo, snapshot novo: host novo com a chave nova.
        fresh = await dispatch(ShuntRequest("anthropic", BODY, {}), moved, pool)
    finally:
        await pool.aclose()
    assert fresh.status == 200
    assert new.call_count == 1
    assert new.calls[0].request.headers["authorization"] == "Bearer NEW-KEY"
    assert old.call_count == 1


def _moving_while_answering(pool: UpstreamPool, response: httpx.Response):
    """Side effect do candidato anterior da cadeia: o admin troca o catalogo
    enquanto ele responde, e o pedido segue no fallback com o snapshot dele."""

    async def answer(request):
        await pool.update(_local_moved())
        return response

    return answer


@respx.mock
async def test_a_fallback_after_a_base_url_and_key_change_keeps_host_and_key_together():
    settings = _settings("loc2", "loc")
    pool = UpstreamPool(settings)
    respx.post(LOCAL2_URL).mock(
        side_effect=_moving_while_answering(
            pool, httpx.Response(400, json={"error": {"message": "nao"}})
        )
    )
    old = respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(200, json=ok_payload("vendor/local"))
    )
    new = respx.post(NEW_LOCAL_URL).mock(
        return_value=httpx.Response(200, json=ok_payload("vendor/local"))
    )
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), settings, pool)
    finally:
        await pool.aclose()
    assert result.status == 200
    assert new.call_count == 0
    assert old.call_count == 1
    assert old.calls[0].request.headers["authorization"] == "Bearer k"


@respx.mock
async def test_a_streaming_fallback_after_a_base_url_and_key_change_keeps_host_and_key_together():
    settings = _settings("loc2", "loc")
    pool = UpstreamPool(settings)
    respx.post(LOCAL2_URL).mock(
        side_effect=_moving_while_answering(
            pool, httpx.Response(400, json={"error": {"message": "nao"}})
        )
    )
    old = respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(200, text=_sse("do-antigo"), headers=SSE_HEADERS)
    )
    new = respx.post(NEW_LOCAL_URL).mock(
        return_value=httpx.Response(200, text=_sse("do-novo"), headers=SSE_HEADERS)
    )
    try:
        body = await _collect(settings, pool)
    finally:
        await pool.aclose()
    assert b"do-antigo" in body
    assert new.call_count == 0
    assert old.calls[0].request.headers["authorization"] == "Bearer k"


@respx.mock
async def test_a_streaming_last_resort_waiter_released_by_removing_its_provider_is_served(
    monkeypatch,
):
    settings = _settings("loc")
    respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(200, text=_sse("do-local"), headers=SSE_HEADERS)
    )
    pool = UpstreamPool(settings)
    held = _occupy(pool, "local")
    stream = await _enter_last_resort(monkeypatch, settings, pool)
    try:
        await pool.update(_without_local())
        body = b"".join([chunk async for chunk in stream])
    finally:
        await stream.aclose()
        held.release()
        await pool.aclose()
    assert b"do-local" in body


@respx.mock
async def test_a_stream_in_flight_across_a_base_url_and_key_change_keeps_host_and_key_together(
    monkeypatch,
):
    settings = _settings("loc")
    old = respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(200, text=_sse("do-antigo"), headers=SSE_HEADERS)
    )
    new = respx.post(NEW_LOCAL_URL).mock(
        return_value=httpx.Response(200, text=_sse("do-novo"), headers=SSE_HEADERS)
    )
    pool = UpstreamPool(settings)
    held = _occupy(pool, "local")
    stream = await _enter_last_resort(monkeypatch, settings, pool)
    try:
        await pool.update(_local_moved())
        held.release()
        body = b"".join([chunk async for chunk in stream])
    finally:
        await stream.aclose()
        await pool.aclose()
    assert b"do-antigo" in body
    assert new.call_count == 0
    assert old.calls[0].request.headers["authorization"] == "Bearer k"
