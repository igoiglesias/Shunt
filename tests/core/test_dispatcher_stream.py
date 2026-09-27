"""Streaming dispatch: fallback is only free until the first byte is emitted.

Every test here is about that single asymmetry. Before the first valid event
the dispatcher may still change its mind and try the next candidate; after it,
it is committed and an upstream failure becomes an error event on the wire.
"""

import asyncio
import json

import httpx
import pytest
import respx

from app.config.config import MAX_ATTEMPTS, PING_INTERVAL
from app.config.settings import ModelCaps, ModelConfig, ProviderConfig, Settings
from app.core import dispatcher
from app.core.dispatcher import (
    ShuntRequest,
    dispatch_stream,
    is_first_valid_event,
)
from app.core.upstream import UpstreamPool
from app.translate.sse_parse import SSEEvent
from tests.core.test_dispatcher import CLIENT_HEADERS, NO_ANTHROPIC_DECLARED, SETTINGS, TRANSPARENT

BODY = {
    "model": "claude-opus-4-5",
    "max_tokens": 64,
    "stream": True,
    "messages": [{"role": "user", "content": "oi"}],
}

SSE_HEADERS = {"content-type": "text/event-stream"}

ANTHROPIC_SETTINGS = Settings(
    providers={
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test", protocol="anthropic", api_key="sk-teste"
        )
    },
    models={
        "native": ModelConfig(
            provider="anthropic", model="claude-real", context_window=64000, max_output_tokens=8192
        )
    },
    routes=[("opus", ["native"])],
    default_model=None,
)

# Cliente OpenAI, provedor Anthropic: a direcao espelhada.
OPENAI_CLIENT_SETTINGS = Settings(
    providers={
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test", protocol="anthropic", api_key="sk-teste"
        )
    },
    models={
        "native": ModelConfig(
            provider="anthropic", model="claude-real", context_window=64000, max_output_tokens=8192
        )
    },
    routes=[("gpt-4o", ["native"])],
    # "native" amarra o degrau de baixo ao candidato declarado: sem isso,
    # `gpt-*` ganharia o transparente oficial do OpenAI (spec R1) ao fim da
    # cadeia e quebraria a premissa "um unico candidato" destes testes.
    default_model="native",
)

OPENAI_BODY = {
    "model": "gpt-4o",
    "max_tokens": 64,
    "stream": True,
    "messages": [{"role": "user", "content": "oi"}],
}

# Mesmo motivo do override no arquivo de concorrencia: desde o degrau
# transparente oficial (spec R1), um nome `claude-*` com "anthropic" ausente
# do catalogo ganharia um candidato transparente no fim da cadeia, que
# encobriria o comportamento de stream que e o foco daqui. "my-opus" casa na
# rota "opus" e nao deriva provedor, entao a cadeia sai como os testes declaram.
BODY = {**BODY, "model": "my-opus"}

RAW_OPENAI_SETTINGS = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
        )
    },
    models={
        "free": ModelConfig(
            provider="openrouter", model="vendor/free", context_window=64000, max_output_tokens=8192
        )
    },
    routes=[("gpt-4o", ["free"])],
    # Mesmo amarrador do OPENAI_CLIENT_SETTINGS: sem ele o nome `gpt-*`
    # acrescentaria o transparente oficial ao fim da cadeia.
    default_model="free",
)

# Um unico candidato: sem o segundo, uma decisao errada do laco aparece como
# erro no cliente em vez de ser mascarada por um fallback que da certo.
SOLO_SETTINGS = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
        )
    },
    models={
        "free": ModelConfig(
            provider="openrouter", model="vendor/free", context_window=64000, max_output_tokens=8192
        )
    },
    routes=[("opus", ["free"])],
    default_model=None,
)


def sse(*payloads):
    return "".join(f"data: {p}\n\n" for p in payloads) + "data: [DONE]\n\n"


class Chunks(httpx.AsyncByteStream):
    """Stream de bytes com fronteiras que o teste escolhe, e que registra o
    fechamento -- e disso que depende o teste de desconexao do cliente."""

    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class Failing(httpx.AsyncByteStream):
    """Provedor que entrega alguns chunks e entao morre no meio do stream --
    `ReadTimeout` apos 60s de silencio, `RemoteProtocolError` quando a conexao
    cai. A resposta ja existe, entao o `except` do `send` nao cobre isso."""

    def __init__(self, *chunks: bytes, error: Exception) -> None:
        self._chunks = chunks
        self._error = error
        self.closed = False

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
        raise self._error

    async def aclose(self) -> None:
        self.closed = True


class FakeClock:
    """Relogio controlado: cada leitura consome o proximo instante da lista e,
    esgotada, repete o ultimo. Um teste de relogio real passaria por acaso."""

    def __init__(self, *instants: float) -> None:
        self._instants = list(instants)
        self.last = self._instants[0]

    def __call__(self) -> float:
        if self._instants:
            self.last = self._instants.pop(0)
        return self.last


async def collect(req, settings, pool) -> bytes:
    return b"".join([chunk async for chunk in dispatch_stream(req, settings, pool)])


async def run(req, settings) -> bytes:
    pool = UpstreamPool(settings)
    try:
        return await collect(req, settings, pool)
    finally:
        await pool.aclose()


def events_of(body: str) -> list[str]:
    return [
        line.removeprefix("event: ") for line in body.splitlines() if line.startswith("event: ")
    ]


# --------------------------------------------------------------------------
# is_first_valid_event
# --------------------------------------------------------------------------


def test_keepalive_and_error_do_not_count_as_first_event():
    assert is_first_valid_event({"choices": [{"delta": {"content": "oi"}}]}) is True
    assert is_first_valid_event({"choices": [{"delta": {}}]}) is False
    assert is_first_valid_event({"error": {"message": "cheio"}}) is False


def test_a_tool_call_delta_is_a_valid_first_event_and_an_empty_stream_is_not():
    assert is_first_valid_event({"choices": [{"delta": {"tool_calls": [{"index": 0}]}}]}) is True
    assert is_first_valid_event({"choices": []}) is False
    assert is_first_valid_event({}) is False
    assert is_first_valid_event({"choices": [{}]}) is False
    # Um delta com conteudo vazio e um keep-alive disfarcado, nao um comeco.
    assert is_first_valid_event({"choices": [{"delta": {"content": ""}}]}) is False
    # `error` vence mesmo quando ha conteudo junto.
    assert (
        is_first_valid_event({"error": {"message": "x"}, "choices": [{"delta": {"content": "oi"}}]})
        is False
    )


# --------------------------------------------------------------------------
# Direcao Anthropic-cliente / OpenAI-provedor
# --------------------------------------------------------------------------


@respx.mock
async def test_error_inside_a_200_stream_falls_back_to_the_next_candidate():
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            # O erro vem no primeiro chunk e conteudo valido no segundo: o
            # candidato ja esta descartado, e nada dele pode vazar para o cliente.
            httpx.Response(
                200,
                headers=SSE_HEADERS,
                stream=Chunks(
                    b'data: {"error": {"message": "sem credito"}}\n\n'
                    b'data: {"choices": [{"delta": {"content": "depois"}}]}\n\n',
                    b'data: {"choices": [{"delta": {"content": "depois2"}}]}\n\n',
                ),
            ),
            httpx.Response(
                200, text=sse('{"choices": [{"delta": {"content": "ok"}}]}'), headers=SSE_HEADERS
            ),
        ]
    )
    body = await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)
    assert b"message_start" in body
    assert b"sem credito" not in body
    assert b"depois" not in body
    assert b"depois2" not in body
    assert b'"text": "ok"' in body or b'"text":"ok"' in body


@respx.mock
async def test_stream_is_translated_into_anthropic_events_in_order():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            text=sse(
                '{"choices": [{"delta": {"content": "oi"}}]}',
                '{"choices": [{"delta": {}, "finish_reason": "stop"}]}',
            ),
        )
    )
    # O nome pedido e o da era pre-transparente: o override "my-opus" do
    # arquivo quebraria a verificacao de que o cliente ve o modelo que PEDIU.
    # O primeiro candidato atende, entao o degrau transparente oficial que o
    # nome `claude-*` acrescenta ao fim nunca e tocado.
    body = (await run(ShuntRequest("anthropic", {**BODY, "model": "claude-opus-4-5"}, {}), SETTINGS)).decode()
    assert events_of(body) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start = json.loads(body.splitlines()[1].removeprefix("data: "))
    # O cliente ve o modelo que PEDIU, nunca o do provedor que atendeu.
    assert start["message"]["model"] == "claude-opus-4-5"
    # O sentinela `[DONE]` e do dialeto OpenAI: um cliente Anthropic nao o recebe.
    assert "[DONE]" not in body
    # Cada evento carrega os dois campos; so `event:` nao e um SSE util.
    assert body.count("data: ") == 6


@respx.mock
async def test_error_after_the_first_event_is_reported_not_retried():
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            stream=Chunks(
                b'data: {"choices": [{"delta": '
                b'{"content": "a"}}]}\n\n'
                b'data: {"error": {"message": "caiu"}}\n\n'
                b'data: {"choices": [{"delta": '
                b'{"content": "depois"}}]}\n\n',
                b'data: {"choices": [{"delta": {"content": "depois2"}}]}\n\n',
            ),
        )
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert route.call_count == 1
    assert "event: error" in body
    assert "caiu" in body
    # Comprometido: o fecho normal nao sai depois do erro, e nada do que o
    # provedor mandou depois dele chega ao cliente.
    assert "message_stop" not in body
    assert "depois" not in body
    assert "depois2" not in body


@respx.mock
async def test_a_fatal_in_band_error_is_logged_with_the_failure_status(caplog):
    """O erro `{"error": ...}` sai DENTRO do stream em 200 do provedor: registrar
    a linha do banco como 200 sem error_type seria declarar sucesso sobre um
    stream que o cliente viu morrer com `event: error` (502/api_error)."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            stream=Chunks(
                b'data: {"choices": [{"delta": {"content": "comecou"}}]}\n\n',
                b'data: {"error": {"message": "caiu no meio"}}\n\n',
            ),
        )
    )
    pool = UpstreamPool(SETTINGS)
    stream = dispatch_stream(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    with caplog.at_level("INFO", logger="shunt"):
        await collect_into(stream)
    await pool.aclose()
    linhas = [json.loads(r.message) for r in caplog.records if r.name == "shunt"]
    assert linhas, "o stream com erro in-band nao foi registrado"
    linha = linhas[-1]
    assert linha["status"] == 502
    # O mesmo tipo que o `event: error` que o cliente recebeu: 502 no dialeto
    # Anthropic e `api_error`.
    assert linha["error_type"] == "api_error"


@respx.mock
async def test_a_successful_stream_is_still_logged_as_success(caplog):
    """O oposto do erro in-band: o stream terminou com `message_stop` e o banco
    continua dizendo 200 sem error_type. E o que segura o `if state.erro_em_band`
    do fix de integridade -- sem esta linha, mover as duas atribuicoes de 502
    para FORA do `if` deixa a suíte verde e todo stream vira falha no painel."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            text=sse('{"choices": [{"delta": {"content": "oi"}}]}'),
        )
    )
    pool = UpstreamPool(SETTINGS)
    stream = dispatch_stream(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    with caplog.at_level("INFO", logger="shunt"):
        await collect_into(stream)
    await pool.aclose()
    linhas = [json.loads(r.message) for r in caplog.records if r.name == "shunt"]
    assert linhas, "o stream bem-sucedido nao foi registrado"
    linha = linhas[-1]
    assert linha["status"] == 200
    assert linha["error_type"] is None
    assert linha["candidate"] == SETTINGS.models["free"].model


@respx.mock
async def test_chunks_before_the_first_valid_event_do_not_consume_message_start():
    """Um delta vazio antes do conteudo nao pode ser entregue ao tradutor: ele
    marcaria `message_start` como ja emitido e o cliente nunca o veria."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            text=sse('{"choices": [{"delta": {}}]}', '{"choices": [{"delta": {"content": "oi"}}]}'),
        )
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert events_of(body)[0] == "message_start"
    assert events_of(body).count("message_start") == 1


@respx.mock
async def test_the_message_id_is_unique_per_stream():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "oi"}}]}')
        )
    )
    first = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    second = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()

    def message_id(body: str) -> str:
        line = next(ln for ln in body.splitlines() if ln.startswith("data: "))
        return json.loads(line.removeprefix("data: "))["message"]["id"]

    assert message_id(first).startswith("msg_")
    assert message_id(first) != message_id(second)


# --------------------------------------------------------------------------
# Mesmo protocolo dos dois lados: bytes crus
# --------------------------------------------------------------------------


@respx.mock
async def test_same_protocol_on_both_sides_forwards_the_raw_bytes():
    raw = (
        b": keep-alive\n\n"
        b'event: message_start\ndata: {"type": "message_start", "x": 1}\n\n'
        b"event: dialeto_novo\ndata: nao e json\n\n"
    )
    respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, stream=Chunks(raw))
    )
    body = await run(ShuntRequest("anthropic", BODY, {}), ANTHROPIC_SETTINGS)
    # Byte a byte: nem o comentario, nem o evento desconhecido, nem o `data:`
    # que nao e JSON passaram por tradutor nenhum.
    assert body == raw


@respx.mock
async def test_transparent_mode_streams_raw_bytes_with_the_clients_own_headers():
    """Combination Task 19's review flagged as untested: a transparent
    candidate (headers forwarded verbatim, no alias/config of ours) going
    through the STREAMING path, not the buffered one this is already
    covered for (test_transparent_mode_passes_inbound_headers_verbatim_minus_three).
    `TRANSPARENT` has no routes and no default_model, so `resolve()` falls
    through to `_transparent()`, and the client's own protocol matches the
    provider's, so this also exercises the raw-passthrough branch."""
    raw = b'event: message_start\ndata: {"type": "message_start"}\n\ndata: [DONE]\n\n'
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, stream=Chunks(raw))
    )
    # "my-opus" do override nao deriva provedor e derrubaria o teste em
    # `UnknownProviderError`: o nome original `claude-*` e o que cai no
    # transparente.
    body = await run(
        ShuntRequest("anthropic", {**BODY, "model": "claude-opus-4-5"}, dict(CLIENT_HEADERS)),
        TRANSPARENT,
    )
    assert body == raw
    sent = route.calls[0].request.headers
    # O cabecalho do cliente vai embora tal como chegou -- nenhum dos tres
    # nomes que o modo transparente descarta, e a credencial do cliente
    # intacta, nunca a nossa.
    assert sent["authorization"] == "Bearer oauth-da-assinatura"
    assert sent["x-api-key"] == "sk-do-cliente"
    assert sent["anthropic-beta"] == "oauth-2026-01-01"
    assert "host" not in sent or sent["host"] != "localhost:8080"


# --------------------------------------------------------------------------
# Direcao OpenAI-cliente / Anthropic-provedor
# --------------------------------------------------------------------------


@respx.mock
async def test_openai_client_gets_data_only_lines_and_a_done_sentinel():
    upstream = (
        'event: message_start\ndata: {"type": "message_start"}\n\n'
        'event: ping\ndata: {"type": "ping"}\n\n'
        'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, '
        '"content_block": {"type": "text", "text": ""}}\n\n'
        'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, '
        '"delta": {"type": "text_delta", "text": "ok"}}\n\n'
        'event: message_delta\ndata: {"type": "message_delta", '
        '"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}\n\n'
    )
    respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, text=upstream)
    )
    body = (await run(ShuntRequest("openai", OPENAI_BODY, {}), OPENAI_CLIENT_SETTINGS)).decode()
    lines = [ln for ln in body.splitlines() if ln]
    # Nenhuma linha `event:`: o dialeto OpenAI so tem `data:`.
    assert all(ln.startswith("data: ") for ln in lines), lines
    assert lines[-1] == "data: [DONE]"
    assert '"content": "ok"' in body
    assert '"finish_reason": "stop"' in body
    # `message_start` e `ping` nao viram chunk: so o delta e o fecho saem.
    assert len(lines) == 3
    # Cada `data:` e um evento SSE fechado por linha em branco.
    assert body.count("\n\n") == 3
    first = json.loads(lines[0].removeprefix("data: "))
    assert first["id"].startswith("chatcmpl-")
    assert first["object"] == "chat.completion.chunk"
    assert first["model"] == "gpt-4o"


@respx.mock
async def test_openai_client_falls_back_when_the_anthropic_stream_errors_first():
    respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            text='event: error\ndata: {"type": "error", "error": {"message": "sobrecarga"}}\n\n',
        )
    )
    body = (await run(ShuntRequest("openai", OPENAI_BODY, {}), OPENAI_CLIENT_SETTINGS)).decode()
    lines = [ln for ln in body.splitlines() if ln]
    assert all(ln.startswith("data: ") for ln in lines), lines
    # Sem candidato seguinte, o erro final sai no formato OpenAI (`error` na
    # raiz), nunca como `event: error` do dialeto Anthropic.
    assert "event: error" not in body
    payload = json.loads(lines[0].removeprefix("data: "))
    # So a chave `error` na raiz -- o envelope `{"type": "error", ...}` da
    # Anthropic nao pode vazar para um cliente OpenAI.
    assert set(payload) == {"error"}
    # O tipo tambem e do dialeto OpenAI: `server_error`, e nao o `api_error`
    # que a tabela Anthropic daria para o mesmo 502.
    assert payload["error"]["type"] == "server_error"
    assert set(payload["error"]) == {"message", "type", "param", "code"}
    assert "sobrecarga" in payload["error"]["message"]
    assert lines[-1] == "data: [DONE]"


# --------------------------------------------------------------------------
# Deadline do primeiro evento e ping
# --------------------------------------------------------------------------


@respx.mock
async def test_first_event_deadline_gives_the_next_candidate_a_turn(monkeypatch):
    content = b'data: {"choices": [{"delta": {"content": "tarde"}}]}\n\n'
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, headers=SSE_HEADERS, stream=Chunks(b": esperando\n\n", content)),
            httpx.Response(
                200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
            ),
        ]
    )
    # 20.1 > FIRST_EVENT_DEADLINE (20.0): o candidato e abandonado antes de
    # processar o chunk que chegou tarde demais.
    monkeypatch.setattr(dispatcher, "_now", FakeClock(1000.0, 1001.0, 1020.1, 2000.0, 2000.1))
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert "tarde" not in body
    assert '"ok"' in body or '"text": "ok"' in body


@respx.mock
async def test_a_first_event_exactly_at_the_deadline_still_counts(monkeypatch):
    content = b'data: {"choices": [{"delta": {"content": "tarde"}}]}\n\n'
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, stream=Chunks(b": esperando\n\n", content)
        )
    )
    # Exatamente 20.0 nao estoura: o corte e estritamente maior.
    monkeypatch.setattr(dispatcher, "_now", FakeClock(1000.0, 1001.0, 1020.0))
    body = (await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)).decode()
    assert "tarde" in body


@respx.mock
async def test_a_keepalive_is_emitted_only_after_the_interval_while_waiting(monkeypatch):
    content = b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n'
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, stream=Chunks(b": um\n\n", b": dois\n\n", content)
        )
    )
    # Exatamente PING_INTERVAL nao dispara (o corte e estrito), o dobro dispara.
    # Com `>=` o primeiro chunk ja pingaria e o segundo pingaria de novo: dois.
    monkeypatch.setattr(
        dispatcher,
        "_now",
        FakeClock(1000.0, 1000.0 + PING_INTERVAL, 1000.0 + 2 * PING_INTERVAL, 1010.1),
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)).decode()
    # An Anthropic client never sees a `ping` EVENT ahead of `message_start`:
    # the keepalive is a bare SSE comment (no `data:` field), exactly like
    # the OpenAI client already got below. A comment is not an event at all,
    # so `events_of` (which only counts `event:` lines) sees none of it.
    assert events_of(body).count("ping") == 0
    assert body.count(": ping\n\n") == 1
    assert body.index(": ping\n\n") < body.index("event: message_start")


@respx.mock
async def test_the_keepalive_for_an_openai_client_is_a_comment_not_an_event(monkeypatch):
    content = (
        'event: content_block_delta\ndata: {"type": "content_block_delta", '
        '"index": 0, "delta": {"type": "text_delta", "text": "ok"}}\n\n'
    )
    respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, stream=Chunks(b": um\n\n", content.encode())
        )
    )
    monkeypatch.setattr(dispatcher, "_now", FakeClock(1000.0, 1005.1, 1005.2))
    body = (await run(ShuntRequest("openai", OPENAI_BODY, {}), OPENAI_CLIENT_SETTINGS)).decode()
    assert body.startswith(": ping\n\n")
    assert "event: ping" not in body


async def test_the_clock_helper_reads_the_real_monotonic_clock():
    import time

    assert abs(dispatcher._now() - time.monotonic()) < 1.0


# --------------------------------------------------------------------------
# Desconexao do cliente
# --------------------------------------------------------------------------


@respx.mock
async def test_client_disconnect_closes_the_upstream_response():
    stream = Chunks(
        b'data: {"choices": [{"delta": {"content": "a"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "b"}}]}\n\n',
    )
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, stream=stream)
    )
    pool = UpstreamPool(SETTINGS)
    generator = dispatch_stream(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    try:
        await generator.__anext__()
        assert stream.closed is False
        await generator.aclose()
        assert stream.closed is True
    finally:
        await pool.aclose()


# --------------------------------------------------------------------------
# Falhas antes do corpo: status, transporte, chain
# --------------------------------------------------------------------------


@respx.mock
async def test_a_4xx_falls_back_to_the_next_candidate_without_leaking_its_body():
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(402, json={"error": {"message": "sem saldo"}}),
            httpx.Response(
                200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
            ),
        ]
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert "sem saldo" not in body
    assert "message_start" in body


@respx.mock
async def test_a_retryable_status_is_retried_on_the_same_candidate_before_any_byte(monkeypatch):
    """SOLO_SETTINGS has exactly one candidate: there is no next one to fall
    back to. If the request below succeeds at all, it can only be because the
    503 arriving in the response headers -- before any byte of the body
    reached us -- was retried on the SAME candidate, not because a fallback
    masked it. That is what structurally separates this from the 4xx test
    above, which relies on a second candidate existing."""
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    route = respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(503, json={"error": {"message": "sobrecarregado"}}),
            httpx.Response(
                200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
            ),
        ]
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)).decode()
    assert "message_start" in body
    assert route.call_count == 2


@respx.mock
async def test_a_non_retryable_status_is_not_retried_on_the_same_candidate(monkeypatch):
    """Same single-candidate setup, but a 400 is not in `classify`'s retry
    set. With no second candidate to fall back to, the request must fail
    outright and the mock must be hit exactly once -- proving the loop did
    NOT retry a status it has no business retrying."""
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "requisicao invalida"}}),
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)).decode()
    assert events_of(body) == ["error"]
    assert route.call_count == 1


@respx.mock
async def test_a_retryable_status_persisting_past_max_attempts_still_fails(monkeypatch):
    """The retry is bounded by MAX_ATTEMPTS, same as the buffered path: a
    503 on every attempt exhausts the single candidate and reports an error,
    it does not retry forever."""
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(503, json={"error": {"message": "sobrecarregado"}}),
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)).decode()
    assert events_of(body) == ["error"]
    assert route.call_count == MAX_ATTEMPTS


@respx.mock
async def test_a_transport_error_falls_back_to_the_next_candidate(monkeypatch):
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            *[httpx.ConnectError("recusou")] * MAX_ATTEMPTS,
            httpx.Response(
                200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
            ),
        ]
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert "message_start" in body


@respx.mock
async def test_every_candidate_failing_yields_one_error_event_with_the_trace(monkeypatch):
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "x"}}),
            *[httpx.ConnectError("recusou")] * MAX_ATTEMPTS,
        ]
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert events_of(body) == ["error"]
    # 400 e a fronteira: status abaixo dela seguem para o corpo.
    assert "free: 400" in body
    assert "cheap: ConnectError: recusou" in body
    assert "ConnectError: recusou - tried:" in body


@respx.mock
async def test_a_stream_that_ends_before_any_valid_event_falls_back():
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, headers=SSE_HEADERS, text=": so keep-alive\n\ndata: [DONE]\n\n"),
            httpx.Response(
                200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
            ),
        ]
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert "message_start" in body
    assert '"text": "ok"' in body


@respx.mock
async def test_a_candidate_whose_protocol_has_no_such_endpoint_is_skipped():
    settings = Settings(
        providers={
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.test", protocol="anthropic", api_key="sk-teste"
            ),
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
            ),
        },
        models={
            "native": ModelConfig(
                provider="anthropic",
                model="claude-real",
                context_window=64000,
                max_output_tokens=8192,
            ),
            "free": ModelConfig(
                provider="openrouter",
                model="vendor/free",
                context_window=64000,
                max_output_tokens=8192,
            ),
        },
        routes=[("gpt-4o", ["native", "free"])],
        default_model=None,
    )
    route = respx.post("https://api.test/v1/completions").mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
        )
    )
    body = (await run(ShuntRequest("openai", OPENAI_BODY, {}, "completions"), settings)).decode()
    assert route.call_count == 1
    assert '"content": "ok"' in body


async def test_an_empty_chain_is_reported_before_any_request_is_made():
    settings = Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
            )
        },
        models={
            "mudo": ModelConfig(
                provider="openrouter",
                model="vendor/mudo",
                supports=ModelCaps(streaming=False),
                context_window=64000,
                max_output_tokens=8192,
            )
        },
        routes=[("opus", ["mudo"])],
        default_model=None,
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), settings)).decode()
    assert events_of(body) == ["error"]
    assert "vendor/mudo: no streaming support" in body
    assert "no candidate can serve this request" in body


@respx.mock
async def test_a_candidate_whose_payload_cannot_be_rendered_is_skipped(monkeypatch):
    real = dispatcher._payload
    calls: list[str] = []

    def explode(req, candidate, settings):
        calls.append(candidate.model)
        if candidate.model == "vendor/free":
            raise ValueError("shape estranho")
        return real(req, candidate, settings)

    monkeypatch.setattr(dispatcher, "_payload", explode)
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
        )
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert "vendor/cheap" in calls
    assert "message_start" in body
    assert '"text": "ok"' in body


# --------------------------------------------------------------------------
# `_handle_event`: o que nao e um evento utilizavel
# --------------------------------------------------------------------------


@pytest.mark.parametrize("data", ["[DONE]", "  [DONE]  ", "nao e json", '"um texto"', "[1, 2]"])
def test_unusable_event_payloads_emit_nothing_and_decide_nothing(data):
    req = ShuntRequest("anthropic", BODY, {})
    translator = dispatcher._stream_translator(req)
    handled = dispatcher._handle_event(req, translator, SSEEvent(None, data), started=False)
    assert handled.payloads == []
    assert handled.started is False
    assert handled.failed is None
    assert handled.fatal is False


def test_an_unusable_payload_after_the_start_does_not_undo_the_decision():
    req = ShuntRequest("anthropic", BODY, {})
    translator = dispatcher._stream_translator(req)
    handled = dispatcher._handle_event(req, translator, SSEEvent(None, "[DONE]"), started=True)
    assert handled.started is True


def test_finish_emits_the_openai_translators_trailing_chunks_as_data_lines():
    """`AnthropicStreamToOpenAI.finish()` is a documented no-op today; the
    dispatcher still has to render whatever it returns, or a later flush would
    be dropped silently."""

    class Trailing:
        def finish(self):
            return [{"id": "chatcmpl-x", "choices": []}]

    out = list(dispatcher._finish(ShuntRequest("openai", OPENAI_BODY, {}), Trailing()))
    assert out == [b'data: {"id": "chatcmpl-x", "choices": []}\n\n']


@respx.mock
async def test_an_empty_text_delta_does_not_commit_the_openai_direction():
    """A translated chunk whose content is empty is the mirror of an OpenAI
    keep-alive: it must not close the decision either."""
    upstream = (
        'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, '
        '"delta": {"type": "text_delta", "text": ""}}\n\n'
        'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, '
        '"delta": {"type": "text_delta", "text": "ok"}}\n\n'
    )
    respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, text=upstream)
    )
    body = (await run(ShuntRequest("openai", OPENAI_BODY, {}), OPENAI_CLIENT_SETTINGS)).decode()
    lines = [ln for ln in body.splitlines() if ln]
    assert len(lines) == 2
    assert '"content": "ok"' in lines[0]
    assert lines[1] == "data: [DONE]"
    assert '"content": ""' not in body


@respx.mock
async def test_after_the_first_event_the_deadline_no_longer_applies(monkeypatch):
    """The deadline bounds the WAIT, not the stream. A long answer whose later
    chunks arrive well past it must be delivered whole."""
    stream = Chunks(
        b'data: {"choices": [{"delta": {"content": "a"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "b"}}]}\n\n',
    )
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, stream=stream)
    )
    monkeypatch.setattr(dispatcher, "_now", FakeClock(1000.0, 1001.0, 1100.0))
    body = (await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)).decode()
    assert '"text": "a"' in body
    assert '"text": "b"' in body
    assert "event: ping" not in body
    assert events_of(body)[-1] == "message_stop"


@respx.mock
async def test_non_ascii_content_is_not_escaped_on_the_wire():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ação"}}]}')
        )
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert '"ação"' in body
    assert "\\u" not in body


@respx.mock
async def test_a_finish_reason_after_the_start_reaches_the_translator():
    """The chunk that carries `finish_reason` carries no content, so it is not
    a valid FIRST event -- but once started it must still be translated, or the
    stop reason silently degrades to the default."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            text=sse(
                '{"choices": [{"delta": {"content": "oi"}}]}',
                '{"choices": [{"delta": {}, "finish_reason": "length"}]}',
            ),
        )
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert '"stop_reason": "max_tokens"' in body


@respx.mock
async def test_an_event_split_across_two_chunks_is_decoded_as_one():
    """One decoder for the whole response, not one per read: an SSE event has
    nothing to do with a TCP chunk boundary."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            stream=Chunks(b'data: {"choices": [{"delta": {"con', b'tent": "partido"}}]}\n\n'),
        )
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert '"text": "partido"' in body


@respx.mock
async def test_the_in_stream_error_message_reaches_the_final_trace():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, text='data: {"error": {"message": "sem credito"}}\n\n'
        )
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)).decode()
    assert events_of(body) == ["error"]
    # Sem candidato seguinte, o erro que estava DENTRO do stream e exatamente
    # o que o cliente precisa ler -- e o trace tem que nomear o candidato.
    message = json.loads(body.splitlines()[1].removeprefix("data: "))["error"]["message"]
    assert message.startswith('{"message": "sem credito"} - tried: ')
    assert message.endswith('free: {"message": "sem credito"}')


@respx.mock
async def test_the_deadline_message_names_the_deadline(monkeypatch):
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, stream=Chunks(b": um\n\n", b": dois\n\n")
        )
    )
    monkeypatch.setattr(dispatcher, "_now", FakeClock(1000.0, 1001.0, 1020.1))
    body = (await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)).decode()
    assert "free: no valid event within 20.0s" in body
    assert "no valid event within 20.0s - tried:" in body


@respx.mock
async def test_a_stream_that_ends_empty_says_so_in_the_trace():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, text="data: [DONE]\n\n")
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)).decode()
    assert "free: stream ended before the first valid event" in body
    assert "stream ended before the first valid event - tried:" in body


@respx.mock
async def test_the_last_upstream_status_message_is_the_one_reported(monkeypatch):
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            *[httpx.ConnectError("recusou")] * MAX_ATTEMPTS,
            httpx.Response(400, json={"error": {"message": "sem saldo"}}),
        ]
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert "free: ConnectError: recusou" in body
    assert "cheap: 400" in body
    # A mensagem final e a do ULTIMO candidato, extraida do corpo do erro.
    assert "sem saldo - tried:" in body
    assert '"type": "api_error"' in body


@respx.mock
async def test_no_candidate_supports_the_endpoint_at_all():
    settings = Settings(
        providers={
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.test", protocol="anthropic", api_key="sk-teste"
            )
        },
        models={
            "native": ModelConfig(
                provider="anthropic",
                model="claude-real",
                context_window=64000,
                max_output_tokens=8192,
            )
        },
        routes=[("gpt-4o", ["native"])],
        # Sem default o nome `gpt-*` ganharia o transparente oficial do
        # OpenAI (spec R1), que APOIA embeddings: o teste cobrira "nenhum
        # candidato suporta o endpoint" com os candidatos declarados.
        default_model="native",
    )
    body = (await run(ShuntRequest("openai", OPENAI_BODY, {}, "embeddings"), settings)).decode()
    assert "claude-real: endpoint not supported" in body
    assert "no candidate answered - tried:" in body


async def test_a_probe_that_cannot_be_rendered_is_recorded_before_the_filter(monkeypatch):
    """The probe is the FIRST candidate's payload, and its failure is the only
    evidence left once the capability filter empties the chain."""
    settings = Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
            )
        },
        models={
            "free": ModelConfig(
                provider="openrouter",
                model="vendor/free",
                supports=ModelCaps(streaming=False),
                context_window=64000,
                max_output_tokens=8192,
            ),
            "cheap": ModelConfig(
                provider="openrouter",
                model="vendor/cheap",
                supports=ModelCaps(streaming=False),
                context_window=64000,
                max_output_tokens=8192,
            ),
        },
        routes=[("opus", ["free", "cheap"])],
        default_model=None,
    )
    real = dispatcher._payload

    def explode(req, candidate, settings):
        if candidate.model == "vendor/free":
            raise ValueError("shape estranho")
        return real(req, candidate, settings)

    monkeypatch.setattr(dispatcher, "_payload", explode)
    body = (await run(ShuntRequest("anthropic", BODY, {}), settings)).decode()
    assert "probe (vendor/free): request translation failed: shape estranho" in body
    # O corpo cru substitui o probe, entao `stream: true` continua sendo
    # exigido e os dois candidatos mudos caem.
    assert "vendor/free: no streaming support" in body
    assert "vendor/cheap: no streaming support" in body


@respx.mock
async def test_every_candidate_failing_to_render_is_named_in_the_trace(monkeypatch):
    monkeypatch.setattr(
        dispatcher,
        "_payload",
        lambda req, candidate, settings: (_ for _ in ()).throw(ValueError("shape estranho")),
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert "free: request translation failed: shape estranho" in body
    assert "cheap: request translation failed: shape estranho" in body
    assert "no candidate answered - tried:" in body


@respx.mock
async def test_the_outbound_request_carries_our_credential_and_the_translated_body(monkeypatch):
    settings = Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key="segredo"
            )
        },
        models={
            "free": ModelConfig(
                provider="openrouter",
                model="vendor/free",
                context_window=64000,
                max_output_tokens=8192,
            )
        },
        routes=[("opus", ["free"])],
        default_model=None,
    )
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
        )
    )
    await run(ShuntRequest("anthropic", BODY, {}), settings)
    request = route.calls[0].request
    assert request.method == "POST"
    assert request.headers["authorization"] == "Bearer segredo"
    sent = json.loads(request.content)
    # O corpo que sobe e o TRADUZIDO, com o modelo real do candidato.
    assert sent["model"] == "vendor/free"
    assert sent["stream"] is True
    assert sent["messages"] == [{"role": "user", "content": "oi"}]


@respx.mock
async def test_client_disconnect_closes_a_raw_passthrough_too():
    stream = Chunks(
        b'event: message_start\ndata: {"type": "message_start"}\n\n',
        b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
    )
    respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, stream=stream)
    )
    pool = UpstreamPool(ANTHROPIC_SETTINGS)
    generator = dispatch_stream(ShuntRequest("anthropic", BODY, {}), ANTHROPIC_SETTINGS, pool)
    try:
        await generator.__anext__()
        assert stream.closed is False
        await generator.aclose()
        assert stream.closed is True
    finally:
        await pool.aclose()


@respx.mock
async def test_an_empty_delta_before_an_error_still_leaves_the_fallback_open():
    """The chunk that opens the message is the one that commits us. A provider
    that sends an empty delta and only then fails has committed nothing, so the
    next candidate must still get its turn."""
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                headers=SSE_HEADERS,
                text='data: {"choices": [{"delta": {}}]}\n\n'
                'data: {"error": {"message": "sem credito"}}\n\n',
            ),
            httpx.Response(
                200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
            ),
        ]
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert "sem credito" not in body
    assert '"text": "ok"' in body
    assert events_of(body)[-1] == "message_stop"


# --------------------------------------------------------------------------
# Falha no MEIO do stream: a resposta ja existe, o `except` do `send` nao pega
# --------------------------------------------------------------------------


@respx.mock
async def test_a_mid_stream_failure_after_the_start_is_reported_and_closed():
    """Depois do primeiro evento nao ha fallback: o que o cliente precisa e o
    erro E um fecho bem formado, senao o stream fica truncado e o parser dele
    nunca solta o buffer."""
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            stream=Failing(
                b'data: {"choices": [{"delta": {"content": "comecou"}}]}\n\n',
                error=httpx.ReadTimeout("tempo esgotado"),
            ),
        )
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert route.call_count == 1
    assert '"text": "comecou"' in body
    assert "event: error" in body
    assert "tempo esgotado" in body
    # O fecho sai DEPOIS do erro: `finish()` e idempotente exatamente por isso.
    assert events_of(body)[-1] == "message_stop"
    assert body.index("event: error") < body.index("event: message_delta")


@respx.mock
async def test_a_mid_stream_failure_before_the_start_falls_back():
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                headers=SSE_HEADERS,
                stream=Failing(b": esperando\n\n", error=httpx.RemoteProtocolError("conexao caiu")),
            ),
            httpx.Response(
                200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
            ),
        ]
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert "conexao caiu" not in body
    assert "event: error" not in body
    assert '"text": "ok"' in body
    assert events_of(body)[-1] == "message_stop"


@respx.mock
async def test_a_mid_stream_failure_before_the_start_is_named_in_the_trace():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            stream=Failing(b": esperando\n\n", error=httpx.RemoteProtocolError("conexao caiu")),
        )
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)).decode()
    assert "free: RemoteProtocolError: conexao caiu" in body
    assert "RemoteProtocolError: conexao caiu - tried:" in body


@respx.mock
async def test_a_mid_stream_failure_closes_the_upstream_response():
    stream = Failing(
        b'data: {"choices": [{"delta": {"content": "a"}}]}\n\n',
        error=httpx.ReadTimeout("tempo esgotado"),
    )
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, stream=stream)
    )
    await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)
    assert stream.closed is True


@respx.mock
async def test_a_mid_stream_failure_for_an_openai_client_keeps_the_dialect():
    upstream = (
        'event: content_block_delta\ndata: {"type": "content_block_delta", '
        '"index": 0, "delta": {"type": "text_delta", "text": "ok"}}\n\n'
    )
    respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            stream=Failing(upstream.encode(), error=httpx.ReadTimeout("tempo esgotado")),
        )
    )
    body = (await run(ShuntRequest("openai", OPENAI_BODY, {}), OPENAI_CLIENT_SETTINGS)).decode()
    lines = [ln for ln in body.splitlines() if ln]
    assert all(ln.startswith("data: ") for ln in lines), lines
    assert "tempo esgotado" in body
    assert lines[-1] == "data: [DONE]"


# --------------------------------------------------------------------------
# `SSEDecoder.flush()`: o provedor que fecha sem a linha em branco final
# --------------------------------------------------------------------------


@respx.mock
async def test_a_last_event_without_its_blank_line_is_still_delivered():
    """Sem `flush()` o evento nao e so perdido: `started` fica False, o
    candidato e anotado como vazio e a chain e queimada num provedor que
    estava funcionando."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, text='data: {"choices": [{"delta": {"content": "unico"}}]}\n'
        )
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)).decode()
    assert '"text": "unico"' in body
    assert "event: error" not in body
    assert events_of(body)[-1] == "message_stop"


@respx.mock
async def test_the_flushed_event_is_judged_like_any_other():
    """O evento recuperado pelo `flush()` passa pelo mesmo criterio: um delta
    vazio no fim do stream continua nao sendo um primeiro evento valido."""
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, headers=SSE_HEADERS, text='data: {"choices": [{"delta": {}}]}\n'),
            httpx.Response(
                200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
            ),
        ]
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert '"text": "ok"' in body
    assert events_of(body).count("message_start") == 1


@respx.mock
async def test_nothing_is_flushed_once_the_loop_has_already_decided():
    """Um erro dentro do stream encerra a leitura; o que ficou no buffer do
    decoder nao pode ressuscitar depois da decisao."""
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                headers=SSE_HEADERS,
                text='data: {"error": {"message": "sem credito"}}\n\n'
                'data: {"choices": [{"delta": {"content": "depois"}}]}\n',
            ),
            httpx.Response(
                200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
            ),
        ]
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert "depois" not in body
    assert '"text": "ok"' in body


# --------------------------------------------------------------------------
# O espelho cru: cliente OpenAI com provedor OpenAI
# --------------------------------------------------------------------------


@respx.mock
async def test_an_openai_provider_keeps_its_own_done_and_gets_no_second_one():
    raw = b': keep-alive\n\ndata: {"choices": [{"delta": {"content": "ok"}}]}\n\ndata: [DONE]\n\n'
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, stream=Chunks(raw))
    )
    body = await run(ShuntRequest("openai", OPENAI_BODY, {}), RAW_OPENAI_SETTINGS)
    # Byte a byte, e com UM sentinela: o provedor ja mandou o dele.
    assert body == raw
    assert body.count(b"data: [DONE]") == 1


@respx.mock
async def test_an_openai_client_still_gets_a_sentinel_when_no_candidate_answers():
    """O sentinela so e suprimido quando os bytes crus do provedor ja o
    trouxeram. Um erro nosso continua precisando dele."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "sem saldo"}})
    )
    body = (await run(ShuntRequest("openai", OPENAI_BODY, {}), RAW_OPENAI_SETTINGS)).decode()
    assert [ln for ln in body.splitlines() if ln][-1] == "data: [DONE]"
    assert "sem saldo" in body


# --------------------------------------------------------------------------
# Retry limitado no `send`, so antes de qualquer byte emitido
# --------------------------------------------------------------------------


@respx.mock
async def test_a_connect_error_retries_the_same_candidate_before_falling_back(monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: seen.append(attempt) or 0.0)
    route = respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.ConnectError("recusou"),
            httpx.ConnectError("recusou"),
            httpx.Response(
                200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
            ),
        ]
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)).decode()
    # Tres tentativas no MESMO candidato, como no caminho bufferizado.
    assert route.call_count == 3
    assert '"text": "ok"' in body
    # A espera cresce com a tentativa; nao e sempre a primeira.
    assert seen == [1, 2]


@respx.mock
async def test_the_retry_budget_is_exhausted_before_the_next_candidate(monkeypatch):
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    route = respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.ConnectError("recusou"),
            httpx.ConnectError("recusou"),
            httpx.ConnectError("recusou"),
            httpx.Response(
                200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
            ),
        ]
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert route.call_count == 4
    assert '"text": "ok"' in body


@respx.mock
async def test_the_backoff_is_awaited_between_attempts(monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: seen.append(attempt) or 0.0)
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.ConnectError("recusou"),
            httpx.Response(
                200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
            ),
        ]
    )
    await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)
    # Espera entre a 1a e a 2a tentativa, e nenhuma depois da ultima.
    assert seen == [1]


@respx.mock
async def test_a_non_transport_http_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    route = respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.DecodingError("corpo ilegivel"),
            httpx.Response(
                200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
            ),
        ]
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)).decode()
    # `DecodingError` e HTTPError mas nao TransportError: `classify` responde
    # SKIP. Uma tentativa e so -- a resposta boa que viria na segunda nunca e
    # pedida, e o cliente recebe o erro.
    assert route.call_count == 1
    assert "corpo ilegivel" in body
    assert events_of(body) == ["error"]


@respx.mock
async def test_the_attempt_number_is_in_the_trace(monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: seen.append(attempt) or 0.0)
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[httpx.ConnectError("recusou")] * MAX_ATTEMPTS
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SOLO_SETTINGS)).decode()
    assert "free: ConnectError: recusou (attempt 1)" in body
    assert f"free: ConnectError: recusou (attempt {MAX_ATTEMPTS})" in body
    # Nenhuma espera depois da ultima tentativa: nada mais vem depois dela.
    assert seen == list(range(1, MAX_ATTEMPTS))


# --------------------------------------------------------------------------
# Raciocinio tambem compromete o stream
# --------------------------------------------------------------------------


def test_a_reasoning_only_delta_commits_the_stream():
    """Medido contra o llama.cpp: um modelo de raciocinio manda dezenas de
    chunks de `reasoning_content` ANTES do primeiro caractere de texto. Com o
    criterio antigo -- so `content` ou `tool_calls` -- todos eram descartados
    aqui, antes de chegarem ao tradutor, e o cliente nunca via o pensamento.
    Raciocinio e saida do modelo: quem o emite ja comecou a responder."""
    assert is_first_valid_event({"choices": [{"delta": {"reasoning_content": "pen"}}]}) is True
    assert is_first_valid_event({"choices": [{"delta": {"reasoning": "pen"}}]}) is True


def test_an_empty_reasoning_field_does_not_commit_the_stream():
    """`reasoning_content: ""` e `reasoning: null` sao o que um provedor manda
    enquanto ainda nao produziu nada. Trata-los como inicio entregaria o
    fallback de graca."""
    assert is_first_valid_event({"choices": [{"delta": {"reasoning_content": ""}}]}) is False
    assert is_first_valid_event({"choices": [{"delta": {"reasoning": None}}]}) is False


@respx.mock
async def test_a_stream_that_reasons_before_answering_delivers_the_thinking():
    """O caminho inteiro, do chunk do provedor ao evento no cliente. E assim
    que o llama.cpp responde: dezenas de chunks de raciocinio e so depois o
    primeiro caractere de texto."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            text=(
                'data: {"choices": [{"delta": {"reasoning_content": "pen"}}]}\n\n'
                'data: {"choices": [{"delta": {"reasoning_content": "sando"}}]}\n\n'
                'data: {"choices": [{"delta": {"content": "Paris"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    assert '"type": "thinking"' in body
    assert '"thinking": "pen"' in body
    assert '"thinking": "sando"' in body
    assert '"text": "Paris"' in body
    # O bloco de pensamento fecha antes de o de texto abrir.
    assert body.index("content_block_stop") < body.index('"type": "text"')


@respx.mock
async def test_a_client_that_walks_away_is_logged_as_such(caplog):
    """Medido no proxy real: stream abandonado no meio era gravado como 200 sem
    candidato, e o painel o exibia como se tivesse sido respondido."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices": [{"delta": {"content": "um"}}]}\n\n',
        )
    )
    pool = UpstreamPool(SETTINGS)
    stream = dispatch_stream(ShuntRequest("anthropic", BODY, {}, endpoint="messages"), SETTINGS, pool)
    with caplog.at_level("INFO", logger="shunt"):
        await anext(stream)  # le um pedaco
        await stream.aclose()  # e desiste
    linhas = [json.loads(r.message) for r in caplog.records if r.name == "shunt"]
    assert linhas, "a requisicao abandonada nao foi registrada"
    linha = linhas[-1]
    assert linha["status"] == 499
    assert linha["error_type"] == "client_disconnected"
    await pool.aclose()


class Holding(httpx.AsyncByteStream):
    """Entrega o primeiro chunk e so depois disso espera num evento, segurando
    o consumidor no meio do stream -- a situacao real do e2e SSE-010, em que o
    cliente corta a conexao enquanto o proxy ainda esta lendo do provedor."""

    def __init__(self, chunk: bytes, release: asyncio.Event) -> None:
        self._chunk = chunk
        self._release = release
        self.closed = False

    async def __aiter__(self):
        yield self._chunk
        await self._release.wait()
        yield self._chunk

    async def aclose(self) -> None:
        self.closed = True


@respx.mock
async def test_a_cancelled_consumption_task_is_logged_as_client_disconnected(caplog):
    """O cancelamento real do framework chega como `CancelledError` no `yield`,
    nao como `GeneratorExit` (esse so e lançado por `aclose()` -- coberto pelo
    teste acima). Com Starlette 1.6 um cliente que corta a conexao cancela a
    task que consome o stream; registrar aquele pedido como 200 esconde
    exatamente as requisicoes que o harness abortou."""
    release = asyncio.Event()
    stream = Holding(b'data: {"choices": [{"delta": {"content": "um"}}]}\n\n', release)
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, stream=stream)
    )
    pool = UpstreamPool(SETTINGS)
    recebido: list[bytes] = []

    async def consume():
        gen = dispatch_stream(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
        chunks: list[bytes] = []
        async for chunk in gen:
            chunks.append(chunk)
            if len(chunks) == 1:
                recebido.append(chunk)
        return b"".join(chunks)

    with caplog.at_level("INFO", logger="shunt"):
        task = asyncio.create_task(consume())
        # Aguarda o primeiro byte sair: sem ele o cancelamento cairia antes de
        # qualquer `yield` e o teste passaria sem medir nada.
        while not recebido:
            assert not task.done(), "o stream terminou antes do cancelamento"
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await pool.aclose()
    linhas = [json.loads(r.message) for r in caplog.records if r.name == "shunt"]
    assert linhas, "o stream cancelado nao foi registrado"
    linha = linhas[-1]
    assert linha["status"] == 499
    assert linha["error_type"] == "client_disconnected"


@respx.mock
async def test_a_fatal_stream_error_is_logged_with_the_failure_status(caplog):
    """O cliente ja viu o `event: error` no fio: registrar a linha do banco
    como 200 sem error_type e declarar sucesso sobre um stream que morreu."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            stream=Failing(
                b'data: {"choices": [{"delta": {"content": "comecou"}}]}\n\n',
                error=httpx.ReadTimeout("tempo esgotado"),
            ),
        )
    )
    pool = UpstreamPool(SETTINGS)
    stream = dispatch_stream(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    with caplog.at_level("INFO", logger="shunt"):
        await collect_into(stream)
    await pool.aclose()
    linhas = [json.loads(r.message) for r in caplog.records if r.name == "shunt"]
    assert linhas, "o stream que falhou no fio nao foi registrado"
    linha = linhas[-1]
    assert linha["status"] == 502
    # O mesmo tipo que `_stream_error` colocou dentro do `event: error` que o
    # cliente recebeu: 502 no dialeto Anthropic e `api_error`.
    assert linha["error_type"] == "api_error"


@respx.mock
async def test_a_size_dropped_candidate_is_named_in_the_trace(caplog):
    """Um candidato pulado por context window tem que aparecer no `attempts` da
    linha do banco: o rastro sem ele diz que o pedido foi atendido pela
    segunda opcao e esconde que a primeira nem foi tentada."""
    settings = Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
            )
        },
        models={
            "tiny": ModelConfig(
                provider="openrouter",
                model="vendor/tiny",
                context_window=50,
                max_output_tokens=16,
            ),
            "free": ModelConfig(
                provider="openrouter",
                model="vendor/free",
                context_window=64000,
                max_output_tokens=8192,
            ),
        },
        routes=[("opus", ["tiny", "free"])],
        default_model=None,
    )
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            text=sse('{"choices": [{"delta": {"content": "ok"}}]}'),
        )
    )
    body = {
        "model": "claude-opus-4-5",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "x" * 8000}],
    }
    pool = UpstreamPool(settings)
    stream = dispatch_stream(ShuntRequest("anthropic", body, {}), settings, pool)
    with caplog.at_level("INFO", logger="shunt"):
        await collect_into(stream)
    await pool.aclose()
    linhas = [json.loads(r.message) for r in caplog.records if r.name == "shunt"]
    assert linhas, "a requisicao nao foi registrada"
    linha = linhas[-1]
    assert any("context window too small" in line for line in linha["attempts"])
    assert any("vendor/tiny" in line for line in linha["attempts"])
    assert linha["candidate"] == "vendor/free"


@respx.mock
async def test_a_mixed_size_and_capability_drop_does_not_take_the_ladder(caplog):
    """Cadeia: primeiro cai por tamanho, segundo cai por capacidade, terceiro
    caberia. A condicao da escada e `all(r == SIZE_DROP)` -- so fecha se TODOS
    os descartes forem de tamanho. Com um de capacidade, a escada NAO fecha e o
    terceiro candidato e chamado. Antes o `if not chain or not all(...)` (bug
    de negacao) ou `any` faria a escada fechar errado e o terceiro seria
    pulado."""
    settings = Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
            )
        },
        models={
            "tiny": ModelConfig(
                provider="openrouter",
                model="vendor/tiny",
                context_window=50,
                max_output_tokens=16,
            ),
            "sem_tools": ModelConfig(
                provider="openrouter",
                model="vendor/sem_tools",
                context_window=64000,
                max_output_tokens=8192,
                supports=ModelCaps(tools=False),
            ),
            "completo": ModelConfig(
                provider="openrouter",
                model="vendor/completo",
                context_window=64000,
                max_output_tokens=8192,
            ),
        },
        routes=[("opus", ["tiny", "sem_tools", "completo"])],
        default_model=None,
    )
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            text=sse('{"choices": [{"delta": {"content": "ok"}}]}'),
        )
    )
    body = {
        "model": "claude-opus-4-5",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "x" * 8000}],
        "tools": [{"type": "function", "function": {"name": "grep"}}],
    }
    pool = UpstreamPool(settings)
    stream = dispatch_stream(ShuntRequest("anthropic", body, {}), settings, pool)
    with caplog.at_level("INFO", logger="shunt"):
        await collect_into(stream)
    await pool.aclose()
    linhas = [json.loads(r.message) for r in caplog.records if r.name == "shunt"]
    assert linhas, "a requisicao nao foi registrada"
    linha = linhas[-1]
    # O terceiro (completo) respondeu; o segundo foi descartado por "no tool support"
    assert any("context window too small" in line for line in linha["attempts"])
    assert any("no tool support" in line for line in linha["attempts"])
    assert linha["candidate"] == "vendor/completo"


@respx.mock
async def test_an_empty_chain_with_mixed_drops_errors_with_the_trace(caplog):
    """Cadeia inteira descartada com motivo misto (tamanho + capacidade) e
    sem escada resolvel: o erro do fio leva os dois motivos e a linha do
    banco fica sem candidato. Antes do fix de integridade o descarte por
    tamanho nem entrava no rastro, e o erro dizia so `no tool support`."""
    settings = Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
            )
        },
        models={
            "tiny": ModelConfig(
                provider="openrouter",
                model="vendor/tiny",
                context_window=50,
                max_output_tokens=16,
            ),
            "sem_tools": ModelConfig(
                provider="openrouter",
                model="vendor/sem_tools",
                context_window=64000,
                max_output_tokens=8192,
                supports=ModelCaps(tools=False),
            ),
        },
        routes=[("opus", ["tiny", "sem_tools"])],
        default_model=None,
    )
    # "my-opus" casa na rota "opus" sem derivar provedor: com `claude-*` e
    # sem default, o degrau transparente oficial entraria na cadeia e o
    # teste cobriria a queda dele, nao o rastro dos dois descartes.
    body = {
        "model": "my-opus",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "x" * 8000}],
        "tools": [{"type": "function", "function": {"name": "grep"}}],
    }
    pool = UpstreamPool(settings)
    stream = dispatch_stream(ShuntRequest("anthropic", body, {}), settings, pool)
    chunks: list[bytes] = []
    with caplog.at_level("INFO", logger="shunt"):
        async for chunk in stream:
            chunks.append(chunk)
    await pool.aclose()
    fio = b"".join(chunks).decode()
    assert "no candidate can serve this request" in fio
    assert "context window too small" in fio
    assert "no tool support" in fio
    linhas = [json.loads(r.message) for r in caplog.records if r.name == "shunt"]
    assert linhas
    assert linhas[-1]["candidate"] is None


async def collect_into(stream) -> None:
    # Consome ate o fim e descarta os bytes: so a linha de log interessa aqui.
    async for _ in stream:
        pass


# -- Guarda 400 do caminho streaming: cadeia esvaziada por skip T4/T5 ---------
# Espelho do caminho bufferizado (test_transparent_with_only_the_shunt_token_
# is_skipped_and_answers_400 e test_anthropic_caller_gpt5_no_default_returns_400):
# quando TODA a cadeia foi pulada por `_transparent_skip_reason` e nenhum
# upstream foi chamado, o erro no fio e 400 no envelope do chamador, nao o 502
# generico de "nada respondeu" (que seria mentira: nada foi chamado).

@respx.mock
async def test_stream_with_only_the_shunt_token_is_skipped_and_answers_400(caplog):
    """Streaming T4: cadeia so com o transparente e a unica credencial era o
    token do Shunt: nenhum upstream e chamado, 400 no fio no envelope do
    chamador e a linha do banco registra 400, nao 502."""
    route = respx.post("https://api.anthropic.com/v1/messages")
    body = {
        "model": "claude-sonnet-5",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "oi"}],
    }
    req = ShuntRequest(
        "anthropic", body, {"x-api-key": "token-do-shunt"}, endpoint="messages",
        credential_is_token=True,
    )
    pool = UpstreamPool(NO_ANTHROPIC_DECLARED)
    chunks: list[bytes] = []
    with caplog.at_level("INFO", logger="shunt"):
        try:
            stream = dispatch_stream(req, NO_ANTHROPIC_DECLARED, pool)
            async for chunk in stream:
                chunks.append(chunk)
        finally:
            await pool.aclose()
    fio = b"".join(chunks).decode()
    assert route.call_count == 0
    # envelope Anthropic: o erro sai num `event: error` com o motivo do pulo
    assert "event: error" in fio
    assert "shunt token" in fio
    linhas = [json.loads(r.message) for r in caplog.records if r.name == "shunt"]
    assert linhas, "o stream pulado inteiro nao foi registrado"
    assert linhas[-1]["status"] == 400
    assert linhas[-1]["candidate"] is None


@respx.mock
async def test_stream_anthropic_caller_gpt5_no_default_answers_400(caplog):
    """Streaming T5, o exemplo canonico: Claude Code pedindo gpt-5 sem default.
    O transparente openai NAO e elegivel para a credencial anthropic: pulado,
    sem outro candidato, 400 no fio no envelope Anthropic, upstream zero."""
    route = respx.post("https://api.openai.com/v1/chat/completions")
    body = {
        "model": "gpt-5",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "oi"}],
    }
    req = ShuntRequest("anthropic", body, {"x-api-key": "sk-do-cliente"}, endpoint="messages")
    pool = UpstreamPool(NO_ANTHROPIC_DECLARED)
    chunks: list[bytes] = []
    with caplog.at_level("INFO", logger="shunt"):
        try:
            stream = dispatch_stream(req, NO_ANTHROPIC_DECLARED, pool)
            async for chunk in stream:
                chunks.append(chunk)
        finally:
            await pool.aclose()
    fio = b"".join(chunks).decode()
    assert route.call_count == 0
    assert "event: error" in fio
    assert "gpt-5" in fio
    linhas = [json.loads(r.message) for r in caplog.records if r.name == "shunt"]
    assert linhas, "o stream pulado inteiro nao foi registrado"
    assert linhas[-1]["status"] == 400


@respx.mock
async def test_stream_path_mismatch_alone_does_not_trigger_the_400_guard(caplog):
    """O pulo por PATH sem entrada (endpoint sem path) NAO conta como skip de
    elegibilidade: so um pulo desses nao pode disparar o 400 da guarda -- o
    erro continua o 502 generico, como antes do fix. Antes do fix o contador
    misturava os dois tipos de pulo e devolvia 400."""
    route = respx.post("https://api.anthropic.com/v1/messages")
    body = {
        "model": "claude-sonnet-5",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "oi"}],
    }
    # ("anthropic", "embeddings") nao existe em PATHS: o unico candidato da
    # cadeia cai no "endpoint not supported", que nao e skip T4/T5.
    req = ShuntRequest("anthropic", body, {}, endpoint="embeddings")
    pool = UpstreamPool(NO_ANTHROPIC_DECLARED)
    chunks: list[bytes] = []
    with caplog.at_level("INFO", logger="shunt"):
        try:
            stream = dispatch_stream(req, NO_ANTHROPIC_DECLARED, pool)
            async for chunk in stream:
                chunks.append(chunk)
        finally:
            await pool.aclose()
    fio = b"".join(chunks).decode()
    assert route.call_count == 0
    assert "endpoint not supported" in fio
    linhas = [json.loads(r.message) for r in caplog.records if r.name == "shunt"]
    assert linhas, "a requisicao sem path nao foi registrada"
    assert linhas[-1]["status"] == 502


@respx.mock
async def test_a_stream_whose_chain_does_not_fit_takes_the_default_model():
    """O caminho de streaming segue a mesma escada do caminho bufferizado."""
    settings = Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
            )
        },
        models={
            "curto": ModelConfig(
                provider="openrouter",
                model="vendor/curto",
                context_window=50,
                max_output_tokens=16,
            )
        },
        routes=[("opus", ["curto"])],
        default_model="curto",
    )
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"delta":{"content":"oi"}}]}\n\ndata: [DONE]\n\n',
        )
    )
    body = {
        "model": "claude-opus-4-5",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "x" * 8000}],
    }
    pool = UpstreamPool(settings)
    try:
        chunks = [
            chunk
            async for chunk in dispatch_stream(ShuntRequest("anthropic", body, {}), settings, pool)
        ]
    finally:
        await pool.aclose()
    texto = b"".join(chunks).decode()
    assert "content_block_delta" in texto


# --- um candidato, um nome, tambem no streaming ---------------------------

ALIAS_DIFFERS = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
        )
    },
    models={
        "qwen-local": ModelConfig(
            provider="openrouter",
            model="qwen3.8-27b",
            context_window=64000,
            max_output_tokens=8192,
            supports=ModelCaps(streaming=True),
        )
    },
    routes=[("opus", ["qwen-local"])],
    default_model=None,
)


@respx.mock
async def test_the_stream_attempt_label_is_the_model_not_the_alias():
    """O rastro do streaming carrega o mesmo rotulo do caminho bufferizado.
    Pelo alias, o cliente le `qwen-local` no erro e `qwen3.8-27b` no campo de
    resultado."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "nao deu"}})
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), ALIAS_DIFFERS)).decode()
    assert "qwen3.8-27b: 400 nao deu (attempt 1)" in body
    assert "qwen-local:" not in body


@respx.mock
async def test_the_stream_400_trace_line_carries_the_upstream_reason():
    """O mesmo do caminho bufferizado: o motivo que o provedor devolveu entra no
    rastro, nao so o status."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "tool unsupported"}})
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), ALIAS_DIFFERS)).decode()
    assert "tool unsupported" in body


@respx.mock
async def test_the_stream_probe_note_names_the_model_not_the_alias(monkeypatch):
    """A nota do probe vale para os dois caminhos. Aqui o candidato tambem nao
    sabe streaming, entao a cadeia esvazia e a nota e a unica evidencia."""
    settings = Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
            )
        },
        models={
            "qwen-local": ModelConfig(
                provider="openrouter",
                model="qwen3.8-27b",
                context_window=64000,
                max_output_tokens=8192,
                supports=ModelCaps(streaming=False),
            )
        },
        routes=[("opus", ["qwen-local"])],
        default_model=None,
    )
    real = dispatcher._payload

    def explode(req, candidate, settings):
        if candidate.model == "qwen3.8-27b":
            raise ValueError("shape estranho")
        return real(req, candidate, settings)

    monkeypatch.setattr(dispatcher, "_payload", explode)
    body = (await run(ShuntRequest("anthropic", BODY, {}), settings)).decode()
    assert "probe (qwen3.8-27b): request translation failed: shape estranho" in body
    # A linha de descarte do filtro nomeia pelo MODELO, igual a linha de
    # tentativa logo acima: `filter_chain` devolve `(model, motivo)`.
    assert "qwen3.8-27b: no streaming support" in body


@respx.mock
async def test_a_read_timeout_on_send_falls_back_without_retrying_the_same_candidate(
    monkeypatch,
):
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    route = respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.ReadTimeout(""),
            httpx.Response(
                200, headers=SSE_HEADERS, text=sse('{"choices": [{"delta": {"content": "ok"}}]}')
            ),
        ]
    )
    body = (await run(ShuntRequest("anthropic", BODY, {}), SETTINGS)).decode()
    # Uma chamada ao primeiro (sem retry), uma ao segundo.
    assert route.call_count == 2
    sent = [json.loads(call.request.content)["model"] for call in route.calls]
    assert sent == ["vendor/free", "vendor/cheap"]
    assert '"text": "ok"' in body
