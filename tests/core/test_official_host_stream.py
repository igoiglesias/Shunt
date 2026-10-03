"""Streaming 200 no host oficial embutido (sem provider no catalogo).

A lacuna mapeada: nenhum teste fazia um `stream: True` chegar a
`https://api.anthropic.com/v1/messages` ou `https://api.openai.com/v1` com
`NO_ANTHROPIC_DECLARED` (provider ausente, default_model=None). Os testes
existentes com essas settings esperavam 400 -- ou o destino estrangeiro da
credencial (T5/R2), ou o token do Shunt como unica credencial (T4). Esta
suíte cobre o sucesso (200) no destino nativo da credencial, onde o
transparente e elegivel: passthrough cru, ja que o protocolo dos dois lados
e o mesmo.

Os dois sentidos cross-protocol (anthropic pedindo gpt-*, openai pedindo
claude-*) sao 400 por design e ja tem guarda em test_dispatcher.py.
"""

import respx

from app.core.dispatcher import dispatch_stream
from app.core.upstream import UpstreamPool
from app.routers.v1 import ShuntRequest
from tests.core.test_dispatcher import NO_ANTHROPIC_DECLARED


# --- Helpers de stream ------------------------------------------------------


def sse_openai_stream(model: str = "gpt-5"):
    """Stream OpenAI SSE que o provider real envia."""
    return (
        f'data: {{"id":"chatcmpl_1","object":"chat.completion","model":"{model}","choices":[{{"delta":{{"content":"Ola"}}}}]}}\n\n'.encode()
        + b'data: {"choices":[{"delta":{"content":" mundo"}}]}\n\n'
        + b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
        + b'data: [DONE]\n\n'
    )


def sse_anthropic_stream(model: str = "claude-opus-5-5"):
    """Stream Anthropic SSE que o provider real envia."""
    return (
        f'event: message_start\ndata: {{"type":"message_start","message":{{"id":"msg_1","type":"message","role":"assistant","model":"{model}","content":[],"stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":5,"output_tokens":0}}}}}}\n\n'.encode()
        + b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        + b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Ola"}}\n\n'
        + b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" mundo"}}\n\n'
        + b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        + b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}\n\n'
        + b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )


async def _drain(req, settings, pool):
    chunks = []
    try:
        async for chunk in dispatch_stream(req, settings, pool):
            chunks.append(chunk)
    finally:
        await pool.aclose()
    return b"".join(chunks).decode()


# --- Testes -----------------------------------------------------------------


@respx.mock
async def test_openai_caller_to_openai_official_host_stream_200():
    """Caller OpenAI pedindo gpt-*, provider openai ausente, passthrough cru.

    Mesmo protocolo nos dois lados: o SSE do upstream e repassado byte a byte,
    sem traducao e sem um segundo `[DONE]` (o do provider ja viajou).
    """
    route = respx.post("https://api.openai.com/v1/chat/completions").respond(
        200, content=sse_openai_stream(), headers={"content-type": "text/event-stream"}
    )
    pool = UpstreamPool(NO_ANTHROPIC_DECLARED)
    req = ShuntRequest(
        "openai",
        {"model": "gpt-5", "max_tokens": 64, "stream": True, "messages": [{"role": "user", "content": "oi"}]},
        {"authorization": "Bearer sk-do-cliente-openai"},
        endpoint="chat",
    )
    wire = await _drain(req, NO_ANTHROPIC_DECLARED, pool)

    assert route.call_count == 1
    sent = route.calls[0].request
    assert sent.url.host == "api.openai.com", f"host errado: {sent.url}"
    assert sent.url.path == "/v1/chat/completions", f"path errado: {sent.url}"
    assert sent.headers["authorization"] == "Bearer sk-do-cliente-openai"

    # Passthrough cru: OpenAI SSE intocado
    assert "data: " in wire
    assert '"choices"' in wire
    assert "data: [DONE]" in wire
    # O [DONE] do proxy NAO se soma ao do provider no passthrough
    assert wire.count("[DONE]") == 1


@respx.mock
async def test_anthropic_caller_to_anthropic_official_host_stream_200():
    """Caller Anthropic pedindo claude-*, provider anthropic ausente, passthrough cru."""
    route = respx.post("https://api.anthropic.com/v1/messages").respond(
        200, content=sse_anthropic_stream(), headers={"content-type": "text/event-stream"}
    )
    pool = UpstreamPool(NO_ANTHROPIC_DECLARED)
    req = ShuntRequest(
        "anthropic",
        {"model": "claude-opus-5-5", "max_tokens": 64, "stream": True, "messages": [{"role": "user", "content": "oi"}]},
        {"x-api-key": "sk-ant-do-cliente"},
        endpoint="messages",
    )
    wire = await _drain(req, NO_ANTHROPIC_DECLARED, pool)

    assert route.call_count == 1
    sent = route.calls[0].request
    assert sent.url.host == "api.anthropic.com", f"host errado: {sent.url}"
    assert sent.url.path == "/v1/messages", f"path errado: {sent.url}"
    assert sent.headers["x-api-key"] == "sk-ant-do-cliente"

    # Passthrough cru: Anthropic SSE intocado
    assert "event: message_start" in wire
    assert "event: content_block_delta" in wire
    assert "text_delta" in wire
    assert "event: message_delta" in wire
    assert "event: message_stop" in wire
    # Cliente Anthropic nunca ve o sentinel do OpenAI
    assert "[DONE]" not in wire
