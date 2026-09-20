"""Medicao: o OpenAI-cliente que pede stream ve o `usage` antes do `[DONE]`?

Duas rotas medem o mesmo contrato em direcoes opostas:

- OpenAI-cliente para provedor OpenAI e passthrough de bytes: o
  `stream_options.include_usage` que o cliente mandou sobe intacto e o `usage`
  do provedor desce intacto. Este teste e o registro dessa mediacao.
- Anthropic-cliente para provedor OpenAI: o proxy injeta
  `stream_options: {include_usage: true}` no corpo que sobe (to_openai.py) e o
  translator absorve o `usage` de qualquer chunk, inclusive o ultimo, com
  `choices` vazio -- o formato que o OpenAI e o llama.cpp usam para entrega-lo.
"""

from tests.e2e.fake_provider import Scripted, sse_chunks

PROVIDER_STREAM = [
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "pr"}}],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "onto"}, "finish_reason": "stop"}],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    },
]


def _events(text: str) -> list[dict]:
    return [line for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


def test_openai_to_openai_stream_carries_usage_before_done(shunt, provider):
    provider.queue(Scripted(sse=sse_chunks(*PROVIDER_STREAM)))
    with shunt.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "vendor/free",
            "messages": [{"role": "user", "content": "oi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        headers={"authorization": "Bearer sk"},
    ) as resp:
        assert resp.status_code == 200
        text = "".join(chunk for chunk in resp.iter_text())
    events = _events(text)
    usages = [e for e in events if "usage" in e]
    assert len(usages) == 1, f"usage esperado uma vez antes do [DONE], veio: {usages}"
    assert "prompt_tokens\": 7" in usages[0]
    assert "completion_tokens\": 3" in usages[0]
    # `events` ja exclui a linha `[DONE]`, entao o uso do `text.index` aqui
    # prova a ordem: o chunk de usage chegou ANTES do sentinel, nao depois.
    assert text.index(usages[0]) < text.index("data: [DONE]")
    assert text.rstrip().endswith("data: [DONE]")


def test_openai_to_openai_stream_without_include_usage_is_a_byte_passthrough(shunt, provider):
    """O que sobe e o que desce e o proprio do provedor: nenhum campo
    adicionado, nenhum removido. Se o cliente nao pediu usage, o proxy nao
    inventa um."""
    provider.queue(Scripted(sse=sse_chunks(*PROVIDER_STREAM)))
    with shunt.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "vendor/free", "messages": [{"role": "user", "content": "oi"}], "stream": True},
        headers={"authorization": "Bearer sk"},
    ) as resp:
        assert resp.status_code == 200
        text = "".join(chunk for chunk in resp.iter_text())
    events = _events(text)
    assert len(events) == len(PROVIDER_STREAM)
    assert '"usage"' in events[-1]


def test_anthropic_to_openai_stream_injects_include_usage_upstream(shunt, provider):
    provider.queue(Scripted(sse=sse_chunks(*PROVIDER_STREAM)))
    with shunt.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "claude-haiku-4-5",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "oi"}],
        },
        headers={"x-api-key": "sk-ant"},
    ) as resp:
        assert resp.status_code == 200
        text = "".join(chunk for chunk in resp.iter_text())
    body = provider.bodies[0]
    assert body["stream_options"] == {"include_usage": True}
    assert "message_delta" in text


def test_openai_stream_from_an_anthropic_provider_reports_output_tokens(shunt, provider):
    """GAP CONHECIDO: o ultimo chunk emitido pelo `AnthropicStreamToOpenAI`
    carrega so `completion_tokens`, sem o equivalente de `prompt_tokens`.
    Este teste registra a forma atual -- nao e o contrato desejado."""
    provider.queue(
        Scripted(sse=[
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","role":"assistant","usage":{"input_tokens":5}}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"pronto"}}\n\n',
            b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":9},"stop_reason":"end_turn"}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]),
    )
    with shunt.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "oi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        headers={"authorization": "Bearer sk"},
    ) as resp:
        assert resp.status_code == 200
        text = "".join(chunk for chunk in resp.iter_text())
    events = _events(text)
    last = [e for e in events if "usage" in e]
    assert last, f"nenhum chunk com usage na saida: {events}"
    assert '"completion_tokens": 9' in last[-1]
