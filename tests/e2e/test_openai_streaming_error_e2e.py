"""Streaming OpenAI chat/completions e error surface."""

import json

from tests.e2e.fake_provider import Scripted


def _payloads(text: str) -> list[str]:
    """Os conteudos dos eventos `data:` do stream, em ordem.

    O `[DONE]` e um valor de `data:` como qualquer outro, entao ele aparece
    aqui intacto e o teste decide o que ele significa.
    """
    return [line[len("data: "):] for line in text.splitlines() if line.startswith("data: ")]


def test_chat_completions_stream_openai_to_openai(shunt, provider, settings):
    """POST /v1/chat/completions com stream:true retorna stream OpenAI."""
    settings.routes = [("gpt-4o", ["free"])]
    settings.models["free"] = settings.models["free"].model_copy(
        update={"provider": "local", "model": "qwen3-8b"}
    )
    # Provider devolve stream OpenAI (ja no formato certo)
    provider.queue(Scripted(
        sse=[
            b'data: {"id":"cmpl_1","object":"chat.completion.chunk","model":"qwen3-8b","choices":[{"delta":{"content":"Hello"},"finish_reason":null,"index":0}]}\n\n',
            b'data: {"id":"cmpl_1","object":"chat.completion.chunk","model":"qwen3-8b","choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n',
            b'data: [DONE]\n\n',
        ]
    ))

    with shunt.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "oi"}], "stream": True},
        headers={"authorization": "Bearer sk"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = "".join(chunk for chunk in resp.iter_text())

    # Passthrough de bytes: os dois chunks do provider chegam inteiros, na
    # ordem, seguidos do sentinel [DONE] (que tambem viaja como `data:`).
    payloads = _payloads(text)
    assert len(payloads) == 3
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    assert [c["object"] for c in chunks] == ["chat.completion.chunk", "chat.completion.chunk"]
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_stream_anthropic_provider_returns_openai_stream(shunt, provider, settings):
    """OpenAI client -> Anthropic provider com stream: Shunt traduz stream Anthropic -> OpenAI."""
    settings.routes = [("gpt-4o", ["fable"])]
    settings.models["fable"] = settings.models["free"].model_copy(
        update={"provider": "anthropic", "model": "claude-fable-5-1"}
    )
    # Provider Anthropic devolve stream message_start + content_block_delta +
    # message_stop. O SSE da Anthropic carrega a linha `event:` -- sem ela o
    # tradutor (sse_to_openai.feed, despacha por NOME de evento) nao sabe o
    # que e cada payload e nao emite chunk nenhum.
    provider.queue(Scripted(
        sse=[
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-fable-5-1","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
            b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null,"usage":{"output_tokens":2}}}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]
    ))

    with shunt.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "oi"}], "stream": True},
        headers={"authorization": "Bearer sk"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = "".join(chunk for chunk in resp.iter_text())

    # O proxy traduziu o stream Anthropic para o dialeto OpenAI: cada evento
    # vira um `data:` com um chat.completion.chunk e o stream termina no [DONE].
    payloads = _payloads(text)
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    # O rotulo do chunk e o modelo que o CLIENTE pediu, nao o do provider.
    assert all(c["model"] == "gpt-4o" for c in chunks)


def test_chat_completions_stream_false_returns_single_document(shunt, provider, settings):
    """stream:false (default) retorna documento unico, nao stream."""
    settings.routes = [("gpt-4o", ["free"])]
    settings.models["free"] = settings.models["free"].model_copy(
        update={"provider": "local", "model": "qwen3-8b"}
    )
    provider.queue(Scripted(json_body={
        "id": "cmpl_1",
        "object": "chat.completion",
        "model": "qwen3-8b",
        "choices": [{"message": {"content": "Hello", "role": "assistant"}, "finish_reason": "stop", "index": 0}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }))

    body = shunt.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "oi"}]},
        headers={"authorization": "Bearer sk"},
    ).json()

    assert body["object"] == "chat.completion"
    assert "choices" in body
    assert body["choices"][0]["message"]["content"] == "Hello"


def test_error_surface_anthropic_404_unknown_model(shunt):
    """Modelo sem rota retorna 400 Anthropic-shaped."""
    resp = shunt.post(
        "/v1/messages",
        json={"model": "inexistente", "max_tokens": 16, "messages": [{"role": "user", "content": "oi"}]},
        headers={"anthropic-version": "2023-06-01", "x-api-key": "sk-ant"},
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert "provider" in err["message"].lower()


def test_error_surface_openai_404_unknown_model(shunt):
    """Modelo sem rota retorna 400 OpenAI-shaped."""
    resp = shunt.post(
        "/v1/chat/completions",
        json={"model": "inexistente", "messages": [{"role": "user", "content": "oi"}]},
        headers={"authorization": "Bearer sk"},
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert "provider" in err["message"].lower()


def test_error_surface_completions_openai_shape(shunt):
    """Erros em /v1/completions seguem OpenAI shape."""
    resp = shunt.post(
        "/v1/completions",
        json={"model": "inexistente", "prompt": "oi", "max_tokens": 16},
        headers={"authorization": "Bearer sk"},
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"


def test_error_surface_embeddings_openai_shape(shunt):
    """Erros em /v1/embeddings seguem OpenAI shape."""
    resp = shunt.post(
        "/v1/embeddings",
        json={"model": "inexistente", "input": "test"},
        headers={"authorization": "Bearer sk"},
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"


def test_error_surface_count_tokens_anthropic_shape(shunt):
    """Erros em /v1/messages/count_tokens seguem Anthropic shape."""
    resp = shunt.post(
        "/v1/messages/count_tokens",
        json={"model": "inexistente", "messages": [{"role": "user", "content": "oi"}]},
        headers={"anthropic-version": "2023-06-01", "x-api-key": "sk-ant"},
    )
    # count_tokens pode responder localmente (200) se nao for Anthropic
    # Se for Anthropic upstream, erro seria Anthropic shape
    # Aqui modelo inexistente -> 400 com provider missing
    assert resp.status_code == 400
    err = resp.json()["error"]
    # Pode ser Anthropic ou OpenAI shape dependendo do cabecalho
    assert "type" in err