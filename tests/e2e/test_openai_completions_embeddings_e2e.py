"""Endpoints OpenAI-only: /v1/completions e /v1/embeddings."""

from tests.e2e.fake_provider import Scripted

OPENAI_COMPLETION = {
    "id": "cmpl_1",
    "object": "text_completion",
    "model": "qwen3-8b",
    "choices": [{"text": " completado", "finish_reason": "stop", "index": 0}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}

OPENAI_EMBEDDING = {
    "object": "list",
    "data": [
        {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}
    ],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 3, "total_tokens": 3},
}


def test_completions_endpoint_translates_to_provider(shunt, provider, settings):
    """POST /v1/completions roteado para provider via rota configurada."""
    settings.routes = [("text-davinci", ["free"])]
    settings.models["free"] = settings.models["free"].model_copy(
        update={"provider": "local", "model": "qwen3-8b"}
    )
    provider.queue(Scripted(json_body=OPENAI_COMPLETION))

    body = shunt.post(
        "/v1/completions",
        json={"model": "text-davinci", "prompt": "Say hello", "max_tokens": 16},
    ).json()

    assert provider.calls[0].url.path == "/v1/completions"
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == " completado"
    # Cliente e provedor falam o mesmo dialeto (openai -> openai): a resposta
    # e repassada intacta (dispatcher._translate_response, mesmo protocolo),
    # entao `model` e o rotulo do provedor, nao o alias do cliente.
    assert body["model"] == "qwen3-8b"
    assert body["usage"]["total_tokens"] == 5


def test_completions_endpoint_400_when_no_route(shunt):
    """Sem rota configurada, retorna 400."""
    response = shunt.post(
        "/v1/completions",
        json={"model": "inexistente", "prompt": "oi", "max_tokens": 16},
    )
    assert response.status_code == 400
    assert "provider" in response.json()["error"]["message"]


def test_completions_stream_unsupported_returns_single_document(shunt, provider, settings):
    """stream:true em /v1/completions ainda responde documento unico, nunca SSE.

    O endpoint nao tem traducao de stream: `stream` e ignorado e a resposta
    chega como um JSON so, como em qualquer outra chamada bufferizada.
    """
    settings.routes = [("text-davinci", ["free"])]
    settings.models["free"] = settings.models["free"].model_copy(
        update={"provider": "local", "model": "qwen3-8b"}
    )
    provider.queue(Scripted(json_body=OPENAI_COMPLETION))

    resp = shunt.post(
        "/v1/completions",
        json={"model": "text-davinci", "prompt": "Say hello", "stream": True},
    )

    assert resp.status_code == 200
    assert not resp.headers["content-type"].startswith("text/event-stream")
    body = resp.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == " completado"


def test_embeddings_endpoint_translates_to_provider(shunt, provider, settings):
    """POST /v1/embeddings roteado para provider via rota configurada."""
    settings.routes = [("text-embedding-3-small", ["free"])]
    settings.models["free"] = settings.models["free"].model_copy(
        update={"provider": "local", "model": "text-embedding-3-small"}
    )
    provider.queue(Scripted(json_body=OPENAI_EMBEDDING))

    body = shunt.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "test text"},
    ).json()

    assert provider.calls[0].url.path == "/v1/embeddings"
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert body["model"] == "text-embedding-3-small"
    assert body["usage"]["total_tokens"] == 3


def test_embeddings_endpoint_400_when_no_route(shunt):
    """Sem rota configurada, retorna 400."""
    response = shunt.post(
        "/v1/embeddings",
        json={"model": "inexistente", "input": "test"},
    )
    assert response.status_code == 400
    assert "provider" in response.json()["error"]["message"]


def test_embeddings_never_streams_returns_single_document(shunt, provider, settings):
    """/v1/embeddings responde documento unico mesmo com stream:true no corpo.

    Embedding e um lote, nao uma geracao token a token: nao ha o que streamar.
    """
    settings.routes = [("text-embedding-3-small", ["free"])]
    settings.models["free"] = settings.models["free"].model_copy(
        update={"provider": "local", "model": "text-embedding-3-small"}
    )
    provider.queue(Scripted(json_body=OPENAI_EMBEDDING))

    resp = shunt.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "test text", "stream": True},
    )

    assert resp.status_code == 200
    assert not resp.headers["content-type"].startswith("text/event-stream")
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]


def test_completions_and_embeddings_error_shape_openai(shunt):
    """Erros nos endpoints OpenAI retornam envelope OpenAI."""
    # completions
    resp = shunt.post(
        "/v1/completions",
        json={"model": "inexistente", "prompt": "oi", "max_tokens": 16},
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"  # OpenAI shape

    # embeddings
    resp = shunt.post(
        "/v1/embeddings",
        json={"model": "inexistente", "input": "test"},
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"  # OpenAI shape