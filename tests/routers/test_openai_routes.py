import httpx
import respx
from fastapi.testclient import TestClient

from app.core.upstream import UpstreamPool
from app.main import app
from tests.core.test_dispatcher import SETTINGS


def client():
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    return TestClient(app)


@respx.mock
def test_openai_in_openai_out_passes_through_untranslated():
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "vendor/free",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )
    with client() as c:
        body = c.post(
            "/v1/chat/completions",
            json={"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "oi"}]},
        ).json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "ok"
    assert route.calls[0].request.url.path == "/v1/chat/completions"


@respx.mock
def test_embeddings_reach_the_resolved_provider_on_the_embeddings_path():
    route = respx.post("https://api.test/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "model": "vendor/free",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            },
        )
    )
    with client() as c:
        body = c.post("/v1/embeddings", json={"model": "claude-opus-4-5", "input": "texto"}).json()
    assert body["data"][0]["embedding"] == [0.1, 0.2]
    assert route.call_count == 1


@respx.mock
def test_completions_reach_the_completions_path():
    route = respx.post("https://api.test/v1/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "object": "text_completion",
                "model": "vendor/free",
                "choices": [{"text": " pronto", "index": 0, "finish_reason": "stop"}],
            },
        )
    )
    with client() as c:
        body = c.post("/v1/completions", json={"model": "claude-opus-4-5", "prompt": "oi"}).json()
    assert body["choices"][0]["text"] == " pronto"
    assert route.calls[0].request.url.path == "/v1/completions"


@respx.mock
def test_every_openai_route_names_the_model_that_actually_ran():
    """`x-shunt-model` is not a `/v1/messages` courtesy: an OpenAI-speaking
    harness needs the same answer to know which candidate served it."""
    for path, payload in (
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "oi"}]}),
        ("/v1/completions", {"prompt": "oi"}),
        ("/v1/embeddings", {"input": "oi"}),
    ):
        respx.post(f"https://api.test{path}").mock(
            return_value=httpx.Response(200, json={"object": "x", "model": "vendor/free"})
        )
        with client() as c:
            response = c.post(path, json={"model": "claude-opus-4-5", **payload})
        assert response.headers["x-shunt-model"] == "vendor/free", path


@respx.mock
def test_an_openai_client_gets_an_openai_shaped_error_not_an_anthropic_one():
    """The envelope follows the CALLER's protocol. Until this task every error
    was Anthropic-shaped, because `/v1/messages` was the only route; an OpenAI
    client parsing `{"type": "error", ...}` finds no `error.param` and reads a
    message it cannot classify."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "contexto estourado"}})
    )
    # O `last_resort` de `828591e` tenta o transparente no host oficial quando
    # a rota esgota: sem mock dele a chamada saida para a internet.
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(400, json={"error": {"message": "contexto estourado"}})
    )
    with client() as c:
        response = c.post(
            "/v1/chat/completions",
            json={"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "oi"}]},
        )
    assert response.status_code == 400
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"message", "type", "param", "code"}
    assert body["error"]["type"] == "invalid_request_error"
    assert "contexto estourado" in body["error"]["message"]


def test_an_unknown_model_on_an_openai_route_is_an_openai_shaped_400():
    with client() as c:
        response = c.post(
            "/v1/chat/completions",
            json={"model": "modelo-sem-dono", "messages": [{"role": "user", "content": "oi"}]},
        )
    assert response.status_code == 400
    body = response.json()
    assert "type" not in body
    assert body["error"]["type"] == "invalid_request_error"
    assert "provider" in body["error"]["message"]


@respx.mock
def test_chat_completions_streams_when_the_client_asks_for_it():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text='data: {"choices": [{"delta": {"content": "oi"}}]}\n\ndata: [DONE]\n\n',
        )
    )
    ask = {
        "model": "claude-opus-4-5",
        "stream": True,
        "messages": [{"role": "user", "content": "oi"}],
    }
    with (
        client() as c,
        c.stream("POST", "/v1/chat/completions", json=ask) as response,
    ):
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    assert '"content": "oi"' in body
    assert body.count("data: [DONE]") == 1


@respx.mock
def test_embeddings_never_stream_even_when_the_body_asks():
    """`stream` is a chat notion. `/v1/completions` and `/v1/embeddings` answer
    a single JSON document in this proxy; opening an event stream for them
    would hand the client a body its parser never expects."""
    respx.post("https://api.test/v1/embeddings").mock(
        return_value=httpx.Response(
            200, json={"object": "list", "model": "vendor/free", "data": []}
        )
    )
    with client() as c:
        response = c.post(
            "/v1/embeddings", json={"model": "claude-opus-4-5", "input": "oi", "stream": True}
        )
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["object"] == "list"


@respx.mock
def test_completions_never_stream_even_when_the_body_asks():
    respx.post("https://api.test/v1/completions").mock(
        return_value=httpx.Response(
            200, json={"object": "text_completion", "model": "vendor/free", "choices": []}
        )
    )
    with client() as c:
        response = c.post(
            "/v1/completions", json={"model": "claude-opus-4-5", "prompt": "oi", "stream": True}
        )
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["object"] == "text_completion"
