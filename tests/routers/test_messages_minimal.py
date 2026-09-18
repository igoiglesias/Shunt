import httpx
import respx
from fastapi.testclient import TestClient

from app.core.upstream import UpstreamPool
from app.main import app
from tests.core.test_dispatcher import SETTINGS

ASK = {"model": "claude-opus-4-5", "max_tokens": 32,
       "messages": [{"role": "user", "content": "oi"}]}


def client():
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    return TestClient(app)


@respx.mock
def test_messages_route_answers_in_anthropic_shape():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "chatcmpl-1", "model": "vendor/free",
            "choices": [{"message": {"content": "pronto"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
    with client() as c:
        response = c.post("/v1/messages", json=ASK, headers={"x-api-key": "sk-do-cliente"})
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["content"] == [{"type": "text", "text": "pronto"}]
    assert body["model"] == "claude-opus-4-5"
    assert response.headers["x-shunt-model"] == "vendor/free"


@respx.mock
def test_routed_call_does_not_forward_the_client_credential():
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "c", "model": "vendor/free",
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}))
    with client() as c:
        c.post("/v1/messages", json=ASK, headers={"x-api-key": "sk-do-cliente"})
    assert "sk-do-cliente" not in str(dict(route.calls[0].request.headers))


def test_unknown_model_with_no_declared_provider_returns_400():
    with client() as c:
        response = c.post("/v1/messages", json={**ASK, "model": "modelo-sem-dono"})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "provider" in response.json()["error"]["message"]


@respx.mock
def test_exhausted_chain_omits_the_shunt_model_header():
    """No candidate answered, so `result.real_model` stays `None` -- the route
    must not send `x-shunt-model` at all (not even empty) in that case."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "sem sorte"}}))
    with client() as c:
        response = c.post("/v1/messages", json=ASK, headers={"x-api-key": "sk-do-cliente"})
    assert response.status_code == 400
    assert "x-shunt-model" not in response.headers


def test_lifespan_builds_the_pool_when_nothing_is_injected():
    """A guarda `hasattr` do lifespan respeita estado injetado; sem injeção, ele
    monta o pool sozinho. Sem isso o proxy não sobe em produção."""
    from app.main import app as fresh
    for attr in ("settings", "pool"):
        if hasattr(fresh.state, attr):
            delattr(fresh.state, attr)
    with TestClient(fresh):
        assert fresh.state.settings is not None
        assert fresh.state.pool is not None
