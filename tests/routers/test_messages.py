import httpx
import respx
from fastapi.testclient import TestClient

from app.config.settings import ModelConfig, ProviderConfig, Settings
from app.core.tokens import estimate_input_tokens
from app.core.upstream import UpstreamPool
from app.main import app
from tests.core.test_dispatcher import SETTINGS

ASK = {
    "model": "claude-opus-4-5",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "oi"}],
}

# Um destino que fala Anthropic de verdade: e o unico caso em que
# `/v1/messages/count_tokens` tem equivalente upstream para encaminhar.
ANTHROPIC_SETTINGS = Settings(
    providers={
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test",
            protocol="anthropic",
            api_key_env="SHUNT_TEST_KEY",
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


def client(settings=SETTINGS):
    app.state.settings = settings
    app.state.pool = UpstreamPool(settings)
    return TestClient(app)


# --------------------------------------------------------------------------
# Non-streaming (herdado da Task 14b)
# --------------------------------------------------------------------------


@respx.mock
def test_messages_route_answers_in_anthropic_shape():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "vendor/free",
                "choices": [{"message": {"content": "pronto"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    )
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
        return_value=httpx.Response(
            200,
            json={
                "id": "c",
                "model": "vendor/free",
                "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            },
        )
    )
    with client() as c:
        c.post("/v1/messages", json=ASK, headers={"x-api-key": "sk-do-cliente"})
    assert "sk-do-cliente" not in str(dict(route.calls[0].request.headers))


def test_unknown_model_with_no_declared_provider_returns_400():
    with client() as c:
        response = c.post("/v1/messages", json={**ASK, "model": "modelo-sem-dono"})
    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "provider" in body["error"]["message"]


@respx.mock
def test_exhausted_chain_omits_the_shunt_model_header():
    """No candidate answered, so `result.real_model` stays `None` -- the route
    must not send `x-shunt-model` at all (not even empty) in that case."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "sem sorte"}})
    )
    with client() as c:
        response = c.post("/v1/messages", json=ASK, headers={"x-api-key": "sk-do-cliente"})
    assert response.status_code == 400
    assert "x-shunt-model" not in response.headers


def test_lifespan_builds_the_pool_when_nothing_is_injected():
    """A guarda `hasattr` do lifespan respeita estado injetado; sem injecao, ele
    monta o pool sozinho. Sem isso o proxy nao sobe em producao."""
    from app.main import app as fresh

    for attr in ("settings", "pool"):
        if hasattr(fresh.state, attr):
            delattr(fresh.state, attr)
    with TestClient(fresh):
        assert fresh.state.settings is not None
        assert fresh.state.pool is not None


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


@respx.mock
def test_streaming_request_returns_an_event_stream():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text='data: {"choices": [{"delta": {"content": "oi"}}]}\n\n'
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
            "data: [DONE]\n\n",
        )
    )
    with (
        client() as c,
        c.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as response,
    ):
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    assert "event: message_start" in body
    assert "event: message_stop" in body


@respx.mock
def test_a_stream_false_body_is_answered_as_a_single_document():
    """`stream` is read for its value, not for its presence: `false` is the
    default every OpenAI client sends explicitly."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "vendor/free",
                "choices": [{"message": {"content": "pronto"}, "finish_reason": "stop"}],
            },
        )
    )
    with client() as c:
        response = c.post("/v1/messages", json={**ASK, "stream": False})
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["type"] == "message"


def test_an_unknown_model_is_refused_before_a_single_stream_byte_goes_out():
    """Resolution happens in the route, ahead of the response type. Left to
    `dispatch_stream`, `UnknownProviderError` would raise on the first
    iteration -- after the 200 and the `text/event-stream` header were already
    on the wire, where no error can be reported as a status any more."""
    with client() as c:
        response = c.post("/v1/messages", json={**ASK, "model": "modelo-sem-dono", "stream": True})
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["type"] == "invalid_request_error"


# --------------------------------------------------------------------------
# count_tokens
# --------------------------------------------------------------------------


def test_count_tokens_answers_locally_when_the_target_is_not_anthropic():
    ask = {
        "model": "claude-opus-4-5",
        "messages": [{"role": "user", "content": "uma frase de teste"}],
    }
    with client() as c:
        body = c.post("/v1/messages/count_tokens", json=ask).json()
    assert body["input_tokens"] > 0
    # O numero e a estimativa DESTE corpo, nao uma constante qualquer: um
    # `input_tokens` fixo tambem seria `> 0` e nao serviria para nada.
    assert body["input_tokens"] == estimate_input_tokens(ask)


@respx.mock
def test_count_tokens_asks_the_candidate_that_would_actually_serve():
    """A pergunta vale para o PRIMEIRO candidato da cadeia -- o que serviria o
    pedido. Um candidato Anthropic mais atras na cadeia nao torna a contagem
    dele a correta, e consultar o ultimo mandaria o corpo para um provedor que
    nem vai responder a mensagem."""
    settings = ANTHROPIC_SETTINGS.model_copy(
        update={
            "providers": {
                **ANTHROPIC_SETTINGS.providers,
                "openrouter": ProviderConfig(
                    base_url="https://api.test/v1", protocol="openai", api_key_env=None
                ),
            },
            "models": {
                "free": ModelConfig(
                    provider="openrouter",
                    model="vendor/free",
                    context_window=64000,
                    max_output_tokens=8192,
                ),
                **ANTHROPIC_SETTINGS.models,
            },
            "routes": [("opus", ["free", "native"])],
        }
    )
    route = respx.post("https://api.anthropic.test/v1/messages/count_tokens")
    ask = {"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "oi"}]}
    with client(settings) as c:
        body = c.post("/v1/messages/count_tokens", json=ask).json()
    assert route.call_count == 0
    assert body == {"input_tokens": estimate_input_tokens(ask)}


@respx.mock
def test_count_tokens_is_forwarded_when_the_target_speaks_anthropic(monkeypatch):
    """The provider's own count is authoritative; our estimate is not. This
    body would estimate far fewer than 4321 tokens, so the number proves the
    answer came from upstream.

    It also proves the request is built like every other outbound one: the
    model is the candidate's, and the credential is the configured one -- the
    client's own key must not reach a third-party provider here either."""
    monkeypatch.setenv("SHUNT_TEST_KEY", "sk-da-config")
    route = respx.post("https://api.anthropic.test/v1/messages/count_tokens").mock(
        return_value=httpx.Response(200, json={"input_tokens": 4321})
    )
    with client(ANTHROPIC_SETTINGS) as c:
        body = c.post(
            "/v1/messages/count_tokens",
            json={"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "oi"}]},
            headers={"x-api-key": "sk-do-cliente"},
        ).json()
    assert body == {"input_tokens": 4321}
    assert route.call_count == 1
    import json as _json

    sent = _json.loads(route.calls[0].request.content)
    assert sent["model"] == "claude-real"
    assert sent["messages"] == [{"role": "user", "content": "oi"}]
    assert route.calls[0].request.headers["x-api-key"] == "sk-da-config"
    assert "sk-do-cliente" not in str(dict(route.calls[0].request.headers))


@respx.mock
def test_count_tokens_falls_back_locally_when_the_anthropic_upstream_refuses():
    """A 404 from a gateway that declares itself Anthropic but has no
    `count_tokens` must not become a 404 for the client: the local estimate is
    still a useful answer."""
    respx.post("https://api.anthropic.test/v1/messages/count_tokens").mock(
        return_value=httpx.Response(404, json={"error": {"message": "sem rota"}})
    )
    with client(ANTHROPIC_SETTINGS) as c:
        response = c.post(
            "/v1/messages/count_tokens",
            json={"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "oi"}]},
        )
    assert response.status_code == 200
    assert 0 < response.json()["input_tokens"] < 4321


def test_count_tokens_on_an_unknown_model_is_an_anthropic_shaped_400():
    with client() as c:
        response = c.post(
            "/v1/messages/count_tokens",
            json={"model": "modelo-sem-dono", "messages": [{"role": "user", "content": "oi"}]},
        )
    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    # O 400 e o da resolucao, e nao o da validacao de corpo: o corpo aqui e
    # valido, o que nao existe e o modelo.
    assert "provider" in body["error"]["message"]
