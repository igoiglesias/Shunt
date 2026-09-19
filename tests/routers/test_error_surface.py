"""A forma da resposta quando ela e um erro, rota a rota.

A auditoria de cobertura da Task 26 achou estas lacunas: a cobertura de linha
de `app/routers/v1.py` estava em 100%, e mesmo assim nenhum teste dizia que um
envelope de erro sai como JSON, que um 401 do provedor vira
`authentication_error`, ou que `/v1/completions` recusa um modelo sem dono.
Linha coberta nao e comportamento preso.
"""

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.core.upstream import UpstreamPool
from app.main import app
from tests.core.test_dispatcher import SETTINGS

# (rota, corpo minimo valido para aquela rota, dialeto do envelope de erro)
ROUTES = [
    (
        "/v1/messages",
        {"max_tokens": 16, "messages": [{"role": "user", "content": "oi"}]},
        "anthropic",
    ),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "oi"}]}, "openai"),
    ("/v1/completions", {"prompt": "oi"}, "openai"),
    ("/v1/embeddings", {"input": "oi"}, "openai"),
    ("/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "oi"}]}, "anthropic"),
]


def client():
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    return TestClient(app)


def upstream_path(route):
    return "/v1/chat/completions" if route == "/v1/messages" else route


@pytest.mark.parametrize(("route", "payload", "dialect"), ROUTES)
def test_an_unknown_model_is_refused_in_the_dialect_of_the_caller(route, payload, dialect):
    with client() as c:
        response = c.post(route, json={"model": "modelo-sem-dono", **payload})
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "provider" in body["error"]["message"]
    # O envelope Anthropic tem `type` no topo; o da OpenAI nao.
    assert ("type" in body) is (dialect == "anthropic")


@pytest.mark.parametrize(("route", "payload", "dialect"), ROUTES)
def test_a_malformed_body_is_refused_as_json_not_as_text(route, payload, dialect):
    with client() as c:
        response = c.post(
            route, content=b"{nao e json", headers={"content-type": "application/json"}
        )
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["type"] == "invalid_request_error"


@respx.mock
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (500, "api_error"),
        (529, "overloaded_error"),
    ],
)
def test_an_upstream_status_becomes_the_matching_anthropic_error_type(status, expected):
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(status, json={"error": {"message": "nao"}})
    )
    with client() as c:
        response = c.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-5",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "oi"}],
            },
        )
    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["type"] == expected


@respx.mock
def test_a_stream_that_fails_upstream_stays_an_event_stream():
    """Um erro depois do 200 nao pode trocar o content-type no meio do caminho.

    O cabecalho ja foi para a rede quando a falha aparece; o cliente esta com um
    parser de SSE montado, e um corpo JSON ali e lixo para ele.
    """
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "nao deu"}})
    )
    with (
        client() as c,
        c.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "claude-opus-4-5",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "oi"}],
            },
        ) as response,
    ):
        body = "".join(response.iter_text())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: error" in body


@respx.mock
def test_a_streaming_response_does_not_promise_x_shunt_model():
    """Decisao, e nao esquecimento: o cabecalho nao pode ser honesto aqui.

    Quem serviu o stream so se sabe quando o primeiro evento valido chega, e os
    cabecalhos ja sairam antes disso -- a troca de candidato acontece depois. O
    candidato real esta na linha de log da requisicao, que e onde o operador o
    procura. Este teste existe para que ligar o cabecalho sem resolver o
    problema de ordem quebre um teste, em vez de mentir para um cliente.
    """
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices": [{"delta": {"content": "oi"}}]}\n\ndata: [DONE]\n\n',
        )
    )
    with (
        client() as c,
        c.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "claude-opus-4-5",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "oi"}],
            },
        ) as response,
    ):
        "".join(response.iter_text())
    assert "x-shunt-model" not in response.headers


def test_the_model_listing_answers_json_in_both_dialects():
    with client() as c:
        anthropic = c.get("/v1/models", headers={"anthropic-version": "2023-06-01"})
        openai = c.get("/v1/models", headers={"authorization": "Bearer sk"})
    for response in (anthropic, openai):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")


def test_count_tokens_answers_json_when_it_estimates_locally():
    with client() as c:
        response = c.post(
            "/v1/messages/count_tokens",
            json={"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "oi"}]},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["input_tokens"] > 0
