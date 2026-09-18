"""Corpo de requisicao que nao e um objeto JSON.

Enquanto `await request.json()` ficava sem guarda, os dois casos abaixo
subiam como excecao nao tratada e o cliente via 500 -- medido em 25 de 25
combinacoes (cinco rotas POST x cinco corpos). Nenhum dos dois e culpa do
servidor: um corpo que nao parseia, ou que parseia mas nao e um objeto, e
requisicao invalida, e a resposta certa e 400 no dialeto de quem perguntou.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.upstream import UpstreamPool
from app.main import app
from tests.core.test_dispatcher import SETTINGS

JSON = {"content-type": "application/json"}

# Um corpo por forma que o JSON aceita e a rota nao: o truncado nem parseia,
# os outros quatro parseiam para algo que nao tem `.get`.
BAD_BODIES = [
    ('{"model": ', "malformado"),
    ("[1, 2, 3]", "lista"),
    ('"oi"', "string"),
    ("42", "numero"),
    ("null", "null"),
]

ANTHROPIC_ROUTES = ["/v1/messages", "/v1/messages/count_tokens"]
OPENAI_ROUTES = ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"]


def client():
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("route", ANTHROPIC_ROUTES + OPENAI_ROUTES)
@pytest.mark.parametrize("body,shape", BAD_BODIES)
def test_a_body_that_is_not_a_json_object_is_a_400(route, body, shape):
    response = client().post(route, content=body, headers=JSON)
    assert response.status_code == 400, f"{route} com corpo {shape}"


@pytest.mark.parametrize("route", ANTHROPIC_ROUTES)
@pytest.mark.parametrize("body,shape", BAD_BODIES)
def test_the_anthropic_routes_answer_in_the_anthropic_envelope(route, body, shape):
    payload = client().post(route, content=body, headers=JSON).json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["message"]


@pytest.mark.parametrize("route", OPENAI_ROUTES)
@pytest.mark.parametrize("body,shape", BAD_BODIES)
def test_the_openai_routes_answer_in_the_openai_envelope(route, body, shape):
    payload = client().post(route, content=body, headers=JSON).json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["param"] is None
    assert payload["error"]["code"] is None
    assert payload["error"]["message"]
    assert "type" not in payload


@pytest.mark.parametrize("route", ANTHROPIC_ROUTES + OPENAI_ROUTES)
def test_a_body_that_does_not_parse_says_so_and_not_something_about_the_model(route):
    """Sem esta assercao, deixar o corpo quebrado seguir como `{}` passa
    despercebido: o `{}` cai na resolucao, que responde 400 tambem -- mesmo
    status, mesmo envelope, mensagem errada, falando de modelo em vez de JSON.
    """
    message = client().post(route, content='{"model": ', headers=JSON).json()["error"]["message"]
    assert "JSON" in message


@pytest.mark.parametrize("route", ANTHROPIC_ROUTES + OPENAI_ROUTES)
def test_a_body_of_the_wrong_shape_names_the_shape_it_got(route):
    """Mesma armadilha do lado do nao-objeto: a mensagem tem de nomear o tipo
    recebido, senao trocar a guarda por um 400 generico continua verde."""
    message = client().post(route, content="[1, 2]", headers=JSON).json()["error"]["message"]
    assert "objeto JSON" in message
    assert "list" in message


def test_the_message_names_which_of_the_two_faults_it_was():
    """Um 400 que nao diz se o corpo nao parseou ou se parseou para a forma
    errada obriga quem integra a adivinhar. As duas mensagens sao distintas."""
    broke = client().post("/v1/messages", content='{"model": ', headers=JSON).json()
    wrong_shape = client().post("/v1/messages", content="[1, 2]", headers=JSON).json()
    assert broke["error"]["message"] != wrong_shape["error"]["message"]


def test_a_well_formed_object_still_goes_through_to_resolution():
    """A guarda pega corpo invalido, nao corpo desconhecido: um objeto bem
    formado com modelo que nao resolve continua caindo no 400 da resolucao,
    com a mensagem da resolucao e nao com a do parse."""
    payload = client().post(
        "/v1/messages", json={"model": "nao-existe", "messages": []}, headers=JSON
    ).json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "JSON" not in payload["error"]["message"]
