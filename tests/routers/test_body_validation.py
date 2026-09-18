"""Corpo de requisicao que nao e um objeto JSON.

Enquanto `await request.json()` ficava sem guarda, os dois casos abaixo
subiam como excecao nao tratada e o cliente via 500 -- medido em 25 de 25
combinacoes (cinco rotas POST x cinco corpos). Nenhum dos dois e culpa do
servidor: um corpo que nao parseia, ou que parseia mas nao e um objeto, e
requisicao invalida, e a resposta certa e 400 no dialeto de quem perguntou.
"""

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.core.upstream import UpstreamPool
from app.main import app
from tests.core.test_dispatcher import SETTINGS
from tests.routers.test_messages import ANTHROPIC_SETTINGS

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


# --------------------------------------------------------------------------
# A validacao de entrada nao pode custar fidelidade de repasse
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "route,payload",
    [
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "oi"}]}),
        ("/v1/completions", {"prompt": "oi"}),
        ("/v1/embeddings", {"input": "oi"}),
    ],
)
@respx.mock
def test_a_field_the_proxy_does_not_know_still_reaches_the_provider(route, payload):
    """O provedor ganha campo novo antes de este proxy saber dele. Um schema
    que derruba o desconhecido faria o campo sumir em silencio no caminho, e o
    cliente veria o parametro ser ignorado sem nenhum erro. Vale nas tres
    rotas OpenAI: cada uma tem o seu schema, e cada schema tem a sua guarda."""
    mocked = respx.post(f"https://api.test{route}").mock(
        return_value=httpx.Response(200, json={"object": "x", "model": "vendor/free"})
    )
    with client() as c:
        c.post(
            route,
            json={
                "model": "claude-opus-4-5",
                **payload,
                "campo_que_o_proxy_nao_conhece": {"a": 1},
            },
        )
    sent = json.loads(mocked.calls[0].request.content)
    assert sent["campo_que_o_proxy_nao_conhece"] == {"a": 1}


@respx.mock
def test_no_default_the_caller_never_sent_is_added_to_the_forwarded_body():
    """O outro lado da mesma moeda: validar sem `exclude_unset` empurraria
    todo default do schema para cima, e o provedor veria `max_tokens` e
    `temperature` que o cliente nunca pediu."""
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"object": "chat.completion", "model": "vendor/free"})
    )
    with client() as c:
        c.post(
            "/v1/chat/completions",
            json={"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "oi"}]},
        )
    sent = json.loads(route.calls[0].request.content)
    assert set(sent) == {"model", "messages"}


@pytest.mark.parametrize(
    "route,bad",
    [
        ("/v1/messages", {"model": "claude-opus-4-5"}),
        ("/v1/chat/completions", {"model": "claude-opus-4-5"}),
        ("/v1/completions", {"model": "claude-opus-4-5"}),
        ("/v1/embeddings", {"model": "claude-opus-4-5"}),
        ("/v1/messages/count_tokens", {"model": "claude-opus-4-5"}),
        ("/v1/messages", {"model": 7, "messages": []}),
        ("/v1/chat/completions", {"model": 7, "messages": []}),
    ],
)
def test_a_body_the_schema_rejects_is_a_400_that_names_the_field(route, bad):
    """Antes da validacao, corpo sem o campo obrigatorio subia para o provedor
    e voltava com o erro DELE -- status e texto de outro servico, sobre uma
    requisicao que nunca deveria ter saido daqui."""
    response = client().post(route, json=bad)
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert ":" in message


def test_the_rejection_names_the_dialect_of_whoever_asked():
    anthropic = client().post("/v1/messages", json={"model": "claude-opus-4-5"}).json()
    openai = client().post("/v1/chat/completions", json={"model": "claude-opus-4-5"}).json()
    assert anthropic["type"] == "error"
    assert "type" not in openai
    assert openai["error"]["param"] is None


@respx.mock
def test_the_anthropic_route_also_carries_the_unknown_field_through(monkeypatch):
    """A mesma invariante da rota OpenAI, do lado Anthropic e com um provedor
    que fala Anthropic: sem tradutor no meio, o corpo que sobe e o corpo que
    desceu, campo desconhecido incluido."""
    monkeypatch.setenv("SHUNT_TEST_KEY", "sk-da-config")
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, json={"id": "msg_1", "model": "claude-real"})
    )
    app.state.settings = ANTHROPIC_SETTINGS
    app.state.pool = UpstreamPool(ANTHROPIC_SETTINGS)
    with TestClient(app) as c:
        c.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-5",
                "max_tokens": 16,
                "messages": [
                    {"role": "user", "content": "oi", "chave_dentro_da_mensagem": 1}
                ],
                "campo_que_o_proxy_nao_conhece": {"a": 1},
            },
        )
    sent = json.loads(route.calls[0].request.content)
    assert sent["campo_que_o_proxy_nao_conhece"] == {"a": 1}
    # A guarda vale tambem DENTRO da mensagem: o dialeto Anthropic carrega
    # coisas como `cache_control` no proprio item, e um schema de mensagem
    # estrito as derrubaria sem nenhum erro visivel.
    assert sent["messages"][0]["chave_dentro_da_mensagem"] == 1


@respx.mock
def test_count_tokens_also_carries_the_unknown_field_to_the_provider(monkeypatch):
    """`count_tokens` encaminha o corpo quando o candidato fala Anthropic, e
    a contagem do provedor depende do corpo INTEIRO -- um campo derrubado
    aqui devolve uma contagem que nao corresponde ao que vai ser enviado."""
    monkeypatch.setenv("SHUNT_TEST_KEY", "sk-da-config")
    route = respx.post("https://api.anthropic.test/v1/messages/count_tokens").mock(
        return_value=httpx.Response(200, json={"input_tokens": 9})
    )
    app.state.settings = ANTHROPIC_SETTINGS
    app.state.pool = UpstreamPool(ANTHROPIC_SETTINGS)
    with TestClient(app) as c:
        c.post(
            "/v1/messages/count_tokens",
            json={
                "model": "claude-opus-4-5",
                "messages": [{"role": "user", "content": "oi"}],
                "campo_que_o_proxy_nao_conhece": {"a": 1},
            },
        )
    sent = json.loads(route.calls[0].request.content)
    assert sent["campo_que_o_proxy_nao_conhece"] == {"a": 1}
