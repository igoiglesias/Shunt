"""O repasse de rotas desconhecidas, de ponta a ponta, contra o provedor falso.

O que so um E2E mostra: o catch-all (`app/routers/relay.py`) e a ULTIMA rota do
aplicativo, entao um path que nao existe no Shunt atravessa middleware,
roteamento, autenticacao e pool antes de chegar ao host oficial -- e o que
sobe e o path cru do cliente, com a query sem o `token`, no host do protocolo
que os headers anunciam. Nenhum teste de unidade de `relay_target` prova que a
rota montada no `app` entregou esse path; so o app inteiro prova.

O segundo bloco e o contrato do registro: a linha de relay chega a quem
investiga a rede (a auditoria), e NAO chega aos numeros do painel (todos os
agregados filtram por `_model_rows()`).
"""

import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.observability import set_recorder
from app.core.security import issue_jwt
from app.core.upstream import UpstreamPool
from app.main import app as shunt_app
from app.routers import dashboard
from app.stats.models import Base, RequestEvent
from app.stats.recorder import Recorder
from tests.conftest import TEST_SHUNT_TOKEN
from tests.e2e.conftest import E2E_SETTINGS
from tests.e2e.fake_provider import ScriptedTransport

# A auditoria exige sessao admin; o segredo conhecido deixa o teste assinar o
# proprio cookie em vez de passar pela tela de login (mesma forma de
# `tests/browser/test_audit_browser.py` e `tests/e2e/test_analysis_e2e.py`).
ADMIN_SESSION_SECRET = "segredo-fixo-do-teste-da-auditoria"

# Path de 100+ caracteres: `route` e `String(64)`, e SQLite nao trunca (o
# plano assumiu o mesmo para libsql remoto). Se um dia truncar, este teste e o
# que descobre.
LONG_PATH = "/api/v1/organizations/" + "a" * 100 + "/usage"

OAUTH_HEADERS = {
    "anthropic-beta": "oauth-2025-04-20",
    "authorization": "Bearer sk-ant-oat-1",
    "user-agent": "Bun/1.4.3",
}

BUN_HEADERS = {
    "user-agent": "Bun/1.4.3",
    "accept": "*/*",
}


def esperar_o_gravador(engine, filtro, timeout=5.0):
    """O gravador escreve em lote numa thread de fundo, entao a leitura espera.

    Sem isso o teste passaria ou falharia pelo relogio, e nao pelo codigo."""
    limite = time.monotonic() + timeout
    while True:
        with Session(engine) as session:
            linhas = session.query(RequestEvent).filter(filtro).all()
        if linhas or time.monotonic() > limite:
            return linhas
        time.sleep(0.05)


def resposta_do_modelo():
    return {
        "id": "chatcmpl-cadeia",
        "model": "vendor/free",
        "choices": [
            {
                "message": {"role": "assistant", "content": "pela cadeia"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    }


@pytest.fixture
def shunt_with_db(provider, tmp_path):
    """O aplicativo real com banco de arquivo, para o registro do relay
    sobreviver ao lote do gravador e poder ser lido por fora."""
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'relay.db'}")
    Base.metadata.create_all(engine)
    settings = E2E_SETTINGS.model_copy(deep=True)
    shunt_app.state.settings = settings
    shunt_app.state.pool = UpstreamPool(settings, transport=ScriptedTransport(provider))
    shunt_app.state.recorder = Recorder(engine, interval=0.01)
    shunt_app.state.admin_session_secret = ADMIN_SESSION_SECRET
    with TestClient(shunt_app) as client:
        client.cookies.set(
            "shunt_admin",
            issue_jwt(user_id=1, secret=ADMIN_SESSION_SECRET, ttl_seconds=3600),
        )
        yield client, engine, provider
    del shunt_app.state.recorder
    set_recorder(Recorder(None))
    engine.dispose()


def test_get_anthropic_oauth_route_is_relayed_with_clean_query(shunt_with_db):
    """1. Um GET OAuth da Anthropic a uma rota que o Shunt nao implementa vai
    ao host oficial, no MESMO path, com a query do cliente (sem o token) e sem
    o `x-shunt-token`."""
    client, _, provider = shunt_with_db
    provider.queue(json_body={"data": [{"usage": 7}]})

    response = client.get("/api/oauth/usage?beta=true", headers=OAUTH_HEADERS)

    assert response.status_code == 200, response.text
    assert response.json() == {"data": [{"usage": 7}]}

    (call,) = provider.calls
    assert call.url.netloc == b"api.anthropic.com", call.url
    assert call.url.path == "/api/oauth/usage", call.url
    assert call.url.query == b"beta=true", call.url
    assert "x-shunt-token" not in call.headers
    # A credencial e do cliente, nao do catalogo: o relay nao injeta chave.
    assert call.headers["authorization"] == "Bearer sk-ant-oat-1"
    assert provider.raw_bodies == [b""]


def test_post_with_a_raw_body_arrives_byte_identical(shunt_with_db):
    """2. O corpo nao e parseado nem re-serializado: um JSON parcial chega
    identico ao que o cliente enviou."""
    client, _, provider = shunt_with_db
    provider.queue(status=400, json_body={"error": {"message": "bad json"}})

    response = client.post("/v1/messages/batches", content=b'{"a":1,', headers=OAUTH_HEADERS)

    assert response.status_code == 400, response.text
    assert response.json() == {"error": {"message": "bad json"}}
    assert provider.raw_bodies == [b'{"a":1,']


def test_head_without_a_token_is_rejected_without_touching_the_upstream(shunt_with_db):
    """3. O HEAD do Bun sem token: 401 no envelope ANTHROPIC, e nenhuma
    chamada ao upstream -- o `TokenRejected` e levantado antes do pool.

    O `without_shunt_token` (tests/conftest.py) restaura o `TestClient.__init__`,
    mas esta fixture ja criou o cliente com o header injetado -- a ordem da
    resolucao e a da assinatura, e o cliente ja existe. Por isso o header e
    tirado na mao, que e onde ele realmente esta."""
    client, _, provider = shunt_with_db
    del client.headers["x-shunt-token"]

    response = client.head("/api/hello", headers=BUN_HEADERS)

    assert response.status_code == 401, response.text
    assert provider.calls == []
    assert provider.raw_bodies == []
    # O envelope do erro segue o protocolo do chamador (aqui, Anthropic).
    # HEAD nao tem corpo por especificacao; o conteudo do envelope e afirmado
    # na chamada POST abaixo, que tem.
    assert response.headers["content-type"].startswith("application/json"), response.headers

    # A mesma recusa num caminho COM corpo confirma o envelope: e o Anthropic,
    # e nao o OpenAI -- e esse o detalhe que um `x-shunt-token` ausente podia
    # mandar errado se o `relay_target` nao olhasse o UA do Bun. O `TestClient`
    # so injeta o token nos atalhos (`post`, `get`, ...); o verbo cru nao passa
    # por la.
    post = client.request("POST", "/v1/messages/batches", content=b'{"a":1,', headers=BUN_HEADERS)
    assert post.status_code == 401, post.text
    body = post.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "authentication_error"
    assert provider.calls == []
    assert provider.raw_bodies == []


def test_head_with_the_token_prefix_is_relayed_without_the_prefix(shunt_with_db):
    """4. `/t/<token>/api/hello` chega ao host oficial como `/api/hello`:
    a middleware reescreve o path ANTES do roteamento, e a query perde o
    `token`."""
    client, _, provider = shunt_with_db
    provider.queue(status=200, json_body={})

    response = client.head(f"/t/{TEST_SHUNT_TOKEN}/api/hello", headers=BUN_HEADERS)

    assert response.status_code == 200, response.text
    (call,) = provider.calls
    assert call.url.netloc == b"api.anthropic.com", call.url
    assert call.url.path == "/api/hello", call.url
    assert call.url.query == b"", call.url


def test_a_normal_model_request_still_uses_the_chain(shunt_with_db):
    """5. Regressao: a rota real de `/v1/messages` nao caiu no catch-all. O
    modelo `claude-haiku-4-5` casa a rota `haiku` e atravessa o dispatcher."""
    client, _, provider = shunt_with_db
    provider.queue(json_body=resposta_do_modelo())

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-haiku-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "oi"}],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["content"] == [{"type": "text", "text": "pela cadeia"}]
    # O path do provedor, e nao o do cliente: passou por traducao e cadeia. O
    # `base_url` do openrouter ja termina em `/v1`, entao o path final e o
    # completo (o falso registra as duas formas).
    assert provider.calls[-1].url.path in ("/chat/completions", "/v1/chat/completions")
    assert provider.raw_bodies == []


def test_relay_is_listed_but_stays_out_of_the_panel_numbers(shunt_with_db):
    """6. O contrato do registro: a auditoria lista a linha de relay com o
    filtro de tipo, e o painel NAO a conta -- `totals.requests` so conta os
    `POST /v1/messages` feitos, e `recent` so traz rotas de modelo."""
    client, engine, provider = shunt_with_db

    provider.queue(json_body={"ok": 1})
    provider.queue(json_body={"ok": 2})
    assert client.get("/api/oauth/usage", headers=OAUTH_HEADERS).status_code == 200
    assert client.get("/api/oauth/usage", headers=OAUTH_HEADERS).status_code == 200
    for _ in range(2):
        provider.queue(json_body=resposta_do_modelo())
        assert (
            client.post(
                "/v1/messages",
                json={
                    "model": "claude-haiku-4-5",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "oi"}],
                },
            ).status_code
            == 200
        )

    # O gravador escreve em lote numa thread de fundo: os dois `POST` chegam
    # ao banco um instante depois, e so depois o painel pode le-los. A janela
    # padrao do resumo e de horas, entao esperar a fila esvaziar basta.
    esperar_o_gravador(engine, RequestEvent.kind == "model")

    # O cache de `/api/stats` e por janela e pode reter um snapshot de outro
    # teste: a leitura so e estavel depois de limpo.
    dashboard._cache.clear()
    stats = client.get("/api/stats").json()
    assert stats["totals"]["requests"] == 2, stats["totals"]
    for entry in stats["recent"]:
        assert "/api/" not in entry["route"], entry

    relays = client.get("/api/requests?kind=relay").json()
    assert relays["total"] == 2, relays
    for item in relays["events"]:
        assert item["kind"] == "relay"
        assert item["route"] == "/api/oauth/usage"
        # Ausente, nao zero: o repasse nao tem uso de tokens.
        assert item["input_tokens"] is None
    assert len(client.get("/api/requests?kind=model").json()["events"]) == 2

    # As linhas de relay chegaram ao banco de arquivo pelo `store`.
    gravadas = esperar_o_gravador(engine, RequestEvent.kind == "relay")
    assert len(gravadas) == 2
    assert {linha.route for linha in gravadas} == {"/api/oauth/usage"}


def test_a_relay_with_a_long_path_is_stored_whole(shunt_with_db):
    """7. `route` e `String(64)` e o path tem mais de 100 caracteres: o
    SQLite nao trunca, e a linha gravada carrega o path completo. Se um dia
    o banco impor o limite da coluna, este teste e o que descobre."""
    client, engine, provider = shunt_with_db
    provider.queue(json_body={})

    response = client.get(LONG_PATH, headers=OAUTH_HEADERS)

    assert response.status_code == 200, response.text
    (call,) = provider.calls
    assert call.url.path == LONG_PATH, call.url

    (gravada,) = esperar_o_gravador(engine, RequestEvent.route == LONG_PATH)
    assert gravada.kind == "relay"
    assert gravada.route == LONG_PATH
