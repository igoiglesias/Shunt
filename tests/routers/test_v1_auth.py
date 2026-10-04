"""Testes de autenticacao do token Shunt em /v1.

Cobre as tres formas (header, prefixo, query), conflito, ausencia,
rejeicao por banco inexistente, credencial nao-token, e revogacao.
"""
import hashlib

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config.settings import ProviderConfig, Settings
from app.core.upstream import UpstreamPool
from app.main import app
from app.stats.models import ApiToken, Base
from app.stats.recorder import Recorder
from tests.conftest import TEST_SHUNT_TOKEN
from tests.core.test_dispatcher import SETTINGS

ASK = {"model": "claude-opus-4-5", "max_tokens": 8, "messages": [{"role": "user", "content": "oi"}]}
OK = {"id": "chatcmpl-1", "model": "vendor/free", "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
ANTHROPIC_OK = {
    "id": "msg_1",
    "model": "claude-opus-4-5",
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "ok"}],
    "usage": {"input_tokens": 1, "output_tokens": 1},
}

# `anthropic` configurado com chave guardada: modelo `claude-*` sem rota vira
# candidato transparente para este provedor, e a chave que sai e a da config.
TRANSPARENT_SETTINGS = Settings(
    providers={
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test",
            protocol="anthropic",
            api_key="sk-guardada",
        )
    },
    models={},
    routes=[],
    default_model=None,
)


def client():
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    return TestClient(app)


@pytest.mark.parametrize("path,body,top", [
    ("/v1/messages", ASK, "type"),
    ("/v1/chat/completions", {"model": "opus", "messages": []}, "error"),
])
def test_missing_token_is_401_in_the_route_envelope(without_shunt_token, path, body, top):
    with client() as c:
        r = c.post(path, json=body)
    assert r.status_code == 401
    assert top in r.json() and "detail" not in r.json()


def test_conflicting_forms_are_400(without_shunt_token):
    with client() as c:
        r = c.post("/v1/messages?token=outro", json=ASK, headers={"x-shunt-token": TEST_SHUNT_TOKEN})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"


@respx.mock
def test_prefix_and_query_forms_are_accepted(without_shunt_token):
    respx.post("https://api.test/v1/chat/completions").mock(return_value=httpx.Response(200, json=OK))
    with client() as c:
        assert c.post(f"/t/{TEST_SHUNT_TOKEN}/v1/messages", json=ASK).status_code == 200
        assert c.post(f"/v1/messages?token={TEST_SHUNT_TOKEN}", json=ASK).status_code == 200


@respx.mock
def test_token_and_cookie_never_leave_shunt():
    route = respx.post("https://api.test/v1/chat/completions").mock(return_value=httpx.Response(200, json=OK))
    with client() as c:
        c.post("/v1/messages", json=ASK, headers={"cookie": "shunt_admin=x"})
    sent = {k.lower() for k in route.calls[0].request.headers}
    assert "x-shunt-token" not in sent and "cookie" not in sent


@respx.mock
def test_token_in_x_api_key_is_accepted_and_never_forwarded(without_shunt_token):
    route = respx.post("https://api.test/v1/chat/completions").mock(return_value=httpx.Response(200, json=OK))
    with client() as c:
        r = c.post("/v1/messages", json=ASK, headers={"x-api-key": TEST_SHUNT_TOKEN})
    assert r.status_code == 200
    assert TEST_SHUNT_TOKEN not in str(dict(route.calls[0].request.headers))


@pytest.mark.parametrize("path,headers", [
    ("/v1/models", {"x-shunt-token": "desconhecido"}),
    ("/t/desconhecido/v1/models", {}),
    ("/v1/models?token=desconhecido", {}),
])
def test_an_explicit_token_with_no_database_and_no_cache_hit_is_503(without_shunt_token, path, headers):
    """R4: producao sem banco nao tem como conferir um token explicito -> 503.
    A suite roda sem banco (tests/conftest.py esvazia TURSO_DATABASE_URL); o
    `app.state.recorder` e limpo entre testes (conftest) para o lifespan do
    `TestClient` criar o `Recorder(None)` de verdade."""
    with client() as c:
        r = c.get(path, headers={**headers, "authorization": "Bearer x"})
    assert r.status_code == 503 and r.json()["error"]["type"] == "server_error"


@respx.mock
def test_a_harness_key_is_a_credential_even_with_no_database():
    """Valor em `x-api-key`/`Authorization` sem acerto no cache e sem banco nao
    e token: segue como credencial, nunca vira 503 (A1)."""
    respx.post("https://api.test/v1/chat/completions").mock(return_value=httpx.Response(200, json=OK))
    with client() as c:
        r = c.post("/v1/messages", json=ASK, headers={"x-api-key": "sk-ant-do-harness", "authorization": "Bearer sk-openai"})
    assert r.status_code == 200


def test_only_a_harness_key_with_no_database_is_401_not_503(without_shunt_token):
    with client() as c:
        r = c.post("/v1/messages", json=ASK, headers={"x-api-key": "sk-ant-do-harness"})
    assert r.status_code == 401 and r.json()["type"] == "error"


def test_count_tokens_and_models_require_the_token(without_shunt_token):
    with client() as c:
        assert c.post("/v1/messages/count_tokens", json={"model": "x", "messages": []}).status_code == 401
        assert c.get("/v1/models").status_code == 401
        assert c.get("/health").status_code == 200


@pytest.mark.parametrize(
    "path",
    ["/v1/models", "/v1/models/opus"],
)
@pytest.mark.parametrize(
    "headers,dialect",
    [
        ({"anthropic-version": "2023-06-01"}, "anthropic"),
        ({"x-api-key": "sk"}, "anthropic"),
        ({"authorization": "Bearer sk"}, "openai"),
        ({}, "openai"),
    ],
)
def test_a_refused_request_on_the_models_route_uses_that_callers_dialect(
    without_shunt_token, path, headers, dialect
):
    """Ponto (a) no GET: o 401 das rotas de modelo segue o dialeto de quem
    perguntou (as duas rotas de familia sao OpenAI por padrao). Sem isso um
    SDK Anthropic recebendo `{"error": ...}` sem o `type` do topo nao
    classifica a falha -- a mesma exigencia das POST."""
    with client() as c:
        response = c.get(path, headers=headers)
    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert ("type" in body) is (dialect == "anthropic")
    assert body["error"]["type"] == "authentication_error"


@respx.mock
def test_an_explicit_token_with_no_other_credential_injects_the_configured_key():
    """Token explicito e a unica credencial apresentada: a chave que sai ao
    candidato transparente e a configurada, nunca o token."""
    app.state.settings, app.state.pool = TRANSPARENT_SETTINGS, UpstreamPool(TRANSPARENT_SETTINGS)
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_OK)
    )
    with TestClient(app) as c:
        r = c.post(
            "/v1/messages",
            json={"model": "claude-opus-4-5", "max_tokens": 8, "messages": [{"role": "user", "content": "oi"}]},
            headers={"x-shunt-token": TEST_SHUNT_TOKEN},
        )
    assert r.status_code == 200
    sent = route.calls[0].request.headers
    assert sent["x-api-key"] == "sk-guardada"
    assert TEST_SHUNT_TOKEN not in str(dict(sent))


@respx.mock
def test_a_shunt_token_presented_as_credential_never_leaves_the_transparent_route():
    """`x-api-key` que CASOU com um token do Shunt nao e credencial do provedor:
    a chave configurada sai no lugar, e o valor do token fica com o proxy."""
    app.state.settings, app.state.pool = TRANSPARENT_SETTINGS, UpstreamPool(TRANSPARENT_SETTINGS)
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_OK)
    )
    with TestClient(app) as c:
        r = c.post(
            "/v1/messages",
            json={"model": "claude-opus-4-5", "max_tokens": 8, "messages": [{"role": "user", "content": "oi"}]},
            headers={"x-api-key": TEST_SHUNT_TOKEN},
        )
    assert r.status_code == 200
    sent = route.calls[0].request.headers
    assert sent["x-api-key"] == "sk-guardada"
    assert TEST_SHUNT_TOKEN not in str(dict(sent))


@respx.mock
def test_a_client_key_alongside_the_explicit_token_passes_through():
    """Etapa 1.5 (A11 do plano): com a propria chave do cliente presente, o
    transparente a repassa crua; a reescrita completa e da Task 2.4. Este
    teste trava o ponto em que a etapa para, para nao "corrigir" demais aqui."""
    app.state.settings, app.state.pool = TRANSPARENT_SETTINGS, UpstreamPool(TRANSPARENT_SETTINGS)
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_OK)
    )
    with TestClient(app) as c:
        r = c.post(
            "/v1/messages",
            json={"model": "claude-opus-4-5", "max_tokens": 8, "messages": [{"role": "user", "content": "oi"}]},
            headers={"x-shunt-token": TEST_SHUNT_TOKEN, "x-api-key": "sk-do-cliente"},
        )
    assert r.status_code == 200
    assert route.calls[0].request.headers["x-api-key"] == "sk-do-cliente"


@respx.mock
def test_count_tokens_never_sends_the_shunt_token_to_the_provider():
    """O `count_tokens` tambem construi um `ShuntRequest`: o valor do token tem
    a mesma guarda que `_serve` e a chave que sai e a configurada."""
    app.state.settings, app.state.pool = TRANSPARENT_SETTINGS, UpstreamPool(TRANSPARENT_SETTINGS)
    route = respx.post("https://api.anthropic.test/v1/messages/count_tokens").mock(
        return_value=httpx.Response(200, json={"input_tokens": 7})
    )
    with TestClient(app) as c:
        r = c.post(
            "/v1/messages/count_tokens",
            json={"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "oi"}]},
            headers={"x-shunt-token": TEST_SHUNT_TOKEN},
        )
    assert r.status_code == 200 and r.json()["input_tokens"] == 7
    sent = route.calls[0].request.headers
    assert sent["x-api-key"] == "sk-guardada"
    assert TEST_SHUNT_TOKEN not in str(dict(sent))


@respx.mock
def test_count_tokens_treats_a_shunt_token_credential_as_not_a_credential():
    """Token em `x-api-key` no `count_tokens`: mesmo buraco do `_serve`. Sem a
    flag, o transparente repassaria o token crua ao provedor."""
    app.state.settings, app.state.pool = TRANSPARENT_SETTINGS, UpstreamPool(TRANSPARENT_SETTINGS)
    route = respx.post("https://api.anthropic.test/v1/messages/count_tokens").mock(
        return_value=httpx.Response(200, json={"input_tokens": 7})
    )
    with TestClient(app) as c:
        r = c.post(
            "/v1/messages/count_tokens",
            json={"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "oi"}]},
            headers={"x-api-key": TEST_SHUNT_TOKEN},
        )
    assert r.status_code == 200
    sent = route.calls[0].request.headers
    assert sent["x-api-key"] == "sk-guardada"
    assert TEST_SHUNT_TOKEN not in str(dict(sent))


def test_a_client_with_only_x_api_key_and_dead_db_gets_401(tmp_path, without_shunt_token):
    """Banco fora do ar, cliente manda APENAS x-api-key (sem x-shunt-token):
    o fluxo inteiro termina em 401, nunca em repasse cru ao provedor.
    Trava o ponto em que a etapa para: se alguém tirar o 401 final de
    require_shunt_token, este teste pega."""
    app.state.settings, app.state.pool = SETTINGS, UpstreamPool(SETTINGS)
    app.state.recorder = Recorder(create_engine(f"sqlite:///{tmp_path}"))
    with TestClient(app) as c:
        r = c.post("/v1/messages", json=ASK, headers={"x-api-key": "sk-do-harness"})
    assert r.status_code == 401
    assert r.json()["type"] == "error"


def test_an_unknown_explicit_token_with_a_dead_database_is_503_not_500(tmp_path, without_shunt_token):
    """Banco fora do ar com token explicito sem cache: nao ha como conferir, e
    R4 vale para a queda do banco como para a ausencia dele -- 503 no envelope,
    nunca `OperationalError` num 500."""
    app.state.settings, app.state.pool = SETTINGS, UpstreamPool(SETTINGS)
    app.state.recorder = Recorder(create_engine(f"sqlite:///{tmp_path}"))
    with TestClient(app) as c:
        r = c.get("/v1/models", headers={"x-shunt-token": "desconhecido", "authorization": "Bearer x"})
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "server_error"


def test_revoking_a_token_invalidates_the_validation_cache_in_process(tmp_path, without_shunt_token):
    engine = create_engine(f"sqlite:///{tmp_path}/t.db")
    Base.metadata.create_all(engine)
    plain = "revogavel"
    with Session(engine) as s:
        s.add(ApiToken(name="r", token_hash=hashlib.sha256(plain.encode()).hexdigest()))
        s.commit()
        token_id = s.execute(__import__("sqlalchemy").select(ApiToken.id)).scalar_one()
    app.state.recorder = Recorder(engine)
    app.state.admin_session_secret = "s"
    from app.core.security import issue_jwt
    try:
        with client() as c:
            assert c.get("/v1/models", headers={"x-shunt-token": plain}).status_code == 200
            c.delete(f"/admin/tokens/{token_id}", cookies={"shunt_admin": issue_jwt(1, "s", 60)})
            assert c.get("/v1/models", headers={"x-shunt-token": plain}).status_code == 401
    finally:
        del app.state.recorder