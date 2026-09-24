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

from app.core.upstream import UpstreamPool
from app.main import app
from app.stats.models import ApiToken, Base
from app.stats.recorder import Recorder
from tests.conftest import TEST_SHUNT_TOKEN
from tests.core.test_dispatcher import SETTINGS

ASK = {"model": "claude-opus-4-5", "max_tokens": 8, "messages": [{"role": "user", "content": "oi"}]}
OK = {"id": "chatcmpl-1", "model": "vendor/free", "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}


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