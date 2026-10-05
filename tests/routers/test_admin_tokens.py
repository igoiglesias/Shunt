"""Testes CRUD de tokens de API do painel admin."""

import hashlib
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.core.upstream import UpstreamPool
from app.main import app
from app.stats.models import ApiToken, Base
from app.stats.recorder import Recorder

ADMIN_SESSION_SECRET = "segredo-de-teste-do-admin"


def _make_engine(tmp_path: Path):
    db = tmp_path / "admin_tokens_test.db"
    engine = create_engine(f"sqlite:///{db}")
    Base.metadata.create_all(engine)
    return engine


def _reset_app_state():
    app.state.settings = Settings(
        providers={}, models={}, routes=[], default_model=None
    )
    app.state.pool = UpstreamPool(app.state.settings)
    app.state.admin_session_secret = ADMIN_SESSION_SECRET


def client(tmp_path: Path) -> TestClient:
    """Cria o app com um banco fresco e um `Recorder` por teste.

    Devolve o `TestClient` SEM entrar nele: quem chama usa `with client(...) as c`
    e o `with` e o do `TestClient`, que roda o lifespan (start/aclose do
    `Recorder`, no loop do portal). A limpeza do `app.state.recorder` entre
    testes e do `conftest.py` (autouse), nao daqui.
    """
    engine = _make_engine(tmp_path)
    _reset_app_state()
    app.state.recorder = Recorder(engine)
    return TestClient(app)


def _cookie(user_id: int = 1) -> dict:
    from app.core.security import issue_jwt
    token = issue_jwt(user_id=user_id, secret=ADMIN_SESSION_SECRET, ttl_seconds=3600)
    return {"shunt_admin": token}


# ---- Tests ----


def test_list_tokens_empty(tmp_path):
    """GET /admin/tokens sem tokens -> 200, 'Nenhum token'."""
    with client(tmp_path) as c:
        r = c.get("/admin/tokens", cookies=_cookie())
    assert r.status_code == 200
    assert "Nenhum token" in r.text


def test_create_token_returns_plaintext_once(tmp_path):
    """POST /admin/tokens -> 200, resposta contem <code id='new-token'> com token de 43 chars."""
    with client(tmp_path) as c:
        r = c.post("/admin/tokens", data={"name": "meu-token"}, cookies=_cookie(), follow_redirects=False)
    assert r.status_code == 200
    # Verifica presenca do elemento code id="new-token"
    assert 'id="new-token"' in r.text
    # Extrai o token do HTML (entre as tags <code ...> TOKEN </code>)
    import re
    match = re.search(r'<code id="new-token"[^>]*>([^<]+)</code>', r.text)
    assert match is not None
    token_value = match.group(1).strip()
    # secrets.token_urlsafe(32) gera ~43 caracteres
    assert len(token_value) == 43


def test_token_hash_stored_not_plaintext(tmp_path):
    """Token armazenado como SHA-256 no DB, nao em texto claro."""
    with client(tmp_path) as c:
        engine = app.state.recorder.engine
        r = c.post("/admin/tokens", data={"name": "hash-test"}, cookies=_cookie(), follow_redirects=False)
    assert r.status_code == 200
    import re
    match = re.search(r'<code id="new-token"[^>]*>([^<]+)</code>', r.text)
    token_value = match.group(1).strip()
    with Session(engine) as s:
        token = s.execute(select(ApiToken).where(ApiToken.name == "hash-test")).scalar_one_or_none()
    assert token is not None
    assert token.token_hash != token_value  # hash nao e o plaintext
    # Verifica que e um SHA-256 (64 hex chars)
    assert len(token.token_hash) == 64
    assert all(c in "0123456789abcdef" for c in token.token_hash)


def test_create_token_with_expiry(tmp_path):
    """POST com expires_days -> expires_at setado no DB."""
    with client(tmp_path) as c:
        engine = app.state.recorder.engine
        r = c.post("/admin/tokens", data={"name": "token-expirado", "expires_days": "30"}, cookies=_cookie(), follow_redirects=False)
    assert r.status_code == 200
    with Session(engine) as s:
        token = s.execute(select(ApiToken).where(ApiToken.name == "token-expirado")).scalar_one()
    assert token.expires_at is not None


def test_create_token_without_expiry(tmp_path):
    """POST sem expires_days -> expires_at NULL no DB."""
    with client(tmp_path) as c:
        engine = app.state.recorder.engine
        r = c.post("/admin/tokens", data={"name": "token-perpetuo"}, cookies=_cookie(), follow_redirects=False)
    assert r.status_code == 200
    with Session(engine) as s:
        token = s.execute(select(ApiToken).where(ApiToken.name == "token-perpetuo")).scalar_one()
    assert token.expires_at is None


def test_revoke_token(tmp_path):
    """DELETE /admin/tokens/{id} -> 200, removido do DB."""
    with client(tmp_path) as c:
        engine = app.state.recorder.engine
        # Cria primeiro
        c.post("/admin/tokens", data={"name": "token-a-revogar"}, cookies=_cookie(), follow_redirects=False)
        with Session(engine) as s:
            token = s.execute(select(ApiToken).where(ApiToken.name == "token-a-revogar")).scalar_one()
        # Deleta
        r = c.delete(f"/admin/tokens/{token.id}", cookies=_cookie(), follow_redirects=False)
    assert r.status_code == 200
    with Session(engine) as s:
        assert s.get(ApiToken, token.id) is None


def test_revoke_unknown_404(tmp_path):
    """DELETE token inexistente -> 404."""
    with client(tmp_path) as c:
        r = c.delete("/admin/tokens/9999", cookies=_cookie(), follow_redirects=False)
    assert r.status_code == 404
    assert "Token nao encontrado" in r.text


def test_token_list_fragment_contains_its_own_anchor(tmp_path):
    """GET /admin/tokens/list -> o fragmento ja nasce com
    `<div id="token-list">`: sem isso, o primeiro swap outerHTML do
    botao Revogar consumia o alvo e as seguintes davam htmx:targetError
    ate recarregar a pagina (mesmo bug dos usuarios, mesmo padrao da
    tela de config)."""
    with client(tmp_path) as c:
        c.post("/admin/tokens", data={"name": "t-alvo"}, cookies=_cookie(), follow_redirects=False)
        r = c.get("/admin/tokens/list", cookies=_cookie())
    assert r.status_code == 200
    assert r.text.count('<div id="token-list">') == 1


def test_tokens_page_renders_each_anchor_exactly_once(tmp_path):
    """GET /admin/tokens -> `id="token-list"` e `id="token-form"` uma
    vez cada na pagina inteira (ancora duplicada faz o htmx mirar no
    primeiro elemento)."""
    with client(tmp_path) as c:
        c.post("/admin/tokens", data={"name": "t-pagina"}, cookies=_cookie(), follow_redirects=False)
        r = c.get("/admin/tokens", cookies=_cookie())
    assert r.status_code == 200
    assert r.text.count('id="token-list"') == 1
    assert r.text.count('id="token-form"') == 1


def test_new_token_form_fragment_carry_the_form_anchor(tmp_path):
    """GET /admin/tokens/new -> o fragmento carrega
    `<div id="token-form">`: o botao Novo faz swap outerHTML nela."""
    with client(tmp_path) as c:
        r = c.get("/admin/tokens/new", cookies=_cookie())
    assert r.status_code == 200
    assert r.text.count('<div id="token-form">') == 1


def test_tokens_crud_sequence_without_reload_keeps_anchors_alive(tmp_path):
    """Sequencia real do HTMX sem recarregar: cada resposta ainda contem
    a ancora que o proximo botao mira. POST /admin/tokens devolve a
    pagina inteira (com destaque do token em claro), entao a assercao
    central de regressao e no DELETE: a lista re-renderizada apos a
    revogacao precisa trazer `#token-list` de volta."""
    with client(tmp_path) as c:
        r = c.get("/admin/tokens/new", cookies=_cookie())
        assert r.status_code == 200
        assert '<div id="token-form">' in r.text, "botao Novo nao reencontra #token-form"

        r = c.post(
            "/admin/tokens",
            data={"name": "regresso"},
            cookies=_cookie(),
            follow_redirects=False,
        )
        assert r.status_code == 200
        assert 'id="token-list"' in r.text, "pagina apos criar nao traz #token-list"

        engine = app.state.recorder.engine
        with Session(engine) as s:
            token = s.execute(select(ApiToken).where(ApiToken.name == "regresso")).scalar_one()

        r = c.delete(f"/admin/tokens/{token.id}", cookies=_cookie(), follow_redirects=False)
    assert r.status_code == 200
    assert r.text.count('<div id="token-list">') == 1, (
        "apos revogar, a lista re-renderizada nao reencontra #token-list"
    )
    with Session(engine) as s:
        assert s.get(ApiToken, token.id) is None


def test_require_admin(tmp_path):
    """Sem cookie em rotas de tokens -> 303/401."""
    with client(tmp_path) as c:
        # GET
        r = c.get("/admin/tokens", follow_redirects=False)
        assert r.status_code in (302, 303, 401)
        assert "/admin/login" in r.headers.get("location", "") or r.status_code == 401

        # POST
        r = c.post("/admin/tokens", data={"name": "x"}, follow_redirects=False)
        assert r.status_code in (302, 303, 401)

        # DELETE
        r = c.delete("/admin/tokens/1", follow_redirects=False)
        assert r.status_code in (302, 303, 401)

        # GET /new
        r = c.get("/admin/tokens/new", follow_redirects=False)
        assert r.status_code in (302, 303, 401)


@respx.mock
def test_shunt_token_on_v1_updates_last_used_at_and_does_not_crash(tmp_path):
    """Um token Shunt valido em `/v1/messages` grava `last_used_at` e nao estoura.

    Regressao: a validacao do token rodava com uma `Session` ja fechada -- o
    `commit` do `last_used_at` levanta `InvalidRequestError` e virava 500 antes
    de o dispatcher ver o corpo. O caminho correto grava e segue para a
    resolucao, que aqui e 400 (catalogo vazio), nunca 500.

    O `last_resort` de `828591e` aponta o transparente para o host oficial, e
    sem mock dele a requisicao saida para a internet (medido: um 401 vivo da
    Anthropic voltava no lugar do 400 do catalogo vazio).
    """
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(400, json={"error": {"message": "nao"}})
    )
    plaintext = "shunt-do-regresso"
    token_hash = hashlib.sha256(plaintext.encode()).hexdigest()
    with client(tmp_path) as c:
        engine = app.state.recorder.engine
        # O cache de validacao e novo por teste (fixture do conftest), entao
        # o token abaixo cai no banco da primeira vez.
        with Session(engine) as s:
            s.add(ApiToken(name="regresso", token_hash=token_hash))
            s.commit()
        r = c.post(
            "/v1/messages",
            json={"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "oi"}]},
            headers={"x-shunt-token": plaintext},
        )
    # O toque de `last_used_at` e gravado pelo WORKER do recorder (Task 1.4):
    # conferir dentro do `with` veria a fila ainda cheia. O `aclose` do lifespan
    # (saida do `with`, no loop do portal) drena a fila, entao a leitura e depois.
    assert r.status_code == 400, f"token valido nao pode virar {r.status_code}"
    with Session(engine) as s:
        row = s.execute(select(ApiToken).where(ApiToken.name == "regresso")).scalar_one()
    assert row.last_used_at is not None, "o uso do token nao ficou registrado"


def test_shunt_token_invalid_on_v1_is_401(tmp_path):
    """`x-shunt-token` desconhecido -> 401, sem tocar no banco alem da leitura."""
    with client(tmp_path) as c:
        r = c.post(
            "/v1/messages",
            json={"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "oi"}]},
            headers={"x-shunt-token": "nao-existe"},
        )
    assert r.status_code == 401


def test_create_token_blank_name_is_400(tmp_path):
    """Nome vazio (ou so espacos) -> 400, nenhum token gravado."""
    with client(tmp_path) as c:
        engine = app.state.recorder.engine
        r = c.post("/admin/tokens", data={"name": "   "}, cookies=_cookie(), follow_redirects=False)
    assert r.status_code == 400
    assert "Informe um nome para o token" in r.text
    with Session(engine) as s:
        assert s.execute(select(ApiToken)).scalars().all() == []


@pytest.mark.parametrize("raw", ["abc", "-5", "0"])
def test_create_token_invalid_expires_days_never_expires(tmp_path, raw):
    """expires_days invalido (nao-numerico, negativo ou zero) vira token sem
    expiracao, em vez de 400: o valor so e usado quando e um numero > 0."""
    with client(tmp_path) as c:
        engine = app.state.recorder.engine
        r = c.post(
            "/admin/tokens",
            data={"name": f"exp-{raw}", "expires_days": raw},
            cookies=_cookie(),
            follow_redirects=False,
        )
    assert r.status_code == 200
    with Session(engine) as s:
        token = s.execute(select(ApiToken).where(ApiToken.name == f"exp-{raw}")).scalar_one()
    assert token.expires_at is None
