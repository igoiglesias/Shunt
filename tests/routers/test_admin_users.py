"""Testes CRUD de usuarios do painel admin.

Todos os testes passam por autenticacao JWT cookie (COOKIE) e
usam engine SQLite temporaria por teste.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.core.security import hash_password
from app.core.upstream import UpstreamPool
from app.main import app
from app.stats.models import Base, User

ADMIN_SESSION_SECRET = "segredo-de-teste-do-admin"


def _make_engine(tmp_path: Path):
    db = tmp_path / "admin_users_test.db"
    engine = create_engine(f"sqlite:///{db}")
    Base.metadata.create_all(engine)
    return engine


def _reset_app_state():
    app.state.settings = Settings(
        providers={}, models={}, routes=[], default_model=None
    )
    app.state.pool = UpstreamPool(app.state.settings)
    app.state.admin_session_secret = ADMIN_SESSION_SECRET


def client(tmp_path: Path):
    """Return a TestClient with a temporary SQLite DB."""
    engine = _make_engine(tmp_path)
    _reset_app_state()
    from app.stats.recorder import Recorder
    app.state.recorder = Recorder(engine)
    return TestClient(app)


def _cookie(user_id: int = 1) -> dict:
    """Monta o cookie `shunt_admin` com um JWT valido."""
    from app.core.security import issue_jwt
    token = issue_jwt(user_id=user_id, secret=ADMIN_SESSION_SECRET, ttl_seconds=3600)
    return {"shunt_admin": token}


def _create_user(session: Session, username: str = "admin") -> User:
    """Cria um usuario na base de teste."""
    user = User(username=username, password_hash=hash_password("senha123"))
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


# ---- Tests ----


def test_list_users_empty(tmp_path):
    """GET /admin/users com DB vazia -> 200, 'Nenhum usuario'."""
    with client(tmp_path) as c:
        r = c.get("/admin/users", cookies=_cookie())
    assert r.status_code == 200
    assert "Nenhum usuario" in r.text


def test_create_user(tmp_path):
    """POST /admin/users -> 200, usuario persistido na DB e na tabela."""
    with client(tmp_path) as c:
        r = c.post("/admin/users", data={"username": "novo", "password": "senha123"}, cookies=_cookie(), follow_redirects=False)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        user = s.execute(select(User).where(User.username == "novo")).scalar_one_or_none()
    assert user is not None
    assert "novo" in r.text


def test_create_user_duplicate_400(tmp_path):
    """POST /admin/users com username existente -> 400."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "existing")
        r = c.post("/admin/users", data={"username": "existing", "password": "senha123"}, cookies=_cookie())
    assert r.status_code == 400
    assert "Usuario ja existe" in r.text


def test_edit_user_rename(tmp_path):
    """PATCH /admin/users/{old} com new_username -> 200, renomeado no DB."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "antigo")
        r = c.patch(
            "/admin/users/antigo",
            data={"new_username": "novo_nome"},
            cookies=_cookie(),
            follow_redirects=False,
        )
    assert r.status_code == 200
    assert "novo_nome" in r.text
    assert "antigo" not in r.text
    with Session(app.state.recorder.engine) as s:
        user = s.execute(select(User).where(User.username == "novo_nome")).scalar_one_or_none()
        old = s.execute(select(User).where(User.username == "antigo")).scalar_one_or_none()
    assert user is not None
    assert old is None


def test_edit_user_password(tmp_path):
    """PATCH /admin/users/{username} com new_password -> 200, pode logar com nova senha."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "testuser")
        r = c.patch(
            "/admin/users/testuser",
            data={"new_password": "nova_senha_456"},
            cookies=_cookie(),
            follow_redirects=False,
        )
    assert r.status_code == 200
    with client(tmp_path) as login_client:
        login = login_client.post(
            "/admin/login",
            data={"username": "testuser", "password": "nova_senha_456"},
            follow_redirects=False,
        )
    assert login.status_code == 303
    assert login.headers["location"] == "/admin/painel"
    assert login.cookies.get("shunt_admin") is not None


def test_edit_user_rename_conflict_400(tmp_path):
    """PATCH renomear para username existente -> 400."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "user1")
            _create_user(s, "user2")
        r = c.patch(
            "/admin/users/user1",
            data={"new_username": "user2"},
            cookies=_cookie(),
        )
    assert r.status_code == 400
    assert "Nome de usuario ja em uso" in r.text


def test_delete_user(tmp_path):
    """DELETE /admin/users/{username} -> 200, removido do DB e tabela."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "delete_me")
            _create_user(s, "current_admin")
        r = c.delete("/admin/users/delete_me", cookies=_cookie(), follow_redirects=False)
    assert r.status_code == 200
    assert "delete_me" not in r.text
    with Session(app.state.recorder.engine) as s:
        user = s.execute(select(User).where(User.username == "delete_me")).scalar_one_or_none()
    assert user is None


def test_delete_last_admin_blocked(tmp_path):
    """Tentar deletar o unico usuario (o logado) -> 400."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "only_admin")
        r = c.delete("/admin/users/only_admin", cookies=_cookie(), follow_redirects=False)
    assert r.status_code == 400
    assert "Exclusao do unico admin bloqueada" in r.text


# ---- Ancoras HTMX: o fragmento carrega a propria ancora (regressao) ----


def test_user_list_fragment_contains_its_own_anchor(tmp_path):
    """GET /admin/users/list -> o fragmento ja nasce com `<div id="user-list">`.

    Regressao: sem a ancora no fragmento, o primeiro swap outerHTML dos
    botoes Editar/Excluir consumia o alvo e as acoes seguintes davam
    htmx:targetError ate recarregar a pagina."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "alice")
        r = c.get("/admin/users/list", cookies=_cookie())
    assert r.status_code == 200
    assert r.text.count('<div id="user-list">') == 1


def test_users_page_renders_each_anchor_exactly_once(tmp_path):
    """GET /admin/users -> a pagina inteira tem `id="user-list"` e
    `id="user-form"` uma vez cada (nada de ancora duplicada, que faz o
    htmx mirar no primeiro elemento)."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "alice")
        r = c.get("/admin/users", cookies=_cookie())
    assert r.status_code == 200
    assert r.text.count('id="user-list"') == 1
    assert r.text.count('id="user-form"') == 1


def test_user_form_fragments_carry_the_form_anchor(tmp_path):
    """GET /admin/users/new e /{username}/edit -> o fragmento do form
    carrega `<div id="user-form">`: o botao Novo/Editar faz swap
    outerHTML nela e precisa reencontra-la a cada troca."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "alice")
        r_new = c.get("/admin/users/new", cookies=_cookie())
        r_edit = c.get("/admin/users/alice/edit", cookies=_cookie())
    for r in (r_new, r_edit):
        assert r.status_code == 200
        assert r.text.count('<div id="user-form">') == 1


def test_users_crud_sequence_without_reload_keeps_anchors_alive(tmp_path):
    """Sequencia completa sem recarregar a pagina (o fluxo real do HTMX):
    cada resposta de fragmento ainda contem a ancora que o proximo botao
    mira -- e isso que mantinha as acoes seguintes vivas apos o bug do
    swap outerHTML."""
    with client(tmp_path) as c:
        # O `guarda` (id=1, mesmo do cookie) impede o bloqueio de
        # "exclusao do unico admin" quando o usuario da sequencia sair.
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "guarda")

        r = c.get("/admin/users/new", cookies=_cookie())
        assert r.status_code == 200
        assert '<div id="user-form">' in r.text, "botao Novo nao reencontra #user-form"

        r = c.post(
            "/admin/users",
            data={"username": "regresso", "password": "senha123"},
            cookies=_cookie(),
            follow_redirects=False,
        )
        assert r.status_code == 200
        assert "regresso" in r.text
        assert '<div id="user-list">' in r.text, "apos criar, Excluir/Editar nao reencontram #user-list"

        r = c.get("/admin/users/regresso/edit", cookies=_cookie())
        assert r.status_code == 200
        assert '<div id="user-form">' in r.text, "botao Editar nao reencontra #user-form"

        r = c.patch(
            "/admin/users/regresso",
            data={"new_username": "renomeado"},
            cookies=_cookie(),
            follow_redirects=False,
        )
        assert r.status_code == 200
        assert "renomeado" in r.text
        assert '<div id="user-list">' in r.text, "apos editar, Excluir nao reencontra #user-list"

        r = c.delete("/admin/users/renomeado", cookies=_cookie(), follow_redirects=False)
    assert r.status_code == 200
    assert "renomeado" not in r.text
    assert '<div id="user-list">' in r.text, "apos excluir, o proximo botao nao reencontra #user-list"


def test_require_admin_on_all_routes(tmp_path):
    """Sem cookie em qualquer rota admin -> 303 redirect para /admin/login."""
    with client(tmp_path) as c:
        # GET na lista de usuarios
        r = c.get("/admin/users", follow_redirects=False)
        assert r.status_code in (302, 303)
        assert "/admin/login" in r.headers.get("location", "")

        # POST criar usuario
        r = c.post("/admin/users", data={"username": "x", "password": "y"}, follow_redirects=False)
        assert r.status_code in (302, 303)

        # PATCH editar (precisa de usuario no DB primeiro)
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "alice")
        r = c.patch("/admin/users/alice", data={"new_username": "bob"}, follow_redirects=False)
        assert r.status_code in (302, 303)

        # DELETE
        r = c.delete("/admin/users/alice", follow_redirects=False)
        assert r.status_code in (302, 303)

        # GET /new form
        r = c.get("/admin/users/new", follow_redirects=False)
        assert r.status_code in (302, 303)

        # GET /edit form
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "bob")
        r = c.get("/admin/users/bob/edit", follow_redirects=False)
        assert r.status_code in (302, 303)

def test_create_user_missing_username_or_password_is_422(tmp_path):
    """Campo ausente no form chega como None (o Starlette trata o valor vazio
    como ausente): o contrato HTTP e 422 do FastAPI, nao o 400 da guarda."""
    with client(tmp_path) as c:
        engine = app.state.recorder.engine
        r1 = c.post("/admin/users", data={"username": "", "password": "senha123"}, cookies=_cookie(), follow_redirects=False)
        r2 = c.post("/admin/users", data={"username": "novo", "password": ""}, cookies=_cookie(), follow_redirects=False)
    assert r1.status_code == 422
    assert r2.status_code == 422
    with Session(engine) as s:
        assert s.execute(select(User)).scalars().all() == []


async def test_create_user_rejects_blank_username_before_touching_the_db(monkeypatch):
    """A guarda `if not username or not password` (admin_users.py:64-65) e
    inalcançavel pelo HTTP: o form nao entrega string vazia (o Starlette
    trata vazio como ausente e o FastAPI responde 422 antes do handler).
    Com username vazio em chamada direta o handler levanta o 400 ANTES de
    tocar na engine, o que prova a ordem da guarda."""
    from fastapi import HTTPException

    from app.routers import admin_users
    from app.routers.admin_users import create_user

    def boom(*args, **kwargs):
        raise AssertionError("banco tocado antes da guarda")

    monkeypatch.setattr(admin_users, "request_engine", boom)
    with pytest.raises(HTTPException) as exc:
        await create_user(request=None, username="", password="senha123", _=None)
    assert exc.value.status_code == 400
    assert exc.value.detail == "Usuario e senha sao obrigatorios"


def test_edit_user_form_unknown_is_404(tmp_path):
    with client(tmp_path) as c:
        r = c.get("/admin/users/nao-existe/edit", cookies=_cookie())
    assert r.status_code == 404
    assert "Usuario nao encontrado" in r.text


def test_update_user_unknown_is_404(tmp_path):
    with client(tmp_path) as c:
        r = c.patch("/admin/users/nao-existe", data={"new_username": "x"}, cookies=_cookie(), follow_redirects=False)
    assert r.status_code == 404
    assert "Usuario nao encontrado" in r.text


def test_delete_user_unknown_is_404(tmp_path):
    with client(tmp_path) as c:
        r = c.delete("/admin/users/nao-existe", cookies=_cookie(), follow_redirects=False)
    assert r.status_code == 404
    assert "Usuario nao encontrado" in r.text
