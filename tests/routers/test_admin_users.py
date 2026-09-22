"""Testes CRUD de usuarios do painel admin.

Todos os testes passam por autenticacao JWT cookie (COOKIE) e
usam engine SQLite temporaria por teste.
"""

from pathlib import Path

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