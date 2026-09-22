"""Testes para fluxo de autenticacao e login do painel admin.

Testa a criacao do primeiro admin, login com usuario existente, e logout.
"""

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.core.security import issue_jwt
from app.core.upstream import UpstreamPool
from app.main import app
from app.stats.models import Base, User

ADMIN_SESSION_SECRET = "segredo-de-teste-do-admin"


def _make_engine(tmp_path: Path):
    db = tmp_path / "admin_auth_test.db"
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
    """Return a TestClient with a temporary SQLite DB and Recorder."""
    engine = _make_engine(tmp_path)
    _reset_app_state()
    from app.stats.recorder import Recorder
    app.state.recorder = Recorder(engine)
    return TestClient(app)


def _cookie(user_id: int = 1, secret: str = ADMIN_SESSION_SECRET, ttl: int = 3600) -> dict:
    """Monta o cookie `shunt_admin` com um JWT valido."""
    token = issue_jwt(user_id=user_id, secret=secret, ttl_seconds=ttl)
    return {"shunt_admin": token}


def _create_user(session: Session, username: str = "admin", password: str = "senha123") -> User:
    """Cria um usuario na base de teste."""
    from app.core.security import hash_password
    user = User(username=username, password_hash=hash_password(password))
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


# ---- Tests ----


def test_login_page_zero_users(tmp_path):
    """GET /admin/login com DB vazia -> 200, modo criacao (contem 'Primeiro acesso')."""
    with client(tmp_path) as c:
        r = c.get("/admin/login")
    assert r.status_code == 200
    assert "Primeiro acesso" in r.text


def test_login_page_existing_users(tmp_path):
    """GET /admin/login quando usuarios existem -> 200, modo login (nao contem 'Primeiro acesso')."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s)
        r = c.get("/admin/login")
    assert r.status_code == 200
    assert "Primeiro acesso" not in r.text


def test_create_first_admin(tmp_path):
    """POST /admin/login com username+pw+confirm no modo criacao -> 303 /admin/painel, cookie shunt_admin set."""
    with client(tmp_path) as c:
        r = c.post(
            "/admin/login",
            data={"username": "primeiro", "password": "senha123", "confirm": "senha123"},
            follow_redirects=False,
        )
    assert r.status_code == 303
    assert r.headers["location"] == "/admin/painel"
    cookie = r.cookies.get("shunt_admin")
    assert cookie is not None
    # Verifica que o usuario foi persistido
    with Session(app.state.recorder.engine) as s:
        user = s.execute(select(User).where(User.username == "primeiro")).scalar_one()
    assert user is not None


def test_login_success(tmp_path):
    """Usuario existente, senha valida -> 303 /admin/painel, cookie shunt_admin set."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "testuser", "senha123")
        r = c.post(
            "/admin/login",
            data={"username": "testuser", "password": "senha123"},
            follow_redirects=False,
        )
    assert r.status_code == 303
    assert r.headers["location"] == "/admin/painel"
    cookie = r.cookies.get("shunt_admin")
    assert cookie is not None


def test_login_wrong_password(tmp_path):
    """Usuario existente, senha errada -> 401, mensagem de erro."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "testuser", "senha123")
        r = c.post(
            "/admin/login",
            data={"username": "testuser", "password": "errada"},
        )
    assert r.status_code == 401
    assert "Usuario ou senha invalidos" in r.text


def test_login_wrong_username(tmp_path):
    """Usuario inexistente -> 401."""
    with client(tmp_path) as c:
        # First create a user, then test with wrong username
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "existing_user", "senha123")
        r = c.post(
            "/admin/login",
            data={"username": "inexistente", "password": "qualquer"},
        )
    assert r.status_code == 401


def test_logout_clears_cookie(tmp_path):
    """Login -> POST /admin/logout -> 303 /admin/login, cookie deletado (max_age=0)."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s)
        # Primeiro, faz login para obter o cookie
        r = c.post(
            "/admin/login",
            data={"username": "admin", "password": "senha123"},
            follow_redirects=False,
        )
        assert r.status_code == 303
        # Agora faz logout
        r = c.post("/admin/logout", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/admin/login"
    # O cookie deve ser marcado para exclusão pelo navegador.
    set_cookie = r.headers.get("set-cookie", "")
    assert "shunt_admin" in set_cookie
    assert "Max-Age=0" in set_cookie or "max-age=0" in set_cookie.lower()
    assert "Expires=" in set_cookie or "expires=" in set_cookie.lower()


def test_session_persists(tmp_path):
    """Cookie de login funciona em /admin/painel, /admin/users, /admin/tokens, /admin/config."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "admin", "senha123")
        # Faz login para obter o cookie
        r = c.post(
            "/admin/login",
            data={"username": "admin", "password": "senha123"},
            follow_redirects=False,
        )
        cookie = r.cookies.get("shunt_admin")
        assert cookie is not None

        # Agora testa que o cookie funciona em todas as rotas protegidas
        for path in ["/admin/painel", "/admin/users", "/admin/tokens", "/admin/config"]:
            r = c.get(path, cookies={"shunt_admin": cookie})
            assert r.status_code == 200, f"Falhou para {path}"
