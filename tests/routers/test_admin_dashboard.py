"""Testes para as paginas e APIs do painel admin."""

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.core.upstream import UpstreamPool
from app.main import app
from app.stats.models import Base

ADMIN_SESSION_SECRET = "segredo-de-teste-do-admin"


def _make_engine(tmp_path: Path):
    db = tmp_path / "admin_dashboard_test.db"
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
    engine = _make_engine(tmp_path)
    _reset_app_state()
    from app.stats.recorder import Recorder
    app.state.recorder = Recorder(engine)
    return TestClient(app)


def _cookie(user_id: int = 1) -> dict:
    from app.core.security import issue_jwt
    token = issue_jwt(user_id=user_id, secret=ADMIN_SESSION_SECRET, ttl_seconds=3600)
    return {"shunt_admin": token}


def _create_user(session: Session, username: str = "admin") -> None:
    from app.core.security import hash_password
    from app.stats.models import User
    user = User(username=username, password_hash=hash_password("senha123"))
    session.add(user)
    session.commit()


# ---- Tests ----


def test_painel_requires_login(tmp_path):
    """GET /admin/painel sem cookie -> 303 redirect para /admin/login."""
    with client(tmp_path) as c:
        r = c.get("/admin/painel", follow_redirects=False)
    assert r.status_code in (302, 303)
    assert "/admin/login" in r.headers.get("location", "")


def test_painel_with_cookie(tmp_path):
    """GET /admin/painel com cookie -> 200, nav presente."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s)
        r = c.get("/admin/painel", cookies=_cookie())
    assert r.status_code == 200
    # Verifica presenca do nav
    assert "<nav" in r.text or "admin-nav" in r.text


def test_requests_requires_login(tmp_path):
    """GET /admin/requests sem cookie -> 303 redirect para /admin/login."""
    with client(tmp_path) as c:
        r = c.get("/admin/requests", follow_redirects=False)
    assert r.status_code in (302, 303)
    assert "/admin/login" in r.headers.get("location", "")


def test_requests_with_cookie(tmp_path):
    """GET /admin/requests com cookie -> 200."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s)
        r = c.get("/admin/requests", cookies=_cookie())
    assert r.status_code == 200


def test_root_redirects_to_painel(tmp_path):
    """GET / -> 302 /admin/painel."""
    with client(tmp_path) as c:
        r = c.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/admin/painel"


def test_api_stats_requires_admin(tmp_path):
    """GET /api/stats sem cookie -> 401."""
    with client(tmp_path) as c:
        r = c.get("/api/stats", headers={"Accept": "application/json"})
    assert r.status_code == 401


def test_api_stats_with_cookie(tmp_path):
    """GET /api/stats com cookie -> 200 (mesmo DB vazio)."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s)
        r = c.get("/api/stats", cookies=_cookie())
    assert r.status_code == 200


def test_api_clear_requires_admin(tmp_path):
    """POST /api/stats/clear sem cookie -> 401."""
    with client(tmp_path) as c:
        r = c.post(
            "/api/stats/clear",
            json={"confirm": True},
            headers={"Accept": "application/json"},
            follow_redirects=False,
        )
    assert r.status_code == 401


def test_api_stream_requires_admin(tmp_path):
    """GET /api/stats/stream sem cookie -> 401 (nao 200 SSE)."""
    with client(tmp_path) as c:
        r = c.get("/api/stats/stream", headers={"Accept": "text/event-stream"}, follow_redirects=False)
    assert r.status_code != 200
    assert r.status_code in (302, 303, 401)
