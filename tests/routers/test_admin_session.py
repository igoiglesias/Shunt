"""Testes para a sessao de admin por JWT cookie.

Reutiliza a mesma estrutura de `tests/routers/test_admin_config.py` (client
com TestClient, app.state injetado, Recorder com SQLite temporario), mas
autentica pelo cookie `shunt_admin` com um JWT em vez do cabecalho
`x-admin-token`. O segredo do JWT mora em `app.state.admin_session_secret`.
"""

from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.config.settings import Settings
from app.core.security import decode_jwt, issue_jwt
from app.core.upstream import UpstreamPool
from app.main import app
from app.stats.models import Base
from app.stats.recorder import Recorder


def _make_engine(tmp_path):
    db = tmp_path / "admin_session_test.db"
    engine = create_engine(f"sqlite:///{db}")
    Base.metadata.create_all(engine)
    return engine


def _reset_app_state():
    app.state.settings = Settings(
        providers={}, models={}, routes=[], default_model=None
    )
    app.state.pool = UpstreamPool(app.state.settings)


def client(tmp_path, *, admin_secret: str = "segredo-da-sessao-admin"):
    """TestClient com SQLite temporario, Recorder e o segredo de JWT definido."""
    engine = _make_engine(tmp_path)
    _reset_app_state()
    app.state.recorder = Recorder(engine)
    app.state.admin_session_secret = admin_secret
    return TestClient(app)


def _cookie(user_id: int = 1, secret: str = "segredo-da-sessao-admin", ttl: int = 3600) -> dict:
    """Monta o cookie `shunt_admin` com um JWT valido."""
    token = issue_jwt(user_id=user_id, secret=secret, ttl_seconds=ttl)
    return {"shunt_admin": token}


def test_admin_page_requires_login_when_no_cookie(monkeypatch, tmp_path):
    """GET an admin HTML page with NO cookie -> redirect to /admin/login."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.get("/admin/config", follow_redirects=False)
    assert r.status_code in (302, 303)
    assert "/admin/login" in r.headers.get("location", "")


def test_admin_page_ok_with_valid_cookie(monkeypatch, tmp_path):
    """Set a valid JWT cookie -> the previously-protected route returns 200."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.get("/admin/config", cookies=_cookie(), follow_redirects=False)
    assert r.status_code == 200


def test_admin_htmx_401_when_no_cookie(monkeypatch, tmp_path):
    """An HTMX admin request (HX-Request: true) with no cookie -> 401."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.get(
            "/admin/config",
            headers={"HX-Request": "true"},
            follow_redirects=False,
        )
    assert r.status_code == 401


def test_admin_page_expired_cookie_redirects(monkeypatch, tmp_path):
    """Cookie from a JWT whose exp is in the past -> redirect to /admin/login."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.get(
            "/admin/config",
            cookies=_cookie(ttl=-1),
            follow_redirects=False,
        )
    assert r.status_code in (302, 303)
    assert "/admin/login" in r.headers.get("location", "")


def test_admin_page_wrong_secret_cookie_redirects(monkeypatch, tmp_path):
    """Cookie signed with a different secret -> redirect to /admin/login."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        # JWT with a secret that does not match app.state.admin_session_secret.
        token = issue_jwt(user_id=1, secret="outro-segredo", ttl_seconds=3600)
        r = c.get("/admin/config", cookies={"shunt_admin": token}, follow_redirects=False)
    assert r.status_code in (302, 303)
    assert "/admin/login" in r.headers.get("location", "")


def test_decode_jwt_claims_roundtrip():
    """issue then decode; claims["sub"] == <user_id>."""
    token = issue_jwt(user_id=7, secret="segredo-da-sessao-admin", ttl_seconds=3600)
    claims = decode_jwt(token, "segredo-da-sessao-admin")
    assert claims is not None
    assert claims["sub"] == 7


def test_decode_jwt_expired_returns_none():
    """issue with negative ttl so exp is in the past; decode returns None."""
    token = issue_jwt(user_id=1, secret="segredo-da-sessao-admin", ttl_seconds=-1)
    assert decode_jwt(token, "segredo-da-sessao-admin") is None


def test_decode_jwt_tampered_returns_none():
    """flip a character in the signature part; decode returns None."""
    token = issue_jwt(user_id=1, secret="segredo-da-sessao-admin", ttl_seconds=3600)
    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
    assert decode_jwt(tampered, "segredo-da-sessao-admin") is None


def test_decode_jwt_wrong_secret_returns_none():
    """decode with a different secret -> None."""
    token = issue_jwt(user_id=1, secret="segredo-correto", ttl_seconds=3600)
    assert decode_jwt(token, "segredo-errado") is None