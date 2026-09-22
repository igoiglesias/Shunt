"""O painel e a tela de requisicoes, agora dentro do admin e protegidos por sessao."""

import time

from fastapi.testclient import TestClient

from app.core.security import issue_jwt
from app.core.upstream import UpstreamPool
from app.main import app
from app.stats.recorder import Recorder
from tests.core.test_dispatcher import SETTINGS

ADMIN_SESSION_SECRET = "segredo-de-teste-do-admin"


def _reset_app_state():
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    app.state.recorder = Recorder(None)
    app.state.admin_session_secret = ADMIN_SESSION_SECRET


def _cookie(user_id: int = 1) -> dict:
    return {"shunt_admin": issue_jwt(user_id=user_id, secret=ADMIN_SESSION_SECRET, ttl_seconds=3600)}


def client():
    _reset_app_state()
    return TestClient(app)


def test_root_redirects_to_admin_painel():
    """GET / redireciona para /admin/painel (que por sua vez vai ao login sem cookie)."""
    with client() as c:
        r = c.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers.get("location") == "/admin/painel"


def test_painel_requires_login():
    """Painel sem cookie redireciona para /admin/login."""
    with client() as c:
        r = c.get("/admin/painel", follow_redirects=False)
    assert r.status_code == 303
    assert "/admin/login" in r.headers.get("location", "")


def test_painel_with_session_cookie():
    """Painel com cookie valido devolve 200 e o HTML do painel."""
    with client() as c:
        r = c.get("/admin/painel", cookies=_cookie())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Shunt" in r.text
    # Menu compartilhado presente
    assert "Painel" in r.text
    assert "Requisições" in r.text
    assert "Configuração" in r.text
    assert "Usuários" in r.text
    assert "Tokens" in r.text


def test_requests_requires_login():
    """Tela de requisicoes sem cookie redireciona para /admin/login."""
    with client() as c:
        r = c.get("/admin/requests", follow_redirects=False)
    assert r.status_code == 303
    assert "/admin/login" in r.headers.get("location", "")


def test_requests_with_session_cookie():
    """Tela de requisicoes com cookie valido devolve 200 e o HTML da auditoria."""
    with client() as c:
        r = c.get("/admin/requests", cookies=_cookie())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Shunt" in r.text
    assert "Painel" in r.text
    assert "Requisições" in r.text
    assert "Configuração" in r.text
    assert "Usuários" in r.text
    assert "Tokens" in r.text


def test_health_still_public():
    """/health continua publico e sem autenticacao."""
    with client() as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_dashboard_apis_require_admin():
    """As APIs de dados do painel exigem sessao (401 sem cookie)."""
    with client() as c:
        for path in ("/api/stats", "/api/stats/stream"):
            r = c.get(path)
            assert r.status_code == 401, f"{path} should be 401, got {r.status_code}"
        r = c.post("/api/stats/clear")
        assert r.status_code == 401, f"/api/stats/clear should be 401, got {r.status_code}"


def test_dashboard_apis_with_session():
    """Com sessao valida, as APIs respondem 200 (mesmo sem dados)."""
    with client() as c:
        r = c.get("/api/stats", cookies=_cookie())
    assert r.status_code == 200
    data = r.json()
    assert "totals" in data
    assert "health" in data


async def test_a_database_that_hangs_at_boot_does_not_hang_the_proxy(monkeypatch):
    """Medido contra uma porta morta: `create_all` nao levanta, ele PENDURA.

    Um extra que impede o servico de subir deixou de ser um extra. O boot
    desiste do banco e serve sem persistencia.
    """
    from app import main

    def never_answers():
        time.sleep(2)

    monkeypatch.setattr(main, "build_engine", never_answers)
    monkeypatch.setattr(main, "ENGINE_BOOT_TIMEOUT", 0.05)
    started = time.monotonic()
    engine = await main._engine_or_none()
    assert engine is None
    assert time.monotonic() - started < 5


def test_neither_page_is_cached_by_the_browser():
    """Tanto o painel quanto a tela de requisicoes sao servidos com no-store."""
    with client() as c:
        for path in ("/admin/painel", "/admin/requests"):
            r = c.get(path, cookies=_cookie())
            assert r.status_code == 200
            assert "no-store" in r.headers.get("cache-control", ""), path


def test_panel_shows_called_tools_first():
    """Consistencia com a tela de requisicoes: ferramentas chamadas aparecem primeiro."""
    with client() as c:
        r = c.get("/admin/painel", cookies=_cookie())
    page = r.text
    assert "const usadas = tools.filter((t) => t.called > 0);" in page
    assert "oferecidas e nunca chamadas" in page