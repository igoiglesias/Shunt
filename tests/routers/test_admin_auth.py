"""Testes para fluxo de autenticacao e login do painel admin.

Testa a criacao do primeiro admin, login com usuario existente, e logout.
"""

from pathlib import Path

import pytest
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


def test_login_page_without_database_offers_login_not_create(tmp_path):
    """Sem banco a tela e de LOGIN, nao de criacao.

    O POST recusa criar admin sem persistencia (503); oferecer a tela de
    "criar" quando o banco esta desligado era uma porta que se fecha no envio.
    """
    _reset_app_state()  # nao instala recorder
    if hasattr(app.state, "recorder"):
        del app.state.recorder
    with TestClient(app) as c:
        r = c.get("/admin/login")
    assert r.status_code == 200
    assert "Primeiro acesso" not in r.text
    assert "Entrar" in r.text


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


def test_first_admin_race_returns_clean_error_not_500(tmp_path, monkeypatch):
    """Perdeu a corrida do primeiro admin (UNIQUE colidiu) -> 400, nao 500.

    Em `make prod` ha varios workers: dois veem a tabela vazia ao mesmo tempo,
    e o segundo insert colide com a UNIQUE de `username`. O ramo de perda tem
    que devolver uma mensagem, nao um stack trace. Simula-se a colidendo no
    primeiro `commit`, que e exatamente o que o worker perdedor encontra.
    """
    from sqlalchemy.exc import IntegrityError

    with client(tmp_path):
        real_session = Session

        def racing_session(bind):
            real = real_session(bind)
            state = {"first_commit": True}

            def committing(*args, **kwargs):
                if state["first_commit"]:
                    state["first_commit"] = False
                    raise IntegrityError("colisao", {}, None)
                return real.commit(*args, **kwargs)

            real.commit = committing
            return real

        monkeypatch.setattr("app.routers.admin_auth.Session", racing_session)
        r = TestClient(app).post(
            "/admin/login",
            data={"username": "x", "password": "senha123", "confirm": "senha123"},
            follow_redirects=False,
        )
    assert r.status_code == 400
    assert "Já existe um administrador" in r.text


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


# ---- next= : volta a pagina pedida depois do login ----

# Vetores de open redirect que o POST tem de recusar (espelham o RED 2 de
# `tests/core/test_auth.py`); cada um cai no painel.
_BAD_NEXT = [
    "//evil.com",
    "https://evil.com",
    "http:evil.com",
    "/\\evil.com",
    "\\\\evil.com",
    "javascript:alert(1)",
    "/%2F%2Fevil.com",
    "%2F%2Fevil.com",
    "/%5Cevil.com",
    "/api/stats",
    "/docs",
    "/",
    "/admin/login",
    "/admin/login?next=/admin",
    "/admin\n",
    " /admin",
]


def _hidden_next(html: str) -> str | None:
    """Valor do campo oculto `next` como o navegador o envia (entidades decodificadas)."""
    import html as html_lib
    import re

    m = re.search(r'<input type="hidden" name="next" value="([^"]*)">', html)
    return html_lib.unescape(m.group(1)) if m else None


def test_protected_page_redirects_to_login_with_next(tmp_path):
    """GET /admin/requests?q=x sem cookie -> 303 para o login levando o caminho pedido.

    A codificacao e `quote(path?query, safe="/")`: `?`, `=` e `&` do destino
    viram `%3F`, `%3D`, `%26` para nao se misturarem a query do login.
    """
    with client(tmp_path) as c:
        r = c.get("/admin/requests?q=x&page=2", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/admin/login?next=/admin/requests%3Fq%3Dx%26page%3D2"


def test_protected_page_without_query_has_no_question_mark(tmp_path):
    """Sem query string, o `next` e so o caminho (sem `?` pendurado)."""
    with client(tmp_path) as c:
        r = c.get("/admin/requests", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/admin/login?next=/admin/requests"


def test_api_without_cookie_is_401_without_location(tmp_path):
    """GET /api/stats sem cookie -> 401 puro: API nao ganha redirect nem next."""
    with client(tmp_path) as c:
        r = c.get("/api/stats", follow_redirects=False)
    assert r.status_code == 401
    assert "location" not in r.headers


def test_login_page_carries_next_in_hidden_field(tmp_path):
    """GET /admin/login?next=/admin/requests -> campo oculto com esse valor."""
    with client(tmp_path) as c:
        r = c.get("/admin/login", params={"next": "/admin/requests?q=1"})
    assert r.status_code == 200
    assert _hidden_next(r.text) == "/admin/requests?q=1"


def test_login_page_without_next_points_hidden_field_to_panel(tmp_path):
    """Sem `next`, o campo oculto leva ao painel (o destino de sempre)."""
    with client(tmp_path) as c:
        r = c.get("/admin/login")
    assert _hidden_next(r.text) == "/admin/painel"


def test_login_page_discards_unsafe_next(tmp_path):
    """Um `next` fora da regra nao chega ao HTML: o campo leva ao painel."""
    with client(tmp_path) as c:
        r = c.get("/admin/login", params={"next": "//evil.com"})
    assert _hidden_next(r.text) == "/admin/painel"
    assert "evil.com" not in r.text


def test_login_page_next_script_injection_not_raw(tmp_path):
    """`"><script>` no next nao sai cru no HTML."""
    payload = '/admin/x"><script>alert(1)</script>'
    with client(tmp_path) as c:
        r = c.get("/admin/login", params={"next": payload})
    assert r.status_code == 200
    assert '"><script>' not in r.text


def test_login_success_follows_next(tmp_path):
    """Login valido com next=/admin/requests?q=1 -> 303 para esse caminho."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "testuser", "senha123")
        r = c.post(
            "/admin/login",
            data={"username": "testuser", "password": "senha123", "next": "/admin/requests?q=1"},
            follow_redirects=False,
        )
    assert r.status_code == 303
    assert r.headers["location"] == "/admin/requests?q=1"
    assert r.cookies.get("shunt_admin") is not None


@pytest.mark.parametrize("bad", _BAD_NEXT)
def test_login_success_with_unsafe_next_goes_to_panel(tmp_path, bad):
    """Cada vetor de open redirect no POST -> 303 para /admin/painel."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "testuser", "senha123")
        r = c.post(
            "/admin/login",
            data={"username": "testuser", "password": "senha123", "next": bad},
            follow_redirects=False,
        )
    assert r.status_code == 303
    assert r.headers["location"] == "/admin/painel"


def test_first_admin_follows_next(tmp_path):
    """Primeiro acesso com next -> 303 para o next."""
    with client(tmp_path) as c:
        r = c.post(
            "/admin/login",
            data={
                "username": "primeiro", "password": "senha123",
                "confirm": "senha123", "next": "/admin/users",
            },
            follow_redirects=False,
        )
    assert r.status_code == 303
    assert r.headers["location"] == "/admin/users"


def test_first_admin_with_unsafe_next_goes_to_panel(tmp_path):
    """Primeiro acesso com next fora da regra -> 303 para /admin/painel."""
    with client(tmp_path) as c:
        r = c.post(
            "/admin/login",
            data={
                "username": "primeiro", "password": "senha123",
                "confirm": "senha123", "next": "https://evil.com",
            },
            follow_redirects=False,
        )
    assert r.status_code == 303
    assert r.headers["location"] == "/admin/painel"


def test_wrong_password_keeps_next_hidden(tmp_path):
    """Senha errada com next -> 401 e o campo oculto continua com o next."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "testuser", "senha123")
        r = c.post(
            "/admin/login",
            data={"username": "testuser", "password": "errada", "next": "/admin/requests?q=1"},
        )
    assert r.status_code == 401
    assert _hidden_next(r.text) == "/admin/requests?q=1"


def test_wrong_password_sanitizes_next_in_rerender(tmp_path):
    """Senha errada com next hostil -> o re-render nao devolve o valor cru."""
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "testuser", "senha123")
        r = c.post(
            "/admin/login",
            data={"username": "testuser", "password": "errada", "next": "//evil.com"},
        )
    assert r.status_code == 401
    assert _hidden_next(r.text) == "/admin/painel"


def test_first_admin_validation_error_keeps_next_hidden(tmp_path):
    """Primeiro acesso com senhas diferentes -> 400 e o next continua no form."""
    with client(tmp_path) as c:
        r = c.post(
            "/admin/login",
            data={
                "username": "primeiro", "password": "a",
                "confirm": "b", "next": "/admin/users",
            },
        )
    assert r.status_code == 400
    assert _hidden_next(r.text) == "/admin/users"


def test_first_admin_race_keeps_next_hidden(tmp_path, monkeypatch):
    """Perdeu a corrida do primeiro admin com next -> 400 e o next continua no form."""
    from sqlalchemy.exc import IntegrityError

    with client(tmp_path):
        real_session = Session

        def racing_session(bind):
            real = real_session(bind)

            def committing(*args, **kwargs):
                raise IntegrityError("colisao", {}, None)

            real.commit = committing
            return real

        monkeypatch.setattr("app.routers.admin_auth.Session", racing_session)
        r = TestClient(app).post(
            "/admin/login",
            data={
                "username": "x", "password": "senha123",
                "confirm": "senha123", "next": "/admin/users",
            },
            follow_redirects=False,
        )
    assert r.status_code == 400
    assert _hidden_next(r.text) == "/admin/users"


def test_login_round_trip_returns_to_requested_page(tmp_path):
    """Pagina protegida sem cookie -> login -> POST com o campo oculto -> volta a mesma URL."""
    wanted = "/admin/requests?q=x&page=2"
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _create_user(s, "testuser", "senha123")
        first = c.get(wanted, follow_redirects=False)
        login = c.get(first.headers["location"])
        hidden = _hidden_next(login.text)
        r = c.post(
            "/admin/login",
            data={"username": "testuser", "password": "senha123", "next": hidden},
            follow_redirects=False,
        )
    assert hidden == wanted
    assert r.status_code == 303
    assert r.headers["location"] == wanted
