"""A home: o painel, e a checagem de vida que saiu dela."""

import time

from fastapi.testclient import TestClient

from app.core.upstream import UpstreamPool
from app.main import DASHBOARD, app
from app.stats.recorder import Recorder
from tests.core.test_dispatcher import SETTINGS


def client():
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    app.state.recorder = Recorder(None)
    return TestClient(app)


def test_the_home_serves_the_panel():
    with client() as c:
        response = c.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Shunt</title>" in response.text


def test_the_liveness_check_moved_to_health_and_still_answers():
    with client() as c:
        response = c.get("/health")
    assert response.json() == {"status": "ok"}


def test_the_panel_reads_its_data_from_the_two_api_routes():
    """Se um dia alguem renomear a rota, o painel tem de quebrar aqui e nao no navegador."""
    page = DASHBOARD.read_text(encoding="utf-8")
    assert "/api/stats?window=" in page
    assert 'EventSource("/api/stats/stream")' in page


def test_the_panel_escapes_what_comes_from_the_provider():
    """Nome de modelo e nome de ferramenta vem de fora; nenhum pode virar HTML."""
    page = DASHBOARD.read_text(encoding="utf-8")
    assert "const esc =" in page
    assert "esc(event.requested_model" in page or "esc(r[key])" in page


def test_the_panel_respects_a_reader_who_asked_for_less_motion():
    assert "prefers-reduced-motion" in DASHBOARD.read_text(encoding="utf-8")


async def test_a_database_that_hangs_at_boot_does_not_hang_the_proxy(monkeypatch):
    """Medido contra uma porta morta: `create_all` nao levanta, ele PENDURA.

    Um extra que impede o servico de subir deixou de ser um extra. O boot
    desiste do banco e serve sem persistencia.
    """
    from app import main

    def never_answers():
        # Curto o bastante para nao segurar a suite: o prazo do boot e menor ainda.
        time.sleep(2)

    monkeypatch.setattr(main, "build_engine", never_answers)
    monkeypatch.setattr(main, "ENGINE_BOOT_TIMEOUT", 0.05)
    started = time.monotonic()
    engine = await main._engine_or_none()
    assert engine is None
    assert time.monotonic() - started < 5


def test_the_audit_screen_is_served_and_links_back_to_the_panel():
    with client() as c:
        response = c.get("/requests")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Shunt — requisições</title>" in response.text
    assert 'href="/"' in response.text


def test_the_panel_links_to_the_audit_screen():
    from app.main import DASHBOARD

    assert 'href="/requests"' in DASHBOARD.read_text(encoding="utf-8")


def test_the_audit_screen_reads_the_three_audit_routes():
    from app.main import AUDIT

    page = AUDIT.read_text(encoding="utf-8")
    assert "/api/requests?" in page
    assert "/api/requests/export?" in page
    assert "/api/requests/${encodeURIComponent(requestId)}" in page


def test_neither_page_is_cached_by_the_browser():
    """Medido: com o proxy reiniciado e a pagina nova no disco, o navegador
    seguia desenhando a antiga -- sem o menu e sem os botoes novos.

    HTML sem `Cache-Control` e guardado pela heuristica do navegador, e estas
    paginas mudam a cada deploy sem trocar de URL.
    """
    with client() as c:
        for path in ("/", "/requests"):
            response = c.get(path)
            assert response.status_code == 200
            assert "no-store" in response.headers.get("cache-control", ""), path
