"""O painel conferido num navegador HEADLESS, contra um banco semeado.

Nunca abre janela na maquina de ninguem: `chromium.launch()` e headless por
padrao e nenhum teste aqui passa `headless=False`.

O que este arquivo prende e o que um teste de rota nao alcanca: se o JavaScript
levanta, se o SVG desenha, se a pagina rola de lado no celular, e se o numero
que aparece na tela e o mesmo que a API respondeu.
"""

import json
import os
import socket
import subprocess
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.stats.models import Base, RequestEvent

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def seed(path) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{path}")
    Base.metadata.create_all(engine)
    now = datetime.now(UTC)
    rows = []
    for index in range(30):
        failed = index % 10 == 0
        rows.append(
            RequestEvent(
                request_id=f"b{index}",
                started_at=now - timedelta(minutes=index * 3),
                route="/v1/messages",
                dialect="anthropic",
                stream=index % 2 == 0,
                requested_model="claude-sonnet-4-5",
                rule="family",
                matched="sonnet",
                provider=None if failed else "groq",
                candidate_model=None if failed else "openai/gpt-oss-120b",
                status=429 if failed else 200,
                error_type="rate_limit_error" if failed else None,
                input_tokens=100,
                output_tokens=50,
                ttft_ms=200 if index % 2 == 0 and not failed else None,
                duration_ms=500,
                attempts=["free: 429 (attempt 1)"] if index % 5 == 0 else [],
                fell_back=index % 5 == 0 and not failed,
                tools_offered=["read"],
                tools_called=["read"] if index % 3 == 0 else [],
                thinking_blocks=1,
            )
        )
    with Session(engine) as session:
        session.add_all(rows)
        session.commit()
    engine.dispose()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    database = tmp_path_factory.mktemp("painel") / "stats.db"
    seed(database)
    port = free_port()
    process = subprocess.Popen(
        ["uv", "run", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        env={**os.environ, "TURSO_DATABASE_URL": f"sqlite+pysqlite:///{database}"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.3)
    else:
        process.terminate()
        pytest.fail("o servidor do painel nao subiu")
    yield base
    process.terminate()
    process.wait(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch()  # headless
        yield instance
        instance.close()


def open_panel(browser, base, width, height):
    page = browser.new_page(viewport={"width": width, "height": height})
    problems = []
    page.on("pageerror", lambda error: problems.append(str(error)))
    page.on(
        "console",
        lambda message: problems.append(message.text) if message.type == "error" else None,
    )
    page.goto(base, wait_until="networkidle")
    page.wait_for_selector("#figures .figure")
    return page, problems


def test_the_panel_draws_without_a_single_console_error(browser, server):
    page, problems = open_panel(browser, server, 1440, 1000)
    assert problems == []
    assert page.locator("#flow svg").count() == 1
    assert page.locator("#series path").count() == 2
    page.close()


def test_the_numbers_on_screen_are_the_numbers_the_api_answered(browser, server):
    """Painel bonito com numero errado e pior do que painel nenhum."""
    api = httpx.get(f"{server}/api/stats?window=24", timeout=10).json()
    page, _ = open_panel(browser, server, 1440, 1000)
    shown = page.evaluate(
        """() => ({
            requests: document.querySelector('#figures .figure.requests .value').textContent,
            models: [...document.querySelectorAll('#models tbody tr')].map(
                r => [...r.children].slice(0, 2).map(c => c.textContent.trim())),
            errors: [...document.querySelectorAll('#errors tbody tr td:last-child')].map(
                c => Number(c.textContent)),
        })"""
    )
    page.close()
    assert shown["requests"] == str(api["totals"]["requests"])
    assert shown["models"][0][0] == api["by_model"][0]["model"]
    assert int(shown["models"][0][1]) == api["by_model"][0]["requests"]
    assert sum(shown["errors"]) == sum(row["requests"] for row in api["errors"])


def test_the_panel_never_scrolls_sideways_on_a_phone(browser, server):
    page, problems = open_panel(browser, server, 390, 844)
    measured = page.evaluate(
        """() => ({
            scrollWidth: document.documentElement.scrollWidth,
            innerWidth: window.innerWidth,
            flowLabel: document.querySelector('#flow text.flow-label')?.getBoundingClientRect(),
        })"""
    )
    page.close()
    assert problems == []
    assert measured["scrollWidth"] <= measured["innerWidth"] + 1
    # Rotulo legivel: menos de 9 px de altura na tela e texto que ninguem le.
    assert measured["flowLabel"]["height"] >= 9, measured["flowLabel"]


def test_a_request_served_right_now_lands_on_the_tape(browser, server):
    """O SSE: o painel aberto tem de mostrar a requisicao chegando."""
    page, _ = open_panel(browser, server, 1440, 1000)
    before = page.locator("#tape li").count()
    httpx.get(f"{server}/v1/models", timeout=10)
    page.wait_for_function(f"document.querySelectorAll('#tape li').length > {before}", timeout=10_000)
    first = page.locator("#tape li").first.inner_text()
    page.close()
    assert "models" in first


def test_the_window_buttons_reload_the_summary(browser, server):
    page, problems = open_panel(browser, server, 1440, 1000)
    page.get_by_role("button", name="1h").click()
    page.wait_for_function("document.querySelector('#figures .figure .value').textContent !== ''")
    pressed = page.evaluate(
        """() => [...document.querySelectorAll('#windows button')]
            .filter(b => b.getAttribute('aria-pressed') === 'true').map(b => b.textContent)"""
    )
    page.close()
    assert pressed == ["1h"]
    assert problems == []


def test_the_panel_says_so_when_there_is_no_database(browser, tmp_path_factory):
    """Sem Turso o painel nao pode quebrar: ele avisa e segue ao vivo."""
    port = free_port()
    process = subprocess.Popen(
        ["uv", "run", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        env={k: v for k, v in os.environ.items() if k != "TURSO_DATABASE_URL"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.3)
        page, problems = open_panel(browser, base, 1200, 900)
        text = page.inner_text("body")
        page.close()
    finally:
        process.terminate()
        process.wait(timeout=10)
    assert problems == []
    assert "sem banco" in text
    assert "TURSO_DATABASE_URL" in text


def test_the_api_answers_the_same_json_the_panel_parses(server):
    """Contrato entre as duas metades, preso sem navegador nenhum."""
    body = httpx.get(f"{server}/api/stats", timeout=10).json()
    assert json.loads(json.dumps(body)) == body
    assert set(body) >= {"totals", "per_hour", "by_model", "chain", "tools", "recent", "health"}
