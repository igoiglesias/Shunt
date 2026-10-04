"""A tela de configuracao depois de editar uma rota, num navegador HEADLESS.

Este arquivo existe porque o defeito que ele cobre nao e de logica, e de
estrutura: _routes.html abria a div da ancora `#routes-list` e nao a fechava,
o navegador aninhava nela o formulario de edicao e todas as secoes seguintes, e
o primeiro `outerHTML` (Salvar/Excluir/Reordenar) engolia tudo. O usuario via
os botoes pararem de responder ate dar refresh. So um navegador real anuncia o
DOM que o swap deixou.
"""

import os
import socket
import subprocess
import time

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config.config import ADMIN_COOKIE
from app.core.security import hash_password, issue_jwt
from app.stats.models import Base, User

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright

SESSION_SECRET = "segredo-fixo-do-teste-de-config-de-rotas"
ADMIN_PASSWORD = "senha-do-teste"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def seed(path) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        # So o admin: o catalogo (providers, models, routes) e semeado pelo
        # proprio lifespan no primeiro boot com o banco vazio.
        session.add(User(username="admin", password_hash=hash_password(ADMIN_PASSWORD)))
        session.commit()
    engine.dispose()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    database = tmp_path_factory.mktemp("config-rotas") / "stats.db"
    seed(database)
    port = free_port()
    process = subprocess.Popen(
        ["uv", "run", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        env={
            **os.environ,
            "TURSO_DATABASE_URL": f"sqlite+pysqlite:///{database}",
            "ADMIN_SESSION_SECRET": SESSION_SECRET,
        },
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
        pytest.fail("o servidor de configuracao nao subiu")
    yield base
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch()  # headless
        yield instance
        instance.close()


def abrir(browser, base):
    """Pagina de configuracao autenticada, com os erros de JS coletados."""
    page = browser.new_page(viewport={"width": 1500, "height": 1000})
    page.set_default_timeout(8_000)
    page.context.add_cookies(
        [{"name": ADMIN_COOKIE, "value": issue_jwt(1, SESSION_SECRET, 3600), "url": base}]
    )
    problemas = []
    page.on("pageerror", lambda erro: problemas.append(str(erro)))
    # Excluir usa hx-confirm, que abre um confirm() nativo; sem aceitar a
    # caixa o clique e descartado e a rota nao sai do lugar.
    page.on("dialog", lambda dialogo: dialogo.accept())
    page.goto(f"{base}/admin/config", wait_until="domcontentloaded")
    page.wait_for_selector("#routes-list table tr")
    return page, problemas


def alvos_vivos(page) -> list[dict]:
    """Botoes cujo hx-target nao resolve mais no DOM atual."""
    return page.evaluate(
        """() => [...document.querySelectorAll('[hx-target]')]
          .filter((el) => !document.querySelector(el.getAttribute('hx-target')))
          .map((el) => ({botao: el.textContent.trim(), alvo: el.getAttribute('hx-target')}))"""
    )


def test_editar_uma_rota_nao_quebra_os_botoes_da_tela(browser, server):
    """Editar e salvar uma rota mantem todos os alvos HTMX da tela vivos.

    Sem o fechamento da div da ancora, o swap de #routes-list removia
    #route-form junto, e todo botao que aponta para ele (Nova Rota e todos
    os Editar da tabela) ficava morto ate o refresh.
    """
    page, problemas = abrir(browser, server)

    # A ancora da lista e irma do formulario, nao ascendente.
    aninhados = page.evaluate(
        """() => document.getElementById('routes-list')
             .contains(document.getElementById('route-form'))"""
    )
    assert not aninhados, "#routes-list engole #route-form no load da pagina"

    editaveis_antes = page.locator("#routes-list button", has_text="Editar").count()

    page.locator("#routes-list button", has_text="Editar").first.click()
    page.wait_for_selector("#route-form form")
    page.locator("#route-form button[type=submit]").click()
    page.wait_for_timeout(400)

    # A lista renasceu com a ancora no lugar e os mesmos botoes.
    assert page.locator("#routes-list").count() == 1
    assert page.locator("#routes-list button", has_text="Editar").count() == editaveis_antes
    # E o formulario continua existindo, nao foi engolido pelo swap.
    assert page.locator("#route-form").count() == 1
    assert alvos_vivos(page) == []
    assert problemas == []
    page.close()


def test_excluir_uma_rota_nao_quebra_os_botoes_da_tela(browser, server):
    """O botao Excluir usa o mesmo swap outerHTML e tem a mesma exigencia."""
    page, problemas = abrir(browser, server)

    editaveis_antes = page.locator("#routes-list button", has_text="Editar").count()

    page.locator("#routes-list form.inline-form button").first.click()
    page.wait_for_function(
        "(quantidade) => document.querySelectorAll('#routes-list form.inline-form').length"
        " < quantidade",
        arg=editaveis_antes,
    )

    assert page.locator("#routes-list").count() == 1
    assert page.locator("#route-form").count() == 1
    assert alvos_vivos(page) == []
    assert problemas == []
    page.close()


def test_reordenar_rotas_nao_quebra_os_botoes_da_tela(browser, server):
    """Aplicar a ordem da tabela e o terceiro swap que toca a ancora."""
    page, problemas = abrir(browser, server)

    page.locator("#routes-list button", has_text="Aplicar ordem").click()
    page.wait_for_timeout(400)

    assert page.locator("#routes-list").count() == 1
    assert page.locator("#route-form").count() == 1
    assert alvos_vivos(page) == []
    assert problemas == []
    page.close()
