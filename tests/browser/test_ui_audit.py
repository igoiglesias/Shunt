"""As duas telas passam pela varredura, em todas as larguras.

Este arquivo existe porque olhar o print nao bastou: numa rodada em que eu disse
que estava bom, a varredura encontrou 105 problemas -- contraste abaixo do
minimo em toda tabela, texto sobreposto no diagrama, alvo de clique de 21 px.
"""

import pytest

from tests.browser.test_audit_browser import browser, seed, server  # noqa: F401
from tests.browser.ui_audit import auditar, descrever

LARGURAS = [(1500, 1000), (1180, 900), (390, 844)]


def abrir(browser, base, caminho, largura, altura):
    page = browser.new_page(viewport={"width": largura, "height": altura})
    problemas = []
    page.on("pageerror", lambda erro: problemas.append(str(erro)))
    page.goto(f"{base}{caminho}", wait_until="domcontentloaded")
    page.wait_for_selector("#state div, #rows tr")
    page.wait_for_timeout(700)
    return page, problemas


@pytest.mark.parametrize(("largura", "altura"), LARGURAS)
def test_the_panel_has_no_layout_or_contrast_defects(browser, server, largura, altura):
    page, problemas = abrir(browser, server, "/", largura, altura)
    achados = auditar(page)
    page.close()
    assert problemas == []
    assert achados == [], f"\n{descrever(achados)}"


@pytest.mark.parametrize(("largura", "altura"), LARGURAS)
def test_the_requests_screen_has_no_layout_or_contrast_defects(browser, server, largura, altura):
    page, problemas = abrir(browser, server, "/requests", largura, altura)
    if largura > 720:
        page.click("#rows tr:nth-child(2)")
        page.wait_for_timeout(500)
    achados = auditar(page)
    page.close()
    assert problemas == []
    assert achados == [], f"\n{descrever(achados)}"
