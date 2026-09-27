"""As duas telas passam pela varredura, em todas as larguras.

Este arquivo existe porque olhar o print nao bastou: numa rodada em que eu disse
que estava bom, a varredura encontrou 105 problemas -- contraste abaixo do
minimo em toda tabela, texto sobreposto no diagrama, alvo de clique de 21 px.
"""

import pytest

from app.config.config import ADMIN_COOKIE
from app.core.security import issue_jwt

# As duas fixtures vem do modulo vizinho: o mesmo servidor semeado serve as duas
# varreduras, e subir um segundo dobraria o tempo da suite.
from tests.browser.test_audit_browser import SESSION_SECRET, browser, server  # noqa: F401
from tests.browser.ui_audit import auditar, descrever

LARGURAS = [(1500, 1000), (1180, 900), (390, 844)]


def abrir(browser, base, caminho, largura, altura):  # noqa: F811
    page = browser.new_page(viewport={"width": largura, "height": altura})
    # As telas exigem sessao admin: o cookie vem assinado com o segredo do servidor.
    page.context.add_cookies(
        [{"name": ADMIN_COOKIE, "value": issue_jwt(1, SESSION_SECRET, 3600), "url": base}]
    )
    problemas = []
    page.on("pageerror", lambda erro: problemas.append(str(erro)))
    page.goto(f"{base}{caminho}", wait_until="domcontentloaded")
    page.wait_for_selector("#state div, #rows tr")
    page.wait_for_timeout(700)
    return page, problemas


@pytest.mark.parametrize(("largura", "altura"), LARGURAS)
def test_the_panel_has_no_layout_or_contrast_defects(browser, server, largura, altura):  # noqa: F811
    page, problemas = abrir(browser, server, "/admin/painel", largura, altura)
    achados = auditar(page)
    page.close()
    assert problemas == []
    assert achados == [], f"\n{descrever(achados)}"


@pytest.mark.parametrize(("largura", "altura"), LARGURAS)
def test_the_requests_screen_has_no_layout_or_contrast_defects(
    browser,  # noqa: F811
    server,  # noqa: F811
    largura,
    altura,
):
    page, problemas = abrir(browser, server, "/admin/requests", largura, altura)
    if largura > 720:
        page.click("#rows tr:nth-child(2)")
        page.wait_for_timeout(500)
    achados = auditar(page)
    page.close()
    assert problemas == []
    assert achados == [], f"\n{descrever(achados)}"


@pytest.mark.parametrize(("largura", "altura"), LARGURAS)
def test_the_analysis_panel_has_no_layout_or_contrast_defects(
    browser,  # noqa: F811
    server,  # noqa: F811
    largura,
    altura,
):
    """O painel da analise tambem passa pela regua, e nao pelo meu olho.

    Ele nasce com texto longo do modelo e um bloco de JSON: as duas coisas que
    mais escapam da caixa -- e as duas que um print faz parecer certas.
    """
    from tests.browser.test_audit_browser import stub_analysis

    page, problemas = abrir(browser, server, "/admin/requests", largura, altura)
    stub_analysis(page)
    page.click("#analyse")
    page.wait_for_selector("#analysis-text")
    page.wait_for_timeout(300)
    achados = auditar(page)
    page.click('.tabs button[data-tab="dossier"]')
    page.wait_for_selector("pre.raw")
    page.wait_for_timeout(300)
    achados += auditar(page)
    page.close()
    assert problemas == []
    assert achados == [], f"\n{descrever(achados)}"
