"""A tela de auditoria, sem navegador: o que o `<script>` dela desenha.

A lista, o detalhe e a sincronizacao do filtro nao sao HTML server-rendered, e
sim JavaScript dentro do proprio template (`rowOf`, `detailOf`, `SELECTS`). A
bateria de navegador (`tests/browser/test_audit_browser.py`) cobre o caminho
inteiro com Chromium headless; estes testes cobem o que um RED de task precisa
dizer -- "esta linha mostra 'repasse' e nao '0'" -- sem subir um servidor.

O motor e o Node (`tests/stats/audit_page.js`), que extrai o `<script>` do
template e o roda num sandbox com o minimo de DOM. Sem Node na maquina o modulo
inteiro e pulado, e nao falha: o CI de `unit` nao instala Node, e a bateria de
navegador e que guarda a afirmacao completa la.
"""

import json
import shutil
import subprocess

import pytest

from app.core.security import issue_jwt
from app.core.upstream import UpstreamPool
from app.main import app
from app.stats.recorder import Recorder
from tests.core.test_dispatcher import SETTINGS

ADMIN_SESSION_SECRET = "segredo-de-teste-do-admin"

HARNESS = "tests/stats/audit_page.js"


def _node() -> str | None:
    """O Node, ou None se nao estiver instalado."""
    return shutil.which("node")


node = pytest.mark.skipif(_node() is None, reason="sem node para rodar o JS da tela")


def _run(scenarios: list[str]) -> dict:
    """Roda os cenarios no harness e devolve {cenario: resultado}.

    Cada cenario roda num contexto novo do ponto de vista do harness, e a
    afirmacao de cada teste e sobre o resultado do seu.
    """
    saida = subprocess.run(
        ["node", HARNESS, *scenarios],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(json.loads(saida.stdout))


def _reset_app_state():
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    app.state.recorder = Recorder(None)
    app.state.admin_session_secret = ADMIN_SESSION_SECRET


def _cookie(user_id: int = 1) -> dict:
    return {"shunt_admin": issue_jwt(user_id=user_id, secret=ADMIN_SESSION_SECRET, ttl_seconds=3600)}


@node
def test_o_select_de_tipo_existe_com_tres_opcoes_estaticas():
    """RED 1: `kind` entra em SELECTS e o `<select>` tem 3 options fixas.

    O `<select id="kind">` e estatico de proposito: `/api/requests/facets` nao
    devolve tipos, e a iteracao de `loadFacets` so chama `fill` para as 4
    facets reais. Se ela alcancasse `kind`, apagaria as 3 options.
    """
    resultado = _run(["static_kind_select"])["static_kind_select"]
    assert resultado["selects_has_kind"] is True, (
        "SELECTS precisa de `kind` para a sincronizacao da URL"
    )
    assert resultado["label"] == "tipo"
    assert resultado["kind_option_values"] == ["", "model", "relay"], (
        "value '' = todas, 'model' = so modelo, 'relay' = so relay"
    )
    assert resultado["kind_option_texts"] == ["todas", "so modelo", "so relay"]
    # O risco apontado no brief: depois de `loadFacets`, as options continuam.
    assert resultado["options_after_loadfacets"] == ["", "model", "relay"]


def test_a_pagina_de_auditoria_continha_o_select_estatico_de_tipo():
    """RED 1 (HTML): o select esta no HTML servido por `/admin/requests`."""
    from fastapi.testclient import TestClient

    _reset_app_state()
    with TestClient(app) as c:
        r = c.get("/admin/requests", cookies=_cookie())
    assert r.status_code == 200
    assert 'id="kind"' in r.text, "o `<select id=\"kind\">` tem de estar no HTML"
    assert 'value=""' in r.text
    for texto in ("todas", "so modelo", "so relay"):
        assert f">{texto}</option>" in r.text, f"option {texto!r} ausente"


@node
def test_a_linha_de_relay_mostra_repasse_e_traco_nunca_zero():
    """RED 2: relay e passagem de rede; Tokens nao e "0", e "—".

    Em JS `null + null === 0`, e `compact(0)` devolve "0": sem o ramo por
    `kind` a linha de relay afirmaria ter medido zero tokens.
    """
    resultado = _run(["relay_row"])["relay_row"]
    html = resultado["rows_html"]

    assert resultado["served_label"] == "repasse", (
        "`servedLabel` devolve 'repasse' para a linha de relay"
    )
    # A linha de modelo nao e atingida pelo ramo novo.
    assert resultado["served_label_model"] == "nenhum candidato respondeu"

    assert 'class="tag relay"' in html, "a tag de repasse tem a classe propria"
    assert "repasse · anthropic" in html, "a tag mostra 'repasse' e o dialect"
    assert ">0<" not in html, "a linha de relay nao mostra zero em Tokens"
    # Pedido e Sinais ficam com o traco, e Tokens tambem.
    assert '<td data-label="Pedido">—</td>' in html or ">—</td>" in html
    assert '<span class="none">—</span>' in html, "Sinais sem marks fica com traco"


@node
def test_o_filtro_de_tipo_va_para_a_url_e_para_a_busca():
    """RED 3: escolher o tipo poe `kind=relay` na URL e no pedido da API."""
    resultado = _run(["filter_url"])["filter_url"]
    assert any(
        "kind=relay" in url for _, url in resultado["history"]
    ), "o filtro de tipo tem de chegar a URL"
    assert resultado["state"].count('"kind":"relay"') == 1
    assert any(
        "kind=relay" in u for u in resultado["request_query"]
    ), "o filtro tem de chegar ao pedido de /api/requests"


@node
def test_um_link_com_tipo_preenche_o_select():
    """RED 3 (volta): `?kind=relay` deixa o select no valor certo."""
    resultado = _run(["filter_from_url"])["filter_from_url"]
    assert resultado["kind_value"] == "relay"
    assert '"kind":"relay"' in resultado["state"]


@node
def test_o_detalhe_da_linha_de_relay_esconde_as_secoes_de_modelo():
    """RED 4: no detalhe de relay nao aparecem as secoes que so modelo tem.

    "Regra de rota", "Tokens", "Cache", "Raciocinio", "Cadeia" e "Ferramentas"
    sao sobre a cadeia de provedores -- relay e passagem de rede bruta.
    """
    resultado = _run(["relay_detail"])["relay_detail"]
    html = resultado["detail_html"]

    for ausente in (
        "Regra de rota",
        "<dt>Tokens</dt>",
        "<dt>Cache</dt>",
        "Raciocínio",
        "<h3>Cadeia</h3>",
        "<h3>Ferramentas</h3>",
        "0 entrada",
        "0 saída",
    ):
        assert ausente not in html, f"o detalhe de relay nao tem '{ausente}'"
    # O cabecalco continua dizendo o que aquela linha e.
    assert "repasse" in html
    # Sem a frase, um detalhe sem tokens nem cadeia parece tela quebrada.
    assert resultado["explica"], "o detalhe de relay explica por que nao ha modelo"
    assert "Shunt" in resultado["explica"]


@node
def test_a_linha_de_modelo_continua_como_antes():
    """Regressao: a linha e o detalhe de modelo nao mudaram com o ramo de relay."""
    resultado = _run(["model_row_and_detail"])["model_row_and_detail"]
    assert "15 total" in resultado["detail_html"], "Tokens da linha de modelo somado"
    assert "family → haiku" in resultado["detail_html"]
    assert "respondeu em 100ms" in resultado["detail_html"], "a cadeia de modelo segue no detalhe"


@node
def test_a_linha_de_modelo_sem_tokens_nao_finge_que_mediu_zero():
    """M1: a linha muda de MODELO mostra "—", nunca "0".

    Diferente do relay, esta linha nao tem `kind` para guarda-la: e um modelo
    cujo provedor nao informou usage. `null + null === 0` em JS faria a tela
    afirmar uma medicao de zero tokens que nunca existiu. A parcial (so a
    saida) entra com o lado informado.
    """
    resultado = _run(["muted_model_row"])["muted_model_row"]

    # Linha em silencio total: o traco, e nunca o zero.
    assert 'data-label="Tokens">—<' in resultado["calada"], (
        "a linha sem tokens informados mostra traco, nao zero"
    )
    # Linha parcial: entra com o lado que veio.
    assert 'data-label="Tokens">1.2k<' in resultado["parcial"], (
        "a linha parcial mostra o lado informado"
    )
    # O detalhe tambem distingue: entrada em silencio, saida informada.
    assert "— entrada · 1.2k saída" in resultado["detalho_parcial"], (
        "o detalhe mostra traco no lado que o provedor nao informou"
    )


@node
def test_a_aba_de_erro_so_existe_quando_a_requisicao_falhou():
    """Um 422 na tabela so diz o numero; a aba "Erro" e onde o corpo do
    provedor aparece, e ela so existe quando `status >= 400`.

    O badge de `error_type` na linha carrega o motivo do ultimo candidato no
    `title`, para o hover explicar sem precisar abrir o detalhe.
    """
    resultado = _run(["error_tab_and_tooltip"])["error_tab_and_tooltip"]
    assert resultado["has_error_tab"] is True, "falha tem a aba Erro"
    assert resultado["model_has_no_error_tab"] is False, "sucesso nao tem a aba Erro"
    # O badge da linha traz o motivo no title do hover.
    assert 'title="422 prompt too long (attempt 1)"' in resultado["row_title"], (
        "o badge de erro carrega o motivo do ultimo candidato no title"
    )
    # A linha de sucesso nao ganha o title do motivo.
    assert 'class="tag bad"' not in _run(["model_row_and_detail"])["model_row_and_detail"]["rows_html"]
