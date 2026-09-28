"""A folha de estilo compartilhada: tokens que carregam contraste e regras
que tem de casar com o que a tela realmente renderiza.

O CSS e servidor-renderizado no sentido inverso: ele e um arquivo estatico
servido por `/shunt.css`, mas os SEUS valores sao a unica fonte de verdade
das cores que o usuario ve. Contraste e matematica, nao impressao -- os
valores aqui sao lidos do arquivo e a razao e calculada com a mesma formula
WCAG que o auditor do navegador usa (`tests/browser/ui_audit.py`).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import ESTILO, app
from tests.routers.test_admin_config import ADMIN_SESSION_SECRET

CSS = Path(ESTILO)


def _tokens() -> dict[str, str]:
    """Le os `--token: valor;` do arquivo (a unica fonte de verdade)."""
    tokens: dict[str, str] = {}
    for line in CSS.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("--") and ":" in stripped:
            name, _, value = stripped.partition(":")
            tokens[name.strip()] = value.strip().rstrip(";")
    return tokens


def _hex(color: str) -> tuple[float, float, float]:
    text = color.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"nao sei medir {color!r}; so rgb hex de 6 digitos")
    return tuple(int(text[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _luminance(color: str) -> float:
    """Luminancia relativa WCAG (a mesma formula do auditor headless)."""
    red, green, blue = _hex(color)
    channel = lambda channel: channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4
    return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)


def _blend(foreground: str, background: str, alpha: float) -> str:
    """Compoe alpha de foreground sobre background e devolve o hex resultante
    (o que o navegador faz com rgba(...,alpha))."""
    red_a, green_a, blue_a = _hex(foreground)
    red_b, green_b, blue_b = _hex(background)
    red = round((red_a * alpha + red_b * (1 - alpha)) * 255)
    green = round((green_a * alpha + green_b * (1 - alpha)) * 255)
    blue = round((blue_a * alpha + blue_b * (1 - alpha)) * 255)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _contrast(foreground: str, background: str) -> float:
    light = _luminance(foreground)
    dark = _luminance(background)
    return (max(light, dark) + 0.05) / (min(light, dark) + 0.05)


# ---------------------------------------------------------------------------
# D2: --error e usado no botao de excluir (texto BRANCO sobre ele) e em varios
# sinais. O branco sobre o vermelho precisa de 4.5:1 (AA) -- abaixo disso o
# axe-core relata color-contrast.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("foreground", "background_token"),
    [
        ("#ffffff", "--error"),  # .btn-danger de admin_config e _admin_styles
        ("var(--ground)", "--ink"),  # .btn-primary: texto escuro sobre claro
    ],
)
def test_a_button_label_reaches_aa_contrast(foreground, background_token):
    """Os dois botoes fixos da area admin: o de excluir (branco sobre
    --error) e o primario (escuro sobre --ink)."""
    tokens = _tokens()
    fg = tokens.get(foreground.removeprefix("var(").removesuffix(")"), foreground)
    ratio = _contrast(fg, tokens[background_token])
    assert ratio >= 4.5, (
        f"{fg} sobre {background_token}={tokens[background_token]} "
        f"da {ratio:.2f}:1; o piso AA para texto e 4.5:1"
    )


def test_error_token_is_the_contrast_floor_for_buttons():
    """O botao de excluir e a unica superficie de --error com texto branco:
    medir o token e medir o botao."""
    tokens = _tokens()
    assert tokens["--error"] == "#c13a2e"
    ratio = _contrast("#ffffff", tokens["--error"])
    assert ratio >= 4.5
    # O piso do vermelho anterior (#d5463a) era 4.41:1 -- abaixo de AA.
    assert _contrast("#ffffff", "#d5463a") < 4.5


def test_the_cap_toggle_thumb_stays_contrasting_on_good():
    """O thumb do interruptor de capacidade e o UNICO sinal do estado ligado,
    entao a lampada tem que se destacar do trilho --good.

    Cuidado com a armadilha de medir contra o fundo errado: --ink #f2efe6
    sobre --ground #141311 da ~16:1, mas SOBRE --good da 2.79:1. O trilho e
    a superficie do thumb, e --ground (5.80:1) e o que cumpre AA.
    """
    tokens = _tokens()
    good = tokens["--good"]
    assert _contrast(tokens["--ground"], good) >= 4.5
    assert _contrast(tokens["--ink"], good) < 4.5


# ---------------------------------------------------------------------------
# D4: `.chip.bad`. Existem dois .chip.bad na area admin:
#   - os rotulos de capacidade das tabelas (<span class="chip bad">, SEM
#     aria-pressed);
#   - o botao de filtro "falhas" da auditoria (audit.html:326, COM
#     aria-pressed, alternado pelo JS de audit.html:1148).
# A regra de estado so e morta se NENHUM .chip.bad tiver aria-pressed. Como
# o da auditoria tem, ela serve um elemento real.
# ---------------------------------------------------------------------------


def test_chip_bad_rule_paints_the_error_red():
    source = CSS.read_text(encoding="utf-8")
    assert ".chip.bad {" in source
    # A regra de base pinta de fato o vermelho.
    block = source.split(".chip.bad {", 1)[1].split("}", 1)[0]
    assert "var(--error)" in block


def test_chip_bad_pressed_rule_is_not_dead():
    """O filtro 'falhas' da auditoria e .chip.bad E tem aria-pressed: a
    regra de estado casa um elemento que existe de verdade."""
    audit = (CSS.parent / "audit.html").read_text(encoding="utf-8")
    chips_bad = [
        line for line in audit.splitlines()
        if "chip bad" in line and "aria-pressed" in line
    ]
    assert chips_bad, "o filtro .chip.bad com aria-pressed sumou da auditoria"
    source = CSS.read_text(encoding="utf-8")
    assert '.chip.bad[aria-pressed="true"] {' in source


def test_shunt_css_is_served_with_no_store():
    """O CSS sobe com no-store: ele muda junto com as telas, e uma copia
    velha em cache deixaria a tela nova com a cara antiga."""
    app.state.admin_session_secret = ADMIN_SESSION_SECRET
    with TestClient(app) as c:
        r = c.get("/shunt.css")
    assert r.status_code == 200
    assert "cache-control" in r.headers
    assert "no-store" in r.headers["cache-control"]


# ---------------------------------------------------------------------------
# R1 (revisao D-round): o seletor sticky de acoes estava morto. As quatro
# tabelas da config admin nao tem td.actions/th.actions -- a classe .actions
# esta no DIV de dentro (<td class="n"><div class="actions">). O seletor
# casava nada e a coluna de acoes continuava alem da dobra no mobile.
# ---------------------------------------------------------------------------


CONFIG = CSS.parent / "admin_config.html"


def test_the_sticky_actions_rule_matches_a_real_cell():
    """A regra sticky precisa casar uma celula real. Cada partial poe as
    acoes na ultima coluna, e o seletor tem que alcancar o <td> externo --
    nao o div de dentro."""
    source = CONFIG.read_text(encoding="utf-8")
    media = source.split("@media (max-width: 760px)", 1)[1]
    # O seletor vivo: alcanca a ultima coluna das quatro tabelas.
    assert ".table-container td:last-child" in media
    assert ".table-container th:last-child" in media
    # O seletor morto nao pode sobrar como regra (casava so a classe que
    # esta no div interno). Assercao no seletor, nao em substring solto:
    # o comentario que documenta o bug cita as palavras.
    assert "td.actions," not in media
    assert "td.actions {" not in media
    assert "th.actions," not in media
    assert "th.actions {" not in media


@pytest.mark.parametrize(
    "template",
    ["_models.html", "_providers.html", "_routes.html", "_history.html"],
)
def test_every_config_table_has_actions_as_its_last_cell(template):
    """A regra e :last-child, então ela so vale se a coluna de acoes for
    realmente a ultima de cada tabela. Quebrar a ordem quebra o sticky."""
    path = CSS.parent / template
    for line in path.read_text(encoding="utf-8").splitlines():
        if "<th" in line and "Ações" in line:
            assert line.rstrip().endswith("</th>")
            assert "<td" not in line
            return
    raise AssertionError(f"cabecalho de acoes nao encontrado em {template}")


# ---------------------------------------------------------------------------
# R2: --error escureceu para #c13a2e, e tres textos ainda usavam o token
# direto. O novo vermelho como TEXTO sobre --ground ou sobre a tint falha
# AA (3.46:1 e 3.23:1). --error-text #e2695c existe para isso.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("template", "selector"),
    [
        ("login.html", ".error"),
        ("dashboard.html", ".tape li.bad .what b"),
    ],
)
def test_error_text_uses_the_readable_variant(template, selector):
    """Texto vermelho sobre fundo escuro: tem que ser --error-text, nao
    --error. AA para texto pequeno e 4.5:1."""
    path = CSS.parent / template
    source = path.read_text(encoding="utf-8")
    rule = source.split(selector + " {", 1)[1].split("}", 1)[0]
    assert "var(--error-text)" in rule, (
        f"{template} {selector} nao usa --error-text; contraste insuficiente"
    )
    assert "var(--error)" not in rule


def test_error_text_token_reaches_aa_on_ground():
    """O par que as duas regras acima montam: --error-text sobre --ground."""
    tokens = _tokens()
    ratio = _contrast(tokens["--error-text"], tokens["--ground"])
    assert ratio >= 4.5, f"--error-text sobre --ground da {ratio:.2f}:1"


def test_error_text_token_reaches_aa_on_the_bad_tint():
    """O .tape li.bad tem tint de --error a 10% por baixo do <b>. O piso
    do vermelho direto la era 3.23:1; o --error-text chega a 5.29:1."""
    tokens = _tokens()
    tint = _blend(tokens["--error"], tokens["--panel"], 0.10)
    assert _contrast(tokens["--error-text"], tint) >= 4.5
    # O vermelho direto nao passa no mesmo fundo -- e o que esta regra prova.
    assert _contrast(tokens["--error"], tint) < 4.5


# ---------------------------------------------------------------------------
# R3: o chip .bad pressionado. O estado ligado e generico (texto --ground
# sobre fundo --ink) e pinta por cima do branco do .chip.bad: no novo
# --error, texto escuro sobre vermelho da 3.46:1. O .slow e .chain ja tem
# a sua variante de estado; o .bad e o que faltava.
# ---------------------------------------------------------------------------


def test_chip_bad_pressed_keeps_a_light_text():
    source = CSS.read_text(encoding="utf-8")
    assert '.chip.bad[aria-pressed="true"] {' in source
    block = source.split('.chip.bad[aria-pressed="true"] {', 1)[1].split("}", 1)[0]
    assert "var(--ink)" in block or "#fff" in block
    assert "var(--error)" in block  # mantem o fundo vermelho


def test_chip_bad_pressed_text_reaches_aa():
    tokens = _tokens()
    fg = tokens["--ink"]
    bg = tokens["--error"]
    ratio = _contrast(fg, bg)
    assert ratio >= 4.5, f"--ink sobre --error da {ratio:.2f}:1"
    # O que a regra generica arrastava para o chip: escuro sobre vermelho.
    assert _contrast(tokens["--ground"], bg) < 4.5
