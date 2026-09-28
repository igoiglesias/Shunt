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
# D4: `.chip.bad` (sem filtro de estado). O chip falso vem SEM aria-pressed
# (D1), entao `.chip.bad[aria-pressed="true"]` nunca casa: a regra era morta e
# o vermelho do "sem tools" nunca pintava.
# ---------------------------------------------------------------------------


def test_chip_bad_has_no_dead_state_filter():
    source = CSS.read_text(encoding="utf-8")
    assert ".chip.bad {" in source
    # O seletor antigo colava um estado que o chip nao tem mais (e nunca
    # tera: ele e rotulo, nao controle).
    assert '.chip.bad[aria-pressed="true"]' not in source
    # E a regra que existe pinta de fato o vermelho.
    block = source.split(".chip.bad {", 1)[1].split("}", 1)[0]
    assert "var(--error)" in block


def test_shunt_css_is_served_with_no_store():
    """O CSS sobe com no-store: ele muda junto com as telas, e uma copia
    velha em cache deixaria a tela nova com a cara antiga."""
    app.state.admin_session_secret = ADMIN_SESSION_SECRET
    with TestClient(app) as c:
        r = c.get("/shunt.css")
    assert r.status_code == 200
    assert "cache-control" in r.headers
    assert "no-store" in r.headers["cache-control"]
