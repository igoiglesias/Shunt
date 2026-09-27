"""Testes de `safe_next`: o destino pos-login so aceita caminho sob `/admin`.

O valor vem da query/form, entao e entrada do atacante: um `next` aceito sem
filtro vira open redirect logo apos o login. Cada vetor abaixo ja derrubou
algum filtro de redirect por ai; a lista cobre um vetor por guarda de
`safe_next`, para que remover qualquer guarda deixe um teste vermelho -- com
uma excecao: a guarda de `scheme` e defesa em profundidade (ver docstring de
`safe_next`) e nao tem vetor que a alcance sozinha, porque todo `scheme:...`
real ja e barrado antes pela guarda "sem `/` no inicio".
"""

import pytest

from app.core.auth import safe_next

FALLBACK = "/admin/painel"


@pytest.mark.parametrize(
    "value",
    [
        "/admin/requests?q=1",
        "/admin/painel",
        "/admin",
        "/admin/",
        "/admin/requests?q=1#frag",
        "/admin/users",
    ],
)
def test_safe_next_accepts_same_origin_admin_paths(value):
    """Caminho relativo sob `/admin` volta intacto, com query e fragmento."""
    assert safe_next(value) == value


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        # host de outro dominio, direto ou disfarcado
        "//evil.com",
        "///admin/x",
        "https://evil.com",
        "https://evil.com/admin/x",
        "//evil.com/admin/x",
        "http:evil.com",
        "http:/admin/x",
        "javascript:alert(1)",
        "javascript:/admin/x",
        # barra invertida: o navegador a le como `/`
        "/\\evil.com",
        "\\\\evil.com",
        "/admin/\\evil.com",
        # codificado: decodifica para `//` ou `/\`
        "/%2F%2Fevil.com",
        "%2F%2Fevil.com",
        "/%5Cevil.com",
        # codificado sem barra inicial: decodifica para /admin, mas o valor cru
        # e relativo e o navegador o resolve contra /admin/login
        "%2Fadmin/x",
        "%2fadmin",
        # fora de /admin
        "/api/stats",
        "/docs",
        "/",
        "admin/x",
        "/administrador",
        "/adminx/y",
        # travessia de diretorio para fora de /admin
        "/admin/../docs",
        "/admin/./x",
        "/admin/%2e%2e/docs",
        # laco no proprio login
        "/admin/login",
        "/admin/login?next=/admin",
        "/admin/login/",
        "/admin/%6Cogin",
        # espaco e caracteres de controle
        "/admin\n",
        " /admin",
        "/admin/x\n",
        "/admin/a b",
        "/admin/x\x00",
        "/admin/x\x7f",
    ],
)
def test_safe_next_rejects_to_fallback(value):
    """Qualquer vetor fora da regra cai em `/admin/painel`."""
    assert safe_next(value) == FALLBACK
