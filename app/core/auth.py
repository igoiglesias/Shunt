"""Sessao do admin: o JWT mora num cookie e e validada em toda requisicao.

O cookie carrega um JWT assinado (ver `app.core.security`). Cada rota protegida
passa por `require_admin`, que decodifica o token, confere a validade e devolve
o id do usuario logado. Falhou -- cookie ausente, expirado ou forjado -- e a
resposta depende do tipo de cliente: pagina HTML vai para o login (o navegador
precisa de um destino), chamada de API/HTMX ganha 401 (sem HTML para renderizar).

O redirect nao e devolvido pela dependencia em si, porque o FastAPI injetaria o
`RedirectResponse` no parametro da rota em vez de enviar como resposta, e a rota
entao responderia o proprio HTML. Em vez disso a dependencia levanta
`LoginRequired`; o handler registrado em `app/main.py` o converte no redirect.

O redirect leva o caminho pedido em `?next=`, e o login volta para ele. Esse
valor vem do navegador, entao e entrada hostil: `safe_next` so deixa passar
caminho relativo de mesma origem sob `/admin` (exceto o proprio login, que
seria um laco) e troca todo o resto pelo painel. Sem esse filtro o login vira
open redirect -- o atacante manda um link de login legitimo que, depois da
senha digitada, leva a vitima para o dominio dele.
"""

from urllib.parse import quote, unquote, urlsplit

from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse

# Nome e alcance do cookie de sessao (definicoes em `app/config/config.py`).
# Por que o `path` precisa ser `"/"` e nao `"/admin"` -- as paginas do admin
# (em `/admin/...`) fazem fetch de dados em `/api/...`, entao o cookie tem de
# alcancar os dois -- esta na definicao do valor, em `config.py`.
from app.config.config import (
    ADMIN_COOKIE,
    LOGIN_URL,
)
from app.core.security import decode_jwt


class LoginRequired(Exception):
    """Sinal interno: sessao ausente/invalida em uma navegacao HTML.

    Nao e um `HTTPException` de proposito: o handler que o converte em
    redirect precisa distingui-lo do 401 de API, e uma excecao propria deixa a
    intencao legivel no ponto onde e levantada.
    """


def is_api_request(request: Request) -> bool:
    """True quando o cliente e o HTMX ou pede JSON, e nao uma navegacao de pagina.

    O HTMX manda `HX-Request: true` em toda chamada; o `Accept` com
    `application/json` cobre chamadas de programa. Nos dois casos um redirect
    HTML nao faz sentido -- devolve-se 401 para o cliente tratar.

    Rotas sob `/api/` sao API por definicao: nem o navegador nem o HTMX as
    chamam sem um header que ja identifique-as, mas o contrato delas e
    `401 quando nao autenticado`, nunca um redirect 303 para HTML de login.
    """
    if request.url.path.startswith("/api/"):
        return True
    if request.headers.get("hx-request", "").lower() == "true":
        return True
    accept = request.headers.get("accept", "")
    return "application/json" in accept.lower()


def require_admin(request: Request) -> int:
    """Dependencia de sessao: valida o JWT do cookie e devolve o `user_id`.

    Le o segredo de `app.state` (injetado no boot, ver o lifespan de
    `app/main.py`). Token valido devolve `sub` (o id do usuario). Sessao ruim
    em navegacao HTML levanta `LoginRequired`; em chamada de API levanta 401.
    """
    secret = getattr(request.app.state, "admin_session_secret", None)
    token = request.cookies.get(ADMIN_COOKIE)
    claims = decode_jwt(token, secret) if (secret and token) else None
    if claims is None:
        if is_api_request(request):
            raise HTTPException(status_code=401, detail="Sessao invalida")
        raise LoginRequired()
    return int(claims["sub"])


# Destino pos-login quando o `next` falta ou e recusado.
NEXT_FALLBACK = "/admin/painel"


def safe_next(raw: str | None) -> str:
    """Devolve `raw` se for caminho de mesma origem sob `/admin`, senao o painel.

    Cada guarda fecha um vetor que as outras nao fecham, exceto a de `scheme`
    (ver nota abaixo):
    - barra invertida, espaco e controle: o navegador le `\\` como `/` e
      descarta espaco/controle nas pontas, o que transforma `/\\evil.com` ou
      ` //evil.com` em outro host;
    - sem `/` no inicio: o valor cru seria relativo a `/admin/login`, e as
      guardas seguintes olham o path DECODIFICADO -- `%2Fadmin/x` decodifica
      para `/admin/x` e passaria, devolvendo o cru;
    - `//` no inicio: `urlsplit("///admin/x")` da netloc vazia e path
      `/admin/x`, mas o navegador le `///admin` como host `admin`;
    - `scheme`: defesa em profundidade, sem teste que a alcance sozinha.
      `urlsplit` so preenche `scheme` quando o texto ANTES do `:` bate com
      `[a-zA-Z][a-zA-Z0-9+.-]*` -- e essa mesma checagem so roda depois da
      guarda "sem `/` no inicio" (linha acima), que ja barra todo `scheme:...`
      real (`https://evil.com/admin/x`, `javascript:/admin/x`), porque nenhum
      deles comeca com `/`. Fica como cinto e suspensorio caso a guarda
      anterior mude;
    - path decodificado sob `/admin`: e o que o servidor roteia; `%2F`/`%5C`
      codificados (`/%2F%2Fevil.com`) nao comecam com `/admin/` e caem aqui;
    - segmento `.`/`..`: o navegador resolve `/admin/../docs` (e a forma
      `%2e%2e`) para fora de `/admin`;
    - o proprio login: voltar para ele depois de logar seria um laco.
    """
    if not raw:
        return NEXT_FALLBACK
    if any(ch == "\\" or ch.isspace() or not ch.isprintable() for ch in raw):
        return NEXT_FALLBACK
    if not raw.startswith("/"):
        return NEXT_FALLBACK
    if raw.startswith("//"):
        return NEXT_FALLBACK
    parts = urlsplit(raw)
    if parts.scheme:
        return NEXT_FALLBACK
    path = unquote(parts.path)
    if path != "/admin" and not path.startswith("/admin/"):
        return NEXT_FALLBACK
    segments = path.split("/")
    if "." in segments or ".." in segments:
        return NEXT_FALLBACK
    if path.rstrip("/") == LOGIN_URL:
        return NEXT_FALLBACK
    return raw


def login_redirect(request: Request) -> RedirectResponse:
    """Handler do `LoginRequired`: manda o navegador para a tela de login.

    Leva o caminho pedido (com a query, quando houver) em `?next=`, codificado
    com `quote(..., safe="/")` para que `?`, `=` e `&` do destino nao se
    misturem a query do proprio login. O filtro fica no login (`safe_next`),
    que e onde o valor e consumido.
    """
    target = request.url.path
    if request.url.query:
        target = f"{target}?{request.url.query}"
    return RedirectResponse(f"{LOGIN_URL}?next={quote(target, safe='/')}", status_code=303)
