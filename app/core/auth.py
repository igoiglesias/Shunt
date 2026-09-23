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
"""

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


def login_redirect(request: Request) -> RedirectResponse:
    """Handler do `LoginRequired`: manda o navegador para a tela de login."""
    return RedirectResponse(LOGIN_URL, status_code=303)
