"""Painel e Requisições dentro da area admin, atrás da sessao.

Até aqui as duas paginas viviam em `app/main.py` como arquivos lidos do disco
(`GET /` e `GET /requests`), sem login. Aqui elas ganham endereço sob `/admin`,
passam por `require_admin` e trocam o nav proprio pelo menu compartilhado do
painel (Painel, Requisições, Configuração, Usuários, Tokens). O conteúdo das
paginas continua sendo os mesmos templates, servidos via `render()` -- que,
além de permitir o include do menu, devolve um 500 limpo em vez de derrubar se
uma linha de template quebrar.
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from app.core.auth import require_admin
from app.templates import render

router = APIRouter(prefix="/admin", tags=["admin-dashboard"])

# As duas paginas mudam junto com o estado do proxy; um navegador que as
# cachearia entregaria um painel de dez minutos atras. `no-store` as torna
# sempre do disco, como quando viviam em `app/main.py`.
NO_STORE = {"cache-control": "no-store, must-revalidate"}


@router.get("/painel", response_class=HTMLResponse)
async def painel(request: Request, _: int = Depends(require_admin)):
    """O painel de uso, agora sob /admin e atrás do login."""
    response = render("dashboard.html", request=request, active="painel")
    response.headers.update(NO_STORE)
    return response


@router.get("/requests", response_class=HTMLResponse)
async def requests_page(request: Request, _: int = Depends(require_admin)):
    """A tela de auditoria: buscar, abrir e exportar requisicoes gravadas."""
    response = render("audit.html", request=request, active="requests")
    response.headers.update(NO_STORE)
    return response
