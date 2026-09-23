"""Tokens de acesso a API v1, emitidos pelo painel.

O valor em claro do token e gerado com `secrets.token_urlsafe` e exibido UMA
vez -- na resposta da criacao. So o SHA-256 vai para o banco, entao nao ha como
recuperá-lo depois: quem nao copiou na hora tem que emitir outro. O painel mostra
apenas um prefixo curto do hash para identificar o token, nunca o token em si.
"""

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.auth import require_admin
from app.routers.admin_config import request_engine
from app.stats.models import ApiToken
from app.templates import render

router = APIRouter(
    prefix="/admin/tokens", tags=["admin-tokens"], dependencies=[Depends(require_admin)]
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@router.get("", response_class=HTMLResponse, include_in_schema=False)
async def tokens_page(request: Request, new_token: str | None = None):
    engine = request_engine(request)
    with Session(engine) as session:
        tokens = session.execute(
            select(ApiToken).order_by(ApiToken.created_at.desc())
        ).scalars().all()
    return render(
        "tokens.html",
        request=request,
        tokens=tokens,
        new_token=new_token,
        active="tokens",
    )


@router.get("/list", response_class=HTMLResponse, include_in_schema=False)
async def list_tokens(request: Request):
    """A tabela de tokens; alvo de troca das operacoes via HTMX."""
    engine = request_engine(request)
    with Session(engine) as session:
        tokens = session.execute(
            select(ApiToken).order_by(ApiToken.created_at.desc())
        ).scalars().all()
    return render("_token_list.html", request=request, tokens=tokens, active="tokens")


@router.get("/new", response_class=HTMLResponse, include_in_schema=False)
async def new_token_form(request: Request, _: int = Depends(require_admin)):
    return render("_token_form.html", request=request, active="tokens")


@router.post("", response_class=HTMLResponse, include_in_schema=False)
async def create_token(
    request: Request,
    name: Annotated[str, Form()],
    expires_days: Annotated[str | None, Form()] = None,
    _: int = Depends(require_admin),
):
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Informe um nome para o token")
    # Expiracao opcional em dias; valor invalido vira token sem expiracao.
    expires_at = None
    if expires_days not in (None, ""):
        try:
            days = float(expires_days)
        except ValueError:
            days = None
        if days is not None and days > 0:
            expires_at = datetime.now(UTC) + timedelta(days=days)

    plaintext = secrets.token_urlsafe(32)
    engine = request_engine(request)
    with Session(engine) as session:
        token = ApiToken(name=name.strip(), token_hash=_sha256(plaintext), expires_at=expires_at)
        session.add(token)
        session.commit()
    # Devolve a lista com o valor em claro, exibido uma unica vez no destaque.
    return await tokens_page(request, new_token=plaintext)


@router.delete("/{token_id}", response_class=HTMLResponse, include_in_schema=False)
async def revoke_token(request: Request, token_id: int, _: int = Depends(require_admin)):
    engine = request_engine(request)
    with Session(engine) as session:
        token = session.get(ApiToken, token_id)
        if token is None:
            raise HTTPException(status_code=404, detail="Token nao encontrado")
        session.delete(token)
        session.commit()
    return await list_tokens(request)
