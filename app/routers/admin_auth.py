"""Entrada do painel: login por usuario+senha e criacao do primeiro admin.

As tres rotas aqui sao as unicas da area admin que NAO exigem sessao ativa --
sao a porta de entrada e a saida dela. O resto do admin passa por
`require_admin` (ver `app/core/auth.py`). O fluxo de primeiro acesso cria o
primeiro usuario direto na tela, sem ambiente nem seed: quando a tabela de
usuarios esta vazia, a tela de login vira a de criacao.
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.auth import ADMIN_COOKIE, ADMIN_COOKIE_MAX_AGE, ADMIN_COOKIE_PATH
from app.core.security import hash_password, issue_jwt, verify_password
from app.routers.admin_config import request_engine
from app.stats.models import User
from app.templates import render

router = APIRouter(prefix="/admin", tags=["admin-auth"])


def _set_session_cookie(response: RedirectResponse, request: Request, user_id: int) -> None:
    """Assina o JWT do usuario logado e guarda no cookie de sessao."""
    secret = request.app.state.admin_session_secret
    token = issue_jwt(user_id=user_id, secret=secret, ttl_seconds=ADMIN_COOKIE_MAX_AGE)
    response.set_cookie(
        ADMIN_COOKIE,
        token,
        max_age=ADMIN_COOKIE_MAX_AGE,
        path=ADMIN_COOKIE_PATH,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
    )


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """A porta de entrada. Sem usuarios no banco, vira a tela de criar o primeiro.

    O banco pode nao existir no boot (sem Turso) ou falhar; a tela de login/criacao
    deve responder mesmo assim: se nao ha engine, assume-se modo 'create' (primeiro
    acesso) e deixa o POST tentar abrir a sessao la.
    """
    engine = None
    recorder = getattr(request.app.state, "recorder", None)
    if recorder is not None:
        engine = recorder.engine
    mode = "create"
    if engine is not None:
        with Session(engine) as session:
            count = session.scalar(select(func.count()).select_from(User)) or 0
        if count > 0:
            mode = "login"
    return render("login.html", request=request, mode=mode, error=None)


@router.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    confirm: Annotated[str | None, Form()] = None,
):
    """Valida o login (ou cria o primeiro admin) e abre a sessao no cookie."""
    engine = request_engine(request)
    with Session(engine) as session:
        count = session.scalar(select(func.count()).select_from(User)) or 0
        if count == 0:
            # Primeiro acesso: esta e a tela de criacao. A senha precisa
            # confirmar com ela mesma e nao pode estar vazia.
            if not username or not password or password != confirm:
                resp = render(
                    "login.html", request=request, mode="create",
                    error="Preencha usuario e as duas senhas (iguais).",
                )
                resp.status_code = 400
                return resp
            user = User(username=username, password_hash=hash_password(password))
            session.add(user)
            session.commit()
            session.refresh(user)
            user_id = user.id
        else:
            existing = session.execute(
                select(User).where(User.username == username)
            ).scalar_one_or_none()
            if existing is None or not verify_password(password, existing.password_hash):
                resp = render(
                    "login.html", request=request, mode="login",
                    error="Usuario ou senha invalidos.",
                )
                resp.status_code = 401
                return resp
            existing.last_login_at = datetime.now(UTC)
            session.commit()
            user_id = existing.id

    target = RedirectResponse("/admin/painel", status_code=303)
    _set_session_cookie(target, request, user_id)
    return target


@router.post("/logout")
async def logout(request: Request):
    """Apaga o cookie de sessao e devolve o navegador ao login."""
    response = RedirectResponse("/admin/login", status_code=303)
    response.delete_cookie(
        ADMIN_COOKIE, path=ADMIN_COOKIE_PATH, httponly=True, samesite="strict"
    )
    return response
