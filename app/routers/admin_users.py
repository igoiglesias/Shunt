"""Manutencao de usuarios do painel: listar, criar, editar e excluir.

A area toda exige sessao (`require_admin`). Nao ha reset de senha por e-mail:
e um painel local, e quem administra define a senha na tela de edicao -- trocar
a senha de um usuario e abrir o form dele e preencher um novo valor. O usuario
logado nao se exclui quando e o ultimo admin, para nao trancar o acesso.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.auth import require_admin
from app.core.security import hash_password
from app.routers.admin_config import request_engine
from app.stats.models import User
from app.templates import render

router = APIRouter(
    prefix="/admin/users", tags=["admin-users"], dependencies=[Depends(require_admin)]
)


def _find(session: Session, username: str) -> User | None:
    return session.execute(select(User).where(User.username == username)).scalar_one_or_none()


def _all(session: Session) -> list[User]:
    return list(session.execute(select(User).order_by(User.username)).scalars().all())


@router.get("", response_class=HTMLResponse)
async def users_page(request: Request, current: int = Depends(require_admin)):
    engine = request_engine(request)
    with Session(engine) as session:
        users = _all(session)
    return render("users.html", request=request, users=users, active="users")


@router.get("/list", response_class=HTMLResponse)
async def list_users(request: Request, _: int = Depends(require_admin)):
    """A tabela de usuarios; o alvo de troca das operacoes via HTMX."""
    engine = request_engine(request)
    with Session(engine) as session:
        users = _all(session)
    return render("_user_list.html", request=request, users=users, active="users")


@router.get("/new", response_class=HTMLResponse)
async def new_user_form(request: Request, _: int = Depends(require_admin)):
    return render("_user_form.html", request=request, user=None, active="users")


@router.post("", response_class=HTMLResponse)
async def create_user(
    request: Request,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    _: int = Depends(require_admin),
):
    if not username or not password:
        raise HTTPException(status_code=400, detail="Usuario e senha sao obrigatorios")
    engine = request_engine(request)
    with Session(engine) as session:
        if _find(session, username.strip()):
            raise HTTPException(status_code=400, detail="Usuario ja existe")
        user = User(username=username.strip(), password_hash=hash_password(password))
        session.add(user)
        session.commit()
    return await list_users(request)


@router.get("/{username}/edit", response_class=HTMLResponse)
async def edit_user_form(
    request: Request, username: str, _: int = Depends(require_admin)
):
    engine = request_engine(request)
    with Session(engine) as session:
        user = _find(session, username)
    if user is None:
        raise HTTPException(status_code=404, detail="Usuario nao encontrado")
    return render("_user_form.html", request=request, user=user, active="users")


@router.patch("/{username}", response_class=HTMLResponse)
async def update_user(
    request: Request,
    username: str,
    new_username: Annotated[str | None, Form()] = None,
    new_password: Annotated[str | None, Form()] = None,
    _: int = Depends(require_admin),
):
    engine = request_engine(request)
    with Session(engine) as session:
        user = _find(session, username)
        if user is None:
            raise HTTPException(status_code=404, detail="Usuario nao encontrado")
        rename_to = (new_username or "").strip()
        if rename_to and rename_to != username and _find(session, rename_to):
            raise HTTPException(status_code=400, detail="Nome de usuario ja em uso")
        if rename_to:
            user.username = rename_to
        if new_password:
            user.password_hash = hash_password(new_password)
        session.commit()
    return await list_users(request)


@router.delete("/{username}", response_class=HTMLResponse)
async def delete_user(
    request: Request, username: str, current: int = Depends(require_admin)
):
    engine = request_engine(request)
    with Session(engine) as session:
        user = _find(session, username)
        if user is None:
            raise HTTPException(status_code=404, detail="Usuario nao encontrado")
        count = session.scalar(select(func.count()).select_from(User)) or 0
        if user.id == current and count == 1:
            raise HTTPException(status_code=400, detail="Exclusao do unico admin bloqueada")
        session.delete(user)
        session.commit()
    return await list_users(request)
