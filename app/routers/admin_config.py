"""API de administracao de configuracao (providers, models, routes, default)."""

import hmac
import json
import os
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.settings import Settings, load_settings
from app.stats.models import (
    ConfigVersion,
    Model,
    Provider,
    Route,
    RouteCandidate,
)
from app.templates import render

router = APIRouter(prefix="/admin/config", tags=["admin-config"])

# --- Helpers ---


# Nome e alcance do cookie que guarda o token do admin depois do primeiro
# acesso pelo navegador. O `path` cobre a pagina e todas as rotas HTMX abaixo
# dela, e nao vaza o cookie para o resto do proxy.
ADMIN_COOKIE = "shunt_admin"
ADMIN_COOKIE_PATH = "/admin/config"
ADMIN_COOKIE_MAX_AGE = 12 * 60 * 60


def admin_token_from(request: Request) -> str | None:
    """O token apresentado, na ordem header, cookie, query string.

    A query string e so a porta de entrada do navegador -- quem digita uma URL
    nao consegue mandar header. `admin_config_page` troca ela pelo cookie e
    redireciona, para o token nao ficar no historico nem no log de acesso.
    """
    return (
        request.headers.get("x-admin-token")
        or request.cookies.get(ADMIN_COOKIE)
        or request.query_params.get("token")
    )


def require_admin_token(request: Request) -> None:
    """Protege rotas admin com token simples via env.

    A comparacao e de tempo constante: `!=` em str sai no primeiro byte
    diferente, e a diferenca de tempo entre um prefixo certo e um errado deixa
    o token ser descoberto byte a byte.
    """
    admin_token = os.environ.get("ADMIN_TOKEN")
    if not admin_token:
        raise HTTPException(status_code=404, detail="Admin desabilitado")
    provided = admin_token_from(request)
    if not provided or not hmac.compare_digest(provided, admin_token):
        raise HTTPException(status_code=401, detail="Token invalido")


def cookie_handoff(request: Request) -> RedirectResponse:
    """Guarda o token no cookie e manda o navegador para a URL sem ele.

    303 e nao 302: o metodo vira GET e o navegador troca a URL da barra, que e
    o ponto -- o token some do historico, do `Referer` e do log de acesso
    depois de um unico hop. `secure` acompanha o esquema porque o proxy roda em
    http no localhost, e um cookie `secure` ali nunca seria enviado de volta.
    """
    response = RedirectResponse(url=ADMIN_COOKIE_PATH, status_code=303)
    response.set_cookie(
        ADMIN_COOKIE,
        request.query_params["token"],
        max_age=ADMIN_COOKIE_MAX_AGE,
        path=ADMIN_COOKIE_PATH,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
    )
    return response


def request_engine(request: Request):
    """A engine viva do processo, ou 503 quando a persistencia esta desligada.

    O gravador de estatisticas e o dono da engine (ele a reconecta quando o
    banco volta); pegar de `app.state` aqui mantem uma engine so por processo,
    igual ao painel e a auditoria.
    """
    recorder = getattr(request.app.state, "recorder", None)
    engine = recorder.engine if recorder is not None else None
    if engine is None:
        raise HTTPException(status_code=503, detail="Persistencia desligada")
    return engine


def snapshot_current_settings(session: Session) -> dict:
    """Gera snapshot JSON da configuracao atual."""
    providers = session.execute(select(Provider).order_by(Provider.id)).scalars().all()
    models = session.execute(select(Model).order_by(Model.id)).scalars().all()
    routes = session.execute(select(Route).order_by(Route.order_index)).scalars().all()
    for r in routes:
        r.candidates = session.execute(
            select(RouteCandidate)
            .where(RouteCandidate.route_id == r.id)
            .order_by(RouteCandidate.order_index)
        ).scalars().all()
    default_model = session.execute(select(Model).where(Model.is_default == True)).scalar_one_or_none()
    return {
        "providers": [
            {
                "name": p.name,
                "base_url": p.base_url,
                "protocol": p.protocol,
                "api_key_env": p.api_key_env,
            }
            for p in providers
        ],
        "models": [
            {
                "alias": m.alias,
                "provider": m.provider.name,
                "upstream_model": m.upstream_model,
                "supports_tools": m.supports_tools,
                "supports_streaming": m.supports_streaming,
                "supports_vision": m.supports_vision,
                "context_window": m.context_window,
                "max_output_tokens": m.max_output_tokens,
                "is_default": m.is_default,
            }
            for m in models
        ],
        "routes": [
            {
                "pattern": r.pattern,
                "order_index": r.order_index,
                "candidates": [c.model_alias for c in r.candidates],
            }
            for r in routes
        ],
        "default_model": default_model.alias if default_model else None,
    }


def create_config_version(session: Session) -> ConfigVersion:
    snap = snapshot_current_settings(session)
    cv = ConfigVersion(snapshot_json=json.dumps(snap, ensure_ascii=False))
    session.add(cv)
    session.commit()
    session.refresh(cv)
    return cv


# --- Pages ---


@router.get("", response_class=HTMLResponse)
async def admin_config_page(request: Request, _: None = Depends(require_admin_token)):
    """Pagina principal de administracao de configuracao."""
    if "token" in request.query_params:
        return cookie_handoff(request)
    engine = request_engine(request)
    with Session(engine) as session:
        providers = session.execute(select(Provider).order_by(Provider.name)).scalars().all()
        models = session.execute(select(Model).order_by(Model.alias)).scalars().all()
        routes = session.execute(select(Route).order_by(Route.order_index)).scalars().all()
        for r in routes:
            r.candidates = session.execute(
                select(RouteCandidate)
                .where(RouteCandidate.route_id == r.id)
                .order_by(RouteCandidate.order_index)
            ).scalars().all()
        default_model = session.execute(select(Model).where(Model.is_default == True)).scalar_one_or_none()
        versions = session.execute(
            select(ConfigVersion).order_by(ConfigVersion.created_at.desc()).limit(50)
        ).scalars().all()

    return render(
        "admin_config.html",
        request=request,
        providers=providers,
        models=models,
        routes=routes,
        default_model=default_model,
        versions=versions,
    )


# --- Providers ---


@router.get("/providers", response_class=HTMLResponse)
async def list_providers(request: Request, _: None = Depends(require_admin_token)):
    engine = request_engine(request)
    with Session(engine) as session:
        providers = session.execute(select(Provider).order_by(Provider.name)).scalars().all()
    return render("_providers.html", request=request, providers=providers)


@router.post("/providers", response_class=HTMLResponse)
async def create_provider(
    request: Request,
    name: Annotated[str, Form()],
    base_url: Annotated[str, Form()],
    protocol: Annotated[str, Form()],
    api_key_env: Annotated[str | None, Form()] = None,
    _: None = Depends(require_admin_token),
):
    engine = request_engine(request)
    with Session(engine) as session:
        if session.execute(select(Provider).where(Provider.name == name)).scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Provider ja existe")
        p = Provider(name=name, base_url=base_url, protocol=protocol, api_key_env=api_key_env)
        session.add(p)
        session.commit()
        create_config_version(session)
    return await list_providers(request)


@router.patch("/providers/{name}", response_class=HTMLResponse)
async def update_provider(
    request: Request,
    name: str,
    base_url: Annotated[str, Form()],
    protocol: Annotated[str, Form()],
    api_key_env: Annotated[str | None, Form()] = None,
    _: None = Depends(require_admin_token),
):
    engine = request_engine(request)
    with Session(engine) as session:
        p = session.execute(select(Provider).where(Provider.name == name)).scalar_one_or_none()
        if not p:
            raise HTTPException(status_code=404, detail="Provider nao encontrado")
        p.base_url = base_url
        p.protocol = protocol
        p.api_key_env = api_key_env
        session.commit()
        create_config_version(session)
    return await list_providers(request)


@router.delete("/providers/{name}", response_class=HTMLResponse)
async def delete_provider(
    request: Request,
    name: str,
    _: None = Depends(require_admin_token),
):
    engine = request_engine(request)
    with Session(engine) as session:
        p = session.execute(select(Provider).where(Provider.name == name)).scalar_one_or_none()
        if not p:
            raise HTTPException(status_code=404, detail="Provider nao encontrado")
        # Verifica se ha models referenciando
        if session.execute(select(Model).where(Model.provider_id == p.id)).first():
            raise HTTPException(status_code=400, detail="Provider em uso por models")
        session.delete(p)
        session.commit()
        create_config_version(session)
    return await list_providers(request)


@router.get("/providers/new", response_class=HTMLResponse)
async def new_provider_form(request: Request, _: None = Depends(require_admin_token)):
    return render("_provider_form.html", request=request, provider=None, providers=[])


@router.get("/providers/{name}/edit", response_class=HTMLResponse)
async def edit_provider_form(request: Request, name: str, _: None = Depends(require_admin_token)):
    engine = request_engine(request)
    with Session(engine) as session:
        p = session.execute(select(Provider).where(Provider.name == name)).scalar_one_or_none()
        if not p:
            raise HTTPException(status_code=404, detail="Provider nao encontrado")
    return render("_provider_form.html", request=request, provider=p, providers=[])


# --- Models ---


@router.get("/models", response_class=HTMLResponse)
async def list_models(request: Request, _: None = Depends(require_admin_token)):
    engine = request_engine(request)
    with Session(engine) as session:
        models = session.execute(select(Model).order_by(Model.alias)).scalars().all()
        providers = session.execute(select(Provider).order_by(Provider.name)).scalars().all()
    return render("_models.html", request=request, models=models, providers=providers)


@router.post("/models", response_class=HTMLResponse)
async def create_model(
    request: Request,
    alias: Annotated[str, Form()],
    provider_id: Annotated[int, Form()],
    upstream_model: Annotated[str, Form()],
    supports_tools: Annotated[bool, Form()] = True,
    supports_streaming: Annotated[bool, Form()] = True,
    supports_vision: Annotated[bool, Form()] = False,
    context_window: Annotated[int, Form()] = 131072,
    max_output_tokens: Annotated[int, Form()] = 8192,
    is_default: Annotated[bool, Form()] = False,
    _: None = Depends(require_admin_token),
):
    engine = request_engine(request)
    with Session(engine) as session:
        if session.execute(select(Model).where(Model.alias == alias)).scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Model alias ja existe")
        prov = session.get(Provider, provider_id)
        if not prov:
            raise HTTPException(status_code=400, detail="Provider invalido")
        # Se is_default, limpa default anterior
        if is_default:
            for m in session.execute(select(Model).where(Model.is_default == True)).scalars():
                m.is_default = False
        m = Model(
            alias=alias,
            provider_id=provider_id,
            upstream_model=upstream_model,
            supports_tools=supports_tools,
            supports_streaming=supports_streaming,
            supports_vision=supports_vision,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            is_default=is_default,
        )
        session.add(m)
        session.commit()
        create_config_version(session)
    return await list_models(request)


@router.patch("/models/{alias}", response_class=HTMLResponse)
async def update_model(
    request: Request,
    alias: str,
    provider_id: Annotated[int, Form()],
    upstream_model: Annotated[str, Form()],
    supports_tools: Annotated[bool, Form()] = True,
    supports_streaming: Annotated[bool, Form()] = True,
    supports_vision: Annotated[bool, Form()] = False,
    context_window: Annotated[int, Form()] = 131072,
    max_output_tokens: Annotated[int, Form()] = 8192,
    is_default: Annotated[bool, Form()] = False,
    _: None = Depends(require_admin_token),
):
    engine = request_engine(request)
    with Session(engine) as session:
        m = session.execute(select(Model).where(Model.alias == alias)).scalar_one_or_none()
        if not m:
            raise HTTPException(status_code=404, detail="Model nao encontrado")
        prov = session.get(Provider, provider_id)
        if not prov:
            raise HTTPException(status_code=400, detail="Provider invalido")
        if is_default and not m.is_default:
            for other in session.execute(select(Model).where(Model.is_default == True)).scalars():
                other.is_default = False
        elif not is_default and m.is_default:
            # Nao permite remover default se for o unico
            defaults = list(session.execute(select(Model).where(Model.is_default == True)).scalars())
            if len(defaults) == 1 and defaults[0].alias == alias:
                raise HTTPException(status_code=400, detail="Nao pode remover o unico default")
        m.provider_id = provider_id
        m.upstream_model = upstream_model
        m.supports_tools = supports_tools
        m.supports_streaming = supports_streaming
        m.supports_vision = supports_vision
        m.context_window = context_window
        m.max_output_tokens = max_output_tokens
        m.is_default = is_default
        session.commit()
        create_config_version(session)
    return await list_models(request)


@router.delete("/models/{alias}", response_class=HTMLResponse)
async def delete_model(
    request: Request,
    alias: str,
    _: None = Depends(require_admin_token),
):
    engine = request_engine(request)
    with Session(engine) as session:
        m = session.execute(select(Model).where(Model.alias == alias)).scalar_one_or_none()
        if not m:
            raise HTTPException(status_code=404, detail="Model nao encontrado")
        # Verifica se esta em routes
        if session.execute(select(RouteCandidate).where(RouteCandidate.model_alias == alias)).first():
            raise HTTPException(status_code=400, detail="Model em uso por routes")
        if m.is_default:
            raise HTTPException(status_code=400, detail="Nao pode deletar default model")
        session.delete(m)
        session.commit()
        create_config_version(session)
    return await list_models(request)


@router.get("/models/new", response_class=HTMLResponse)
async def new_model_form(request: Request, _: None = Depends(require_admin_token)):
    engine = request_engine(request)
    with Session(engine) as session:
        providers = session.execute(select(Provider).order_by(Provider.name)).scalars().all()
    return render("_model_form.html", request=request, model=None, providers=providers)


@router.get("/models/{alias}/edit", response_class=HTMLResponse)
async def edit_model_form(request: Request, alias: str, _: None = Depends(require_admin_token)):
    engine = request_engine(request)
    with Session(engine) as session:
        m = session.execute(select(Model).where(Model.alias == alias)).scalar_one_or_none()
        if not m:
            raise HTTPException(status_code=404, detail="Model nao encontrado")
        providers = session.execute(select(Provider).order_by(Provider.name)).scalars().all()
    return render("_model_form.html", request=request, model=m, providers=providers)


# --- Routes ---


@router.get("/routes", response_class=HTMLResponse)
async def list_routes(request: Request, _: None = Depends(require_admin_token)):
    engine = request_engine(request)
    with Session(engine) as session:
        routes = session.execute(select(Route).order_by(Route.order_index)).scalars().all()
        for r in routes:
            r.candidates = session.execute(
                select(RouteCandidate)
                .where(RouteCandidate.route_id == r.id)
                .order_by(RouteCandidate.order_index)
            ).scalars().all()
        models = session.execute(select(Model).order_by(Model.alias)).scalars().all()
    return render("_routes.html", request=request, routes=routes, models=models)


@router.post("/routes", response_class=HTMLResponse)
async def create_route(
    request: Request,
    pattern: Annotated[str, Form()],
    candidates: Annotated[list[str], Form()] = [],
    _: None = Depends(require_admin_token),
):
    if not pattern:
        raise HTTPException(status_code=400, detail="Pattern obrigatorio")
    if not candidates:
        raise HTTPException(status_code=400, detail="Pelo menos um candidato obrigatorio")
    engine = request_engine(request)
    with Session(engine) as session:
        # Valida que todos candidates existem
        for alias in candidates:
            if not session.execute(select(Model).where(Model.alias == alias)).scalar_one_or_none():
                raise HTTPException(status_code=400, detail=f"Model {alias} nao existe")
        # Proximo order_index
        max_order = session.execute(select(Route.order_index).order_by(Route.order_index.desc())).scalar() or -1
        route = Route(pattern=pattern, order_index=max_order + 1)
        session.add(route)
        session.flush()
        for idx, alias in enumerate(candidates):
            session.add(RouteCandidate(route_id=route.id, model_alias=alias, order_index=idx))
        session.commit()
        create_config_version(session)
    return await list_routes(request)


@router.patch("/routes/{route_id}", response_class=HTMLResponse)
async def update_route(
    request: Request,
    route_id: int,
    pattern: Annotated[str, Form()],
    candidates: Annotated[list[str], Form()] = [],
    _: None = Depends(require_admin_token),
):
    if not pattern:
        raise HTTPException(status_code=400, detail="Pattern obrigatorio")
    if not candidates:
        raise HTTPException(status_code=400, detail="Pelo menos um candidato obrigatorio")
    engine = request_engine(request)
    with Session(engine) as session:
        route = session.get(Route, route_id)
        if not route:
            raise HTTPException(status_code=404, detail="Route nao encontrado")
        for alias in candidates:
            if not session.execute(select(Model).where(Model.alias == alias)).scalar_one_or_none():
                raise HTTPException(status_code=400, detail=f"Model {alias} nao existe")
        route.pattern = pattern
        # Recria candidates
        session.execute(
            RouteCandidate.__table__.delete().where(RouteCandidate.route_id == route_id)
        )
        for idx, alias in enumerate(candidates):
            session.add(RouteCandidate(route_id=route.id, model_alias=alias, order_index=idx))
        session.commit()
        create_config_version(session)
    return await list_routes(request)


@router.post("/routes/reorder", response_class=HTMLResponse)
async def reorder_routes(
    request: Request,
    order: Annotated[list[int], Form()],
    _: None = Depends(require_admin_token),
):
    engine = request_engine(request)
    with Session(engine) as session:
        for idx, route_id in enumerate(order):
            route = session.get(Route, route_id)
            if route:
                route.order_index = idx
        session.commit()
        create_config_version(session)
    return await list_routes(request)


@router.delete("/routes/{route_id}", response_class=HTMLResponse)
async def delete_route(
    request: Request,
    route_id: int,
    _: None = Depends(require_admin_token),
):
    engine = request_engine(request)
    with Session(engine) as session:
        route = session.get(Route, route_id)
        if not route:
            raise HTTPException(status_code=404, detail="Route nao encontrado")
        session.delete(route)
        session.commit()
        create_config_version(session)
    return await list_routes(request)


@router.get("/routes/new", response_class=HTMLResponse)
async def new_route_form(request: Request, _: None = Depends(require_admin_token)):
    engine = request_engine(request)
    with Session(engine) as session:
        models = session.execute(select(Model).order_by(Model.alias)).scalars().all()
    return render("_route_form.html", request=request, route=None, models=models)


@router.get("/routes/{route_id}/edit", response_class=HTMLResponse)
async def edit_route_form(request: Request, route_id: int, _: None = Depends(require_admin_token)):
    engine = request_engine(request)
    with Session(engine) as session:
        route = session.get(Route, route_id)
        if not route:
            raise HTTPException(status_code=404, detail="Route nao encontrado")
        route.candidates = session.execute(
            select(RouteCandidate)
            .where(RouteCandidate.route_id == route_id)
            .order_by(RouteCandidate.order_index)
        ).scalars().all()
        models = session.execute(select(Model).order_by(Model.alias)).scalars().all()
    return render("_route_form.html", request=request, route=route, models=models)


# --- Default Model ---


@router.get("/default-model", response_class=HTMLResponse)
async def get_default_model(request: Request, _: None = Depends(require_admin_token)):
    engine = request_engine(request)
    with Session(engine) as session:
        default = session.execute(select(Model).where(Model.is_default == True)).scalar_one_or_none()
        models = session.execute(select(Model).order_by(Model.alias)).scalars().all()
    return render("_default.html", request=request, default=default, models=models)


@router.post("/default-model", response_class=HTMLResponse)
async def set_default_model(
    request: Request,
    alias: Annotated[str, Form()],
    _: None = Depends(require_admin_token),
):
    engine = request_engine(request)
    with Session(engine) as session:
        m = session.execute(select(Model).where(Model.alias == alias)).scalar_one_or_none()
        if not m:
            raise HTTPException(status_code=404, detail="Model nao encontrado")
        for other in session.execute(select(Model).where(Model.is_default == True)).scalars():
            other.is_default = False
        m.is_default = True
        session.commit()
        create_config_version(session)
    return await get_default_model(request)


# --- Reload & Rollback ---


@router.post("/reload")
async def reload_config(request: Request, _: None = Depends(require_admin_token)):
    """Invalida cache e recarrega settings do banco."""
    engine = request_engine(request)
    with Session(engine) as session:
        settings = load_settings(session)
    # Atualiza app.state.settings se existir
    from app.main import app
    app.state.settings = settings
    return JSONResponse({"status": "reloaded", "default_model": settings.default_model})


@router.post("/rollback/{version_id}", response_class=HTMLResponse)
async def rollback_config(
    request: Request,
    version_id: int,
    _: None = Depends(require_admin_token),
):
    engine = request_engine(request)
    with Session(engine) as session:
        cv = session.get(ConfigVersion, version_id)
        if not cv:
            raise HTTPException(status_code=404, detail="Versao nao encontrada")
        snap = json.loads(cv.snapshot_json)
        # Limpa tabelas de config
        session.execute(RouteCandidate.__table__.delete())
        session.execute(Route.__table__.delete())
        session.execute(Model.__table__.delete())
        session.execute(Provider.__table__.delete())
        # Reinsere do snapshot
        for pdata in snap["providers"]:
            p = Provider(**pdata)
            session.add(p)
        session.flush()
        for mdata in snap["models"]:
            prov = session.execute(select(Provider).where(Provider.name == mdata["provider"])).scalar_one()
            m = Model(
                alias=mdata["alias"],
                provider_id=prov.id,
                upstream_model=mdata["upstream_model"],
                supports_tools=mdata["supports_tools"],
                supports_streaming=mdata["supports_streaming"],
                supports_vision=mdata["supports_vision"],
                context_window=mdata["context_window"],
                max_output_tokens=mdata["max_output_tokens"],
                is_default=mdata.get("is_default", False),
            )
            session.add(m)
        session.flush()
        for rdata in snap["routes"]:
            route = Route(pattern=rdata["pattern"], order_index=rdata["order_index"])
            session.add(route)
            session.flush()
            for idx, alias in enumerate(rdata["candidates"]):
                session.add(RouteCandidate(route_id=route.id, model_alias=alias, order_index=idx))
        session.commit()
        create_config_version(session)
    return await admin_config_page(request)


@router.get("/history", response_class=HTMLResponse)
async def config_history(request: Request, _: None = Depends(require_admin_token)):
    engine = request_engine(request)
    with Session(engine) as session:
        versions = session.execute(
            select(ConfigVersion).order_by(ConfigVersion.created_at.desc()).limit(100)
        ).scalars().all()
    return render("_history.html", request=request, versions=versions)