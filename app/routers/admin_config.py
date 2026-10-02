"""API de administracao de configuracao (providers, models, routes, default)."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config.settings import Settings, load_settings_from_db
from app.core.auth import require_admin
from app.core.upstream import UpstreamPool
from app.routers._openapi_docs import ADMIN_SESSION as _SESSION
from app.stats.models import (
    ConfigVersion,
    Model,
    Provider,
    Route,
    RouteCandidate,
)
from app.templates import render

# Sem tag no router: as telas HTML ficam fora do /docs, e a unica rota
# publicada (`/reload`) declara a sua propria tag.
router = APIRouter(prefix="/admin/config")


# --- Helpers ---


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


async def apply_settings(app, settings: Settings) -> None:
    """Troca as settings vivas do processo e reconcilia o pool que as usa.

    O pool nunca troca de identidade: recria-lo cortava streams em voo e
    zerava as contagens de concorrencia. `update` aposenta so o cliente cujo
    `base_url` mudou (fechado quando o ultimo uso termina) e ajusta os gates
    no lugar. Chamado pelo admin deste worker e pelo vigia de versao dos
    outros (`app/core/config_watcher.py`).
    """
    app.state.settings = settings
    pool = getattr(app.state, "pool", None)
    if pool is None:
        app.state.pool = UpstreamPool(settings)
        return
    await pool.update(settings)


# Teto do limite por provedor. Nenhum provedor real atende dez mil pedidos
# simultaneos num processo so; o teto existe para que um numero absurdo vire
# 400 aqui em vez de OverflowError (500) no commit do SQLite.
MAX_CONCURRENCY_CEILING = 10000


def _max_concurrency(raw: str | None) -> int | None:
    """Le o limite de concorrencia do form: vazio = sem limite (NULL).

    So digitos ASCII, de 1 ate o teto. `int()` sozinho aceitava `1_000`,
    `+5` e digitos arabe-indicos, e nao tinha teto. Qualquer outra coisa e 400
    antes de commitar: gravado, o valor quebraria `ProviderConfig` em todo
    reload seguinte.
    """
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    # O comprimento vem antes do `int()`: acima de 4300 digitos o Python
    # recusa a conversao com ValueError, e o pedido virava 500.
    if (
        not (text.isascii() and text.isdigit())
        or len(text) > len(str(MAX_CONCURRENCY_CEILING))
        or not 1 <= int(text) <= MAX_CONCURRENCY_CEILING
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Maximo de requisicoes simultaneas deve ser um inteiro entre 1 e "
                f"{MAX_CONCURRENCY_CEILING}"
            ),
        )
    return int(text)


def snapshot_current_settings(session: Session) -> dict:
    """Gera snapshot JSON da configuracao atual."""
    providers = session.execute(select(Provider).order_by(Provider.id)).scalars().all()
    models = session.execute(select(Model).order_by(Model.id)).scalars().all()
    routes = session.execute(select(Route).order_by(Route.order_index)).scalars().all()
    routes_with_candidates = []
    for r in routes:
        candidates = session.execute(
            select(RouteCandidate)
            .where(RouteCandidate.route_id == r.id)
            .order_by(RouteCandidate.order_index)
        ).scalars().all()
        routes_with_candidates.append((r, candidates))
    default_model = session.execute(select(Model).where(Model.is_default == True)).scalar_one_or_none()
    return {
        "providers": [
            {
                "name": p.name,
                "base_url": p.base_url,
                "protocol": p.protocol,
                # A chave bruta NUNCA entra no snapshot de rollback: se
                # gravarmos a verdadeira em `config_versions`, o banco de
                # auditoria fica com a credencial em claro. Registramos
                # apenas se ela existe; o valor real so muda no providers.
                "api_key": bool(p.api_key),
                # Snapshot antigo nao tem a chave: `Provider(**pdata)` no
                # rollback cai no default da coluna, NULL = sem limite.
                "max_concurrency": p.max_concurrency,
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
                "candidates": [c.model_alias for c in candidates],
            }
            for r, candidates in routes_with_candidates
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


@router.get("", response_class=HTMLResponse, include_in_schema=False)
async def admin_config_page(request: Request, _: None = Depends(require_admin)):
    """Pagina principal de administracao de configuracao."""
    engine = request_engine(request)
    with Session(engine) as session:
        providers = session.execute(select(Provider).order_by(Provider.name)).scalars().all()
        models = session.execute(select(Model).order_by(Model.alias)).scalars().all()
        routes_db = session.execute(select(Route).order_by(Route.order_index)).scalars().all()
        routes_with_candidates = []
        for r in routes_db:
            candidates = session.execute(
                select(RouteCandidate)
                .where(RouteCandidate.route_id == r.id)
                .order_by(RouteCandidate.order_index)
            ).scalars().all()
            routes_with_candidates.append((r, candidates))
        default_model = session.execute(select(Model).where(Model.is_default == True)).scalar_one_or_none()
        versions = session.execute(
            select(ConfigVersion).order_by(ConfigVersion.created_at.desc()).limit(50)
        ).scalars().all()

    return render(
        "admin_config.html",
        request=request,
        active="config",
        providers=providers,
        models=models,
        routes=routes_with_candidates,
        default_model=default_model,
        default=default_model,
        versions=versions,
    )


# --- Providers ---


@router.get("/providers", response_class=HTMLResponse, include_in_schema=False)
async def list_providers(request: Request, _: None = Depends(require_admin)):
    engine = request_engine(request)
    with Session(engine) as session:
        providers = session.execute(select(Provider).order_by(Provider.name)).scalars().all()
    return render("_providers.html", request=request, providers=providers)


@router.post("/providers", response_class=HTMLResponse, include_in_schema=False)
async def create_provider(
    request: Request,
    name: Annotated[str, Form()],
    base_url: Annotated[str, Form()],
    protocol: Annotated[str, Form()],
    api_key: Annotated[str | None, Form()] = None,
    max_concurrency: Annotated[str | None, Form()] = None,
    _: None = Depends(require_admin),
):
    limit = _max_concurrency(max_concurrency)
    engine = request_engine(request)
    with Session(engine) as session:
        if session.execute(select(Provider).where(Provider.name == name)).scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Provider ja existe")
        if protocol not in ("openai", "anthropic"):
            # Sem isso o valor invalido commita e load_settings_from_db levanta
            # ValidationError depois -> 500 em toda mutacao ate a linha sair.
            raise HTTPException(status_code=400, detail="Protocol deve ser openai ou anthropic")
        p = Provider(
            name=name,
            base_url=base_url,
            protocol=protocol,
            api_key=api_key,
            max_concurrency=limit,
        )
        session.add(p)
        session.commit()
        create_config_version(session)
        settings = load_settings_from_db(session)
    await apply_settings(request.app, settings)
    # Reset form via HX-Trigger so the UI clears without manual refresh
    resp = await list_providers(request)
    resp.headers["HX-Trigger"] = "providerFormReset"
    return resp


@router.patch("/providers/{name}", response_class=HTMLResponse, include_in_schema=False)
async def update_provider(
    request: Request,
    name: str,
    base_url: Annotated[str, Form()],
    protocol: Annotated[str, Form()],
    api_key: Annotated[str | None, Form()] = None,
    max_concurrency: Annotated[str | None, Form()] = None,
    _: None = Depends(require_admin),
):
    # O FastAPI entrega ausente e vazio como o mesmo None; aqui eles
    # significam coisas diferentes, entao a presenca vem do form cru.
    limit_sent = "max_concurrency" in await request.form()
    engine = request_engine(request)
    with Session(engine) as session:
        p = session.execute(select(Provider).where(Provider.name == name)).scalar_one_or_none()
        if not p:
            raise HTTPException(status_code=404, detail="Provider nao encontrado")
        if protocol not in ("openai", "anthropic"):
            raise HTTPException(status_code=400, detail="Protocol deve ser openai ou anthropic")
        limit = _max_concurrency(max_concurrency)
        p.base_url = base_url
        p.protocol = protocol
        # Campo AUSENTE preserva o limite (cliente direto que so mexe na URL,
        # mesmo contrato da chave). Presente e vazio SIGNIFICA sem limite: e
        # assim que o form, que sempre envia o campo, remove o limite.
        if limit_sent:
            p.max_concurrency = limit
        # Campo vazio (ausente no form chega como None, presente vazio como
        # None tambem no FastAPI) significa "nao mude a chave": so um valor
        # nao vazio sobrescreve. Assim o Salvar do formulario, que envia o
        # campo vazio por padrao, nunca apaga a chave gravada.
        if api_key:
            p.api_key = api_key
        session.commit()
        create_config_version(session)
        settings = load_settings_from_db(session)
    await apply_settings(request.app, settings)
    # Reset form via HX-Trigger so the UI clears without manual refresh
    resp = await list_providers(request)
    resp.headers["HX-Trigger"] = "providerFormReset"
    return resp


@router.delete("/providers/{name}", response_class=HTMLResponse, include_in_schema=False)
async def delete_provider(
    request: Request,
    name: str,
    _: None = Depends(require_admin),
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
        settings = load_settings_from_db(session)
    await apply_settings(request.app, settings)
    return await list_providers(request)


@router.post("/providers/test", response_class=HTMLResponse, include_in_schema=False)
async def test_provider(
    request: Request,
    base_url: Annotated[str, Form()],
    protocol: Annotated[str, Form()],
    api_key: Annotated[str | None, Form()] = None,
    _: None = Depends(require_admin),
):
    """Testa conexao com o provedor sem persistir. Devolve fragmento com resultado."""
    import httpx

    # Normaliza URL
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    test_url = ""
    method = "GET"
    json_body = None

    if protocol == "openai":
        test_url = f"{url}/models"
    elif protocol == "anthropic":
        # Anthropic precisa de POST com body minimo
        test_url = f"{url}/messages"
        method = "POST"
        json_body = {
            "model": "ping",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
        headers["anthropic-version"] = "2023-06-01"
        headers["content-type"] = "application/json"
    else:
        return HTMLResponse(
            '<span class="notice">✗ Protocolo desconhecido</span>',
            status_code=400,
        )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if method == "GET":
                resp = await client.get(test_url, headers=headers)
            else:
                resp = await client.post(test_url, headers=headers, json=json_body)

        if resp.is_success:
            return HTMLResponse(
                f'<span style="color: var(--good)">✓ Conectado ({resp.elapsed.total_seconds()*1000:.0f} ms)</span>'
            )
        else:
            return HTMLResponse(
                f'<span style="color: var(--error)">✗ {resp.status_code}: {resp.text[:200]}</span>',
                status_code=400,
            )
    except httpx.TimeoutException:
        return HTMLResponse(
            '<span style="color: var(--error)">✗ Timeout (10s)</span>',
            status_code=408,
        )
    except httpx.HTTPError as e:
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        return HTMLResponse(
            f'<span style="color: var(--error)">✗ {msg}</span>',
            status_code=500,
        )


@router.get("/providers/new", response_class=HTMLResponse, include_in_schema=False)
async def new_provider_form(request: Request, _: None = Depends(require_admin)):
    return render("_provider_form.html", request=request, provider=None, providers=[])


@router.get("/providers/{name}/edit", response_class=HTMLResponse, include_in_schema=False)
async def edit_provider_form(request: Request, name: str, _: None = Depends(require_admin)):
    engine = request_engine(request)
    with Session(engine) as session:
        p = session.execute(select(Provider).where(Provider.name == name)).scalar_one_or_none()
        if not p:
            raise HTTPException(status_code=404, detail="Provider nao encontrado")
    return render("_provider_form.html", request=request, provider=p, providers=[])


# --- Forms ---

# Ancoras de form por secao: o botao "Cancelar" troca o form aberto por esta
# ancora vazia via swap outerHTML, fechando o form sem recarregar a pagina.
_FORM_SECTIONS = ("provider", "model", "route")


@router.get("/sections/{name}/form-reset", response_class=HTMLResponse, include_in_schema=False)
async def form_reset(name: str, _: None = Depends(require_admin)):
    if name not in _FORM_SECTIONS:
        raise HTTPException(status_code=404, detail="Secao desconhecida")
    return HTMLResponse(f'<div id="{name}-form"></div>')


# --- Models ---


@router.get("/models", response_class=HTMLResponse, include_in_schema=False)
async def list_models(request: Request, _: None = Depends(require_admin)):
    engine = request_engine(request)
    with Session(engine) as session:
        models = session.execute(select(Model).order_by(Model.alias)).scalars().all()
        providers = session.execute(select(Provider).order_by(Provider.name)).scalars().all()
    return render("_models.html", request=request, models=models, providers=providers)


def _flag(values: list[str] | None, default: bool) -> bool:
    """Deriva um bool do form sem depender da ordem das chaves.

    O template manda um hidden companion `false` ANTES do checkbox `true`:
    desmarcado envia uma chave, marcado envia duas iguais. Derivar de
    `values[-1]` (o que o Starlette faz) deixaria o resultado depender da
    posicao dos inputs no template; derivar do conteudo nao."""
    return default if values is None else ("true" in values)


@router.post("/models", response_class=HTMLResponse, include_in_schema=False)
async def create_model(
    request: Request,
    alias: Annotated[str, Form()],
    provider_id: Annotated[int, Form()],
    upstream_model: Annotated[str, Form()],
    supports_tools: Annotated[list[str] | None, Form()] = None,
    supports_streaming: Annotated[list[str] | None, Form()] = None,
    supports_vision: Annotated[list[str] | None, Form()] = None,
    context_window: Annotated[int, Form()] = 131072,
    max_output_tokens: Annotated[int, Form()] = 8192,
    is_default: Annotated[list[str] | None, Form()] = None,
    _: None = Depends(require_admin),
):
    tools = _flag(supports_tools, True)
    streaming = _flag(supports_streaming, True)
    vision = _flag(supports_vision, False)
    default_flag = _flag(is_default, False)
    engine = request_engine(request)
    with Session(engine) as session:
        if session.execute(select(Model).where(Model.alias == alias)).scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Model alias ja existe")
        prov = session.get(Provider, provider_id)
        if not prov:
            raise HTTPException(status_code=400, detail="Provider invalido")
        # Se is_default, limpa default anterior
        if default_flag:
            for m in session.execute(select(Model).where(Model.is_default == True)).scalars():
                m.is_default = False
        m = Model(
            alias=alias,
            provider_id=provider_id,
            upstream_model=upstream_model,
            supports_tools=tools,
            supports_streaming=streaming,
            supports_vision=vision,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            is_default=default_flag,
        )
        session.add(m)
        session.commit()
        create_config_version(session)
        settings = load_settings_from_db(session)
    await apply_settings(request.app, settings)
    # Reset form via HX-Trigger so the UI clears without manual refresh
    resp = await list_models(request)
    resp.headers["HX-Trigger"] = "providerFormReset"
    return resp


@router.patch("/models/{alias}", response_class=HTMLResponse, include_in_schema=False)
async def update_model(
    request: Request,
    alias: str,
    provider_id: Annotated[int, Form()],
    upstream_model: Annotated[str, Form()],
    supports_tools: Annotated[list[str] | None, Form()] = None,
    supports_streaming: Annotated[list[str] | None, Form()] = None,
    supports_vision: Annotated[list[str] | None, Form()] = None,
    context_window: Annotated[int | None, Form()] = None,
    max_output_tokens: Annotated[int | None, Form()] = None,
    is_default: Annotated[list[str] | None, Form()] = None,
    _: None = Depends(require_admin),
):
    engine = request_engine(request)
    with Session(engine) as session:
        m = session.execute(select(Model).where(Model.alias == alias)).scalar_one_or_none()
        if not m:
            raise HTTPException(status_code=404, detail="Model nao encontrado")
        # Campo ausente preserva o valor atual (o form sempre envia, mas um
        # cliente direto pode omitir; nesse caso nao inventar default). Vale
        # para os flags e para os dois ints: omisso nunca zera nem reseta.
        tools = _flag(supports_tools, m.supports_tools)
        streaming = _flag(supports_streaming, m.supports_streaming)
        vision = _flag(supports_vision, m.supports_vision)
        default_flag = _flag(is_default, m.is_default)
        if context_window is None:
            context_window = m.context_window
        if max_output_tokens is None:
            max_output_tokens = m.max_output_tokens
        prov = session.get(Provider, provider_id)
        if not prov:
            raise HTTPException(status_code=400, detail="Provider invalido")
        if default_flag and not m.is_default:
            for other in session.execute(select(Model).where(Model.is_default == True)).scalars():
                other.is_default = False
        elif not default_flag and m.is_default:
            # Nao permite remover default se for o unico
            defaults = list(session.execute(select(Model).where(Model.is_default == True)).scalars())
            if len(defaults) == 1 and defaults[0].alias == alias:
                raise HTTPException(status_code=400, detail="Nao pode remover o unico default")
        m.provider_id = provider_id
        m.upstream_model = upstream_model
        m.supports_tools = tools
        m.supports_streaming = streaming
        m.supports_vision = vision
        m.context_window = context_window
        m.max_output_tokens = max_output_tokens
        m.is_default = default_flag
        session.commit()
        create_config_version(session)
        settings = load_settings_from_db(session)
    await apply_settings(request.app, settings)
    # Reset form via HX-Trigger so the UI clears without manual refresh
    resp = await list_models(request)
    resp.headers["HX-Trigger"] = "providerFormReset"
    return resp


@router.delete("/models/{alias}", response_class=HTMLResponse, include_in_schema=False)
async def delete_model(
    request: Request,
    alias: str,
    _: None = Depends(require_admin),
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
        settings = load_settings_from_db(session)
    await apply_settings(request.app, settings)
    return await list_models(request)


@router.get("/models/new", response_class=HTMLResponse, include_in_schema=False)
async def new_model_form(request: Request, _: None = Depends(require_admin)):
    engine = request_engine(request)
    with Session(engine) as session:
        providers = session.execute(select(Provider).order_by(Provider.name)).scalars().all()
    return render("_model_form.html", request=request, model=None, providers=providers)


@router.get("/models/{alias}/edit", response_class=HTMLResponse, include_in_schema=False)
async def edit_model_form(request: Request, alias: str, _: None = Depends(require_admin)):
    engine = request_engine(request)
    with Session(engine) as session:
        m = session.execute(select(Model).where(Model.alias == alias)).scalar_one_or_none()
        if not m:
            raise HTTPException(status_code=404, detail="Model nao encontrado")
        providers = session.execute(select(Provider).order_by(Provider.name)).scalars().all()
    return render("_model_form.html", request=request, model=m, providers=providers)


# --- Routes ---


@router.get("/routes", response_class=HTMLResponse, include_in_schema=False)
async def list_routes(request: Request, _: None = Depends(require_admin)):
    engine = request_engine(request)
    with Session(engine) as session:
        routes_db = session.execute(select(Route).order_by(Route.order_index)).scalars().all()
        routes_with_candidates = []
        for r in routes_db:
            candidates = session.execute(
                select(RouteCandidate)
                .where(RouteCandidate.route_id == r.id)
                .order_by(RouteCandidate.order_index)
            ).scalars().all()
            routes_with_candidates.append((r, candidates))
        models = session.execute(select(Model).order_by(Model.alias)).scalars().all()
    return render("_routes.html", request=request, routes=routes_with_candidates, models=models)


@router.post("/routes", response_class=HTMLResponse, include_in_schema=False)
async def create_route(
    request: Request,
    pattern: Annotated[str | None, Form()] = None,
    candidates: Annotated[list[str] | None, Form()] = None,
    _: None = Depends(require_admin),
):
    if not pattern:
        raise HTTPException(status_code=400, detail="Pattern obrigatorio")
    # <select name="candidates"> deixado em "— selecione —" manda value="";
    # filtra vazio e duplicatas (mesmo alias duas vezes = fallback sem sentido),
    # preservando a ordem de prioridade.
    candidates = list(dict.fromkeys(a for a in (candidates or []) if a))
    if not candidates:
        raise HTTPException(status_code=400, detail="Pelo menos um candidato obrigatorio")
    engine = request_engine(request)
    with Session(engine) as session:
        # Valida que todos candidates existem
        for alias in candidates:
            if not session.execute(select(Model).where(Model.alias == alias)).scalar_one_or_none():
                raise HTTPException(status_code=400, detail=f"Model {alias} nao existe")
        # Proximo order_index (None so quando nao ha rota; 0 e valido, entao
        # nulo-teste explicito em vez de `or -1`, que devolveria -1 pro max 0)
        max_order = session.execute(select(Route.order_index).order_by(Route.order_index.desc())).scalar()
        route = Route(pattern=pattern, order_index=(max_order if max_order is not None else -1) + 1)
        session.add(route)
        session.flush()
        for idx, alias in enumerate(candidates):
            session.add(RouteCandidate(route_id=route.id, model_alias=alias, order_index=idx))
        session.commit()
        create_config_version(session)
        settings = load_settings_from_db(session)
    await apply_settings(request.app, settings)
    # Reset form via HX-Trigger so the UI clears without manual refresh
    resp = await list_routes(request)
    resp.headers["HX-Trigger"] = "providerFormReset"
    return resp


@router.patch("/routes/{route_id}", response_class=HTMLResponse, include_in_schema=False)
async def update_route(
    request: Request,
    route_id: int,
    pattern: Annotated[str | None, Form()] = None,
    candidates: Annotated[list[str] | None, Form()] = None,
    _: None = Depends(require_admin),
):
    if not pattern:
        raise HTTPException(status_code=400, detail="Pattern obrigatorio")
    # Mesma limpeza do create: descarta option vazio e duplicatas
    candidates = list(dict.fromkeys(a for a in (candidates or []) if a))
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
        session.execute(delete(RouteCandidate).where(RouteCandidate.route_id == route_id))
        for idx, alias in enumerate(candidates):
            session.add(RouteCandidate(route_id=route.id, model_alias=alias, order_index=idx))
        session.commit()
        create_config_version(session)
        settings = load_settings_from_db(session)
    await apply_settings(request.app, settings)
    return await list_routes(request)


@router.post("/routes/reorder", response_class=HTMLResponse, include_in_schema=False)
async def reorder_routes(
    request: Request,
    order: Annotated[list[int], Form()],
    _: None = Depends(require_admin),
):
    engine = request_engine(request)
    with Session(engine) as session:
        existing = [r.id for r in session.execute(select(Route)).scalars().all()]
        # `order` tem de ser uma permutacao das rotas existentes: payload parcial
        # ou com id duplicado deixaria order_index repetido e prioridade de
        # fallback indeterminada.
        if sorted(order) != sorted(existing):
            raise HTTPException(status_code=400, detail="Order nao cobre todas as rotas exatamente uma vez")
        for idx, route_id in enumerate(order):
            route = session.get(Route, route_id)
            if route:
                route.order_index = idx
        session.commit()
        create_config_version(session)
        settings = load_settings_from_db(session)
    await apply_settings(request.app, settings)
    # Reset form via HX-Trigger so the UI clears without manual refresh
    resp = await list_routes(request)
    resp.headers["HX-Trigger"] = "providerFormReset"
    return resp


@router.delete("/routes/{route_id}", response_class=HTMLResponse, include_in_schema=False)
async def delete_route(
    request: Request,
    route_id: int,
    _: None = Depends(require_admin),
):
    engine = request_engine(request)
    with Session(engine) as session:
        route = session.get(Route, route_id)
        if not route:
            raise HTTPException(status_code=404, detail="Route nao encontrado")
        # Candidados antes: a FK route_candidates.route_id e nullable=False e a
        # relacao nao tem cascade -- session.delete(route) nullificaria o id e
        # quebraria o commit com IntegrityError.
        session.execute(delete(RouteCandidate).where(RouteCandidate.route_id == route_id))
        session.delete(route)
        session.commit()
        create_config_version(session)
        settings = load_settings_from_db(session)
    await apply_settings(request.app, settings)
    return await list_routes(request)


@router.get("/routes/new", response_class=HTMLResponse, include_in_schema=False)
async def new_route_form(request: Request, _: None = Depends(require_admin)):
    engine = request_engine(request)
    with Session(engine) as session:
        models = session.execute(select(Model).order_by(Model.alias)).scalars().all()
    return render("_route_form.html", request=request, route=None, models=models)


@router.get("/routes/{route_id}/edit", response_class=HTMLResponse, include_in_schema=False)
async def edit_route_form(request: Request, route_id: int, _: None = Depends(require_admin)):
    engine = request_engine(request)
    with Session(engine) as session:
        route = session.get(Route, route_id)
        if not route:
            raise HTTPException(status_code=404, detail="Route nao encontrado")
        candidates = session.execute(
            select(RouteCandidate)
            .where(RouteCandidate.route_id == route_id)
            .order_by(RouteCandidate.order_index)
        ).scalars().all()
        models = session.execute(select(Model).order_by(Model.alias)).scalars().all()
    return render("_route_form.html", request=request, route=route, models=models, candidates=candidates)


# --- Default Model ---


@router.get("/default-model", response_class=HTMLResponse, include_in_schema=False)
async def get_default_model(request: Request, _: None = Depends(require_admin)):
    engine = request_engine(request)
    with Session(engine) as session:
        default = session.execute(select(Model).where(Model.is_default == True)).scalar_one_or_none()
        models = session.execute(select(Model).order_by(Model.alias)).scalars().all()
    return render("_default.html", request=request, default=default, models=models)


@router.post("/default-model", response_class=HTMLResponse, include_in_schema=False)
async def set_default_model(
    request: Request,
    alias: Annotated[str, Form()],
    _: None = Depends(require_admin),
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
        settings = load_settings_from_db(session)
    await apply_settings(request.app, settings)
    return await get_default_model(request)


# --- Reload & Rollback ---


@router.post(
    "/reload",
    tags=["Admin"],
    summary="Reload configuration from the database",
    description=(
        "Rebuilds the providers, models and routes from the database and applies them "
        "to the running process, without a restart. Use it after changing the "
        "configuration tables directly."
    ),
    responses={
        "200": {
            "description": "Configuration reloaded.",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["reloaded"]},
                            "default_model": {"type": ["string", "null"]},
                        },
                    }
                }
            },
        },
        "303": {
            "description": "No valid session and the client is a browser: redirect to "
            "the login page."
        },
        "401": {
            "description": "No valid session and the client sent `Accept: "
            "application/json` or `HX-Request: true`."
        },
        "503": {"description": "Persistence is off, so there is no configuration to load."},
    },
    openapi_extra={"security": _SESSION},
)
async def reload_config(request: Request, _: None = Depends(require_admin)):
    """Invalida cache e recarrega settings do banco."""
    engine = request_engine(request)
    with Session(engine) as session:
        settings = load_settings_from_db(session)
    await apply_settings(request.app, settings)
    return JSONResponse({"status": "reloaded", "default_model": settings.default_model})


@router.post("/rollback/{version_id}", response_class=HTMLResponse, include_in_schema=False)
async def rollback_config(
    request: Request,
    version_id: int,
    _: None = Depends(require_admin),
):
    engine = request_engine(request)
    with Session(engine) as session:
        cv = session.get(ConfigVersion, version_id)
        if not cv:
            raise HTTPException(status_code=404, detail="Versao nao encontrada")
        snap = json.loads(cv.snapshot_json)
        # O snapshot nao carrega a chave bruta (so o bool de existencia, para a
        # credencial nunca entrar no historico). A chave atual por nome de
        # provider sobrevive ao rollback; provider novo no snapshot fica sem chave.
        current_keys = {
            p.name: p.api_key for p in session.execute(select(Provider)).scalars()
        }
        # Limpa tabelas de config (ordem respeita FKs)
        session.execute(delete(RouteCandidate))
        session.execute(delete(Route))
        session.execute(delete(Model))
        session.execute(delete(Provider))
        # Reinsere do snapshot
        for pdata in snap["providers"]:
            p = Provider(**pdata)
            p.api_key = current_keys.get(p.name)
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
        settings = load_settings_from_db(session)
    await apply_settings(request.app, settings)
    return await admin_config_page(request)


@router.get("/history", response_class=HTMLResponse, include_in_schema=False)
async def config_history(request: Request, _: None = Depends(require_admin)):
    engine = request_engine(request)
    with Session(engine) as session:
        versions = session.execute(
            select(ConfigVersion).order_by(ConfigVersion.created_at.desc()).limit(100)
        ).scalars().all()
    return render("_history.html", request=request, versions=versions)