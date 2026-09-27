import asyncio
import copy
import logging
import os
import secrets
import tomllib
from contextlib import asynccontextmanager
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fastapi import Depends, FastAPI, Response
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.config.config import ENGINE_BOOT_TIMEOUT, TOKEN_CACHE_TTL
from app.config.seed import seed_catalog_if_empty
from app.config.settings import Settings, load_settings_from_db
from app.core.auth import LoginRequired, login_redirect, require_admin
from app.core.config_watcher import ConfigWatcher
from app.core.dispatcher import error_body
from app.core.observability import configure_logging, set_recorder
from app.core.prefix import TokenPrefixMiddleware
from app.core.token_auth import TokenCache, TokenRejected
from app.core.upstream import UpstreamPool
from app.routers.admin_auth import router as admin_auth_router
from app.routers.admin_config import apply_settings
from app.routers.admin_config import router as admin_config_router
from app.routers.admin_dashboard import router as admin_dashboard_router
from app.routers.admin_tokens import router as admin_tokens_router
from app.routers.admin_users import router as admin_users_router
from app.routers.audit import router as audit_router
from app.routers.dashboard import router as dashboard_router
from app.routers.v1 import DOCUMENTED_MODELS, protocol_of
from app.routers.v1 import router as v1_router
from app.stats.engine import build_engine, database_url
from app.stats.recorder import Recorder


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    # A persistencia e opcional: sem `TURSO_DATABASE_URL` o `Recorder` fica com
    # engine nula e `record()` vira um no-op, sem um `if` sequer no caminho da
    # requisicao.
    if not hasattr(app.state, "recorder"):
        # `reconnect` so existe quando ha URL declarada: sem banco nenhum o
        # gravador continua sendo um no-op, sem tentativa periodica de nada.
        reconnect = build_engine if database_url() else None
        app.state.recorder = Recorder(await _engine_or_none(), reconnect=reconnect)
    set_recorder(app.state.recorder)
    await app.state.recorder.start()
    # Vigia de versao: so com banco declarado (mesma regra do `reconnect`).
    # Le `recorder.engine` a cada ciclo porque o gravador reabre o banco
    # quando ele volta. Sempre atribuido: um teste anterior nao pode deixar um
    # vigia velho em `app.state`. Criado e com a versao inicial lida ANTES de
    # `_settings_for`: se o catalogo fosse carregado primeiro, uma edicao de
    # outro worker que aterrissasse entre os dois pontos gravaria a versao
    # NOVA em `seen` com o catalogo VELHO ja carregado, e o vigia nunca mais
    # recarregaria ate a proxima edicao real (medido: `seen: 1 | novo no
    # catalogo: False`). Nessa ordem, a mesma corrida so custa um reload
    # extra no proximo ciclo.
    url = database_url()
    app.state.config_watcher = (
        ConfigWatcher(
            engine_of=lambda: app.state.recorder.engine,
            apply=partial(apply_settings, app),
            url=url,
        )
        if url
        else None
    )
    if app.state.config_watcher is not None:
        await app.state.config_watcher.read_initial_version()
    # `hasattr` respects state already injected: the route tests and the E2E
    # tests set `state.settings` and `state.pool` before entering the
    # TestClient context.
    if not hasattr(app.state, "settings"):
        app.state.settings = _settings_for(app.state.recorder)
    if not hasattr(app.state, "pool"):
        app.state.pool = UpstreamPool(app.state.settings)
    if app.state.config_watcher is not None:
        await app.state.config_watcher.start()
    # Cache de validacao de token: o `hasattr` respeita o cache ja injetado
    # pelos testes (tests/conftest.py).
    if not hasattr(app.state, "token_cache"):
        app.state.token_cache = TokenCache(ttl=TOKEN_CACHE_TTL)
    # Segredo que assina o JWT de sessao do admin. Vem do ambiente para sessoes
    # sobrevivirem a restart; sem ele um segredo aleatorio e gerado no boot, e a
    # sessao morre quando o processo morre -- aceitavel para um proxy local.
    if not hasattr(app.state, "admin_session_secret"):
        app.state.admin_session_secret = (
            os.environ.get("ADMIN_SESSION_SECRET") or secrets.token_hex(32)
        )
    yield
    # `try/finally`: se `config_watcher.aclose()` levantar (ele pode
    # relevantar um cancelamento mirado no proprio caller do shutdown -- ver
    # `ConfigWatcher.aclose()`), o recorder e o pool tem de fechar do mesmo
    # jeito, senao a conexao com o banco e os clientes HTTP vazam.
    try:
        if app.state.config_watcher is not None:
            await app.state.config_watcher.aclose()
    finally:
        await app.state.recorder.aclose()
        await app.state.pool.aclose()


def _settings_for(recorder: Recorder) -> Settings:
    """O catalogo vem do banco, semeado no primeiro boot.

    `seed_catalog_if_empty` e idempotente: banco com providers nao muda nada,
    e o banco -- nao o codigo -- e a fonte do catalogo em operacao. Sem engine
    o proxy sobe igual, com `Settings` vazio; as rotas nao resolvem nada, que
    e a informacao certa para o painel em vez de um boot quebrado.
    """
    engine = recorder.engine
    if engine is None:
        return Settings(providers={}, models={}, routes=[])
    seed_catalog_if_empty(engine)
    with Session(engine) as session:
        return load_settings_from_db(session)


TEMPLATES = Path(__file__).parent / "templates"
ESTILO = TEMPLATES / "shunt.css"

# Prazo para o banco responder no boot (valor e override em `app/config/config.py`).
# Medido: com a URL apontada para uma porta morta, `create_all` nao levanta --
# ele PENDURA, e o proxy nunca chega a servir a primeira requisicao. Um extra
# que impede o servico de subir deixou de ser um extra, entao aqui ele tem
# relogio.


async def _engine_or_none():
    """A engine, ou None quando o banco demora demais ou falha.

    Roda em thread porque o dialeto do libsql e sincrono: sem isso o `wait_for`
    nao teria como interromper o laco de eventos parado numa chamada bloqueante.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(build_engine), timeout=ENGINE_BOOT_TIMEOUT
        )
    except TimeoutError:
        logging.getLogger("shunt").warning(
            "stats: banco nao respondeu em %.0fs no boot; persistencia desligada",
            ENGINE_BOOT_TIMEOUT,
        )
        return None

def _version() -> str:
    """A versao do `pyproject.toml`.

    Medido: o projeto nao e instalado como distribuicao (`uv sync` sem
    build-system), entao `importlib.metadata` levanta `PackageNotFoundError`
    rodando do repositorio. O `pyproject.toml` e a fonte nesse caso.
    """
    try:
        return version("shunt")
    except PackageNotFoundError:
        pass
    try:
        with (Path(__file__).parent.parent / "pyproject.toml").open("rb") as fh:
            return str(tomllib.load(fh)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "0.0.0"


DESCRIPTION = """\
Shunt is a local proxy that lets a client speaking one LLM API talk to a provider \
speaking another.

`/v1` speaks both dialects: the Anthropic Messages API (`/v1/messages`) and the \
OpenAI API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`). Each \
request is routed by its `model` to a configured provider, translated when the \
provider speaks the other dialect, and answered in the caller's own shape.

`/v1` needs no credential of its own. An `x-shunt-token` created in the admin area \
is checked, and makes Shunt send the provider key it holds. `x-api-key` and \
`Authorization: Bearer` are not checked: Shunt forwards them to transparent \
providers and replaces them with the configured key for the others.

`/api` and `/admin` need the admin session cookie `shunt_admin`, set by signing in \
at `/admin/login`.
"""

OPENAPI_TAGS = [
    {
        "name": "Anthropic",
        "description": "Anthropic Messages API: `/v1/messages` and token counting.",
    },
    {
        "name": "OpenAI",
        "description": "OpenAI API: chat completions, legacy completions and embeddings.",
    },
    {"name": "Models", "description": "The configured model aliases, in the caller's dialect."},
    {
        "name": "Panel",
        "description": "Usage statistics behind the admin panel. Admin session required.",
    },
    {
        "name": "Requests",
        "description": "The recorded requests, one by one. Admin session required.",
    },
    {
        "name": "Analysis",
        "description": "Aggregates over a period of requests. Admin session required.",
    },
    {"name": "Admin", "description": "Sign-in, users, tokens and catalog configuration."},
    {"name": "System", "description": "Liveness of the Shunt process."},
]

_FORWARDED = (
    "Shunt does not check it: it is forwarded to transparent providers and replaced "
    "by the configured provider key for the others."
)

SECURITY_SCHEMES = {
    "anthropicApiKey": {
        "type": "apiKey",
        "in": "header",
        "name": "x-api-key",
        "description": f"The client's Anthropic-style key. {_FORWARDED}",
    },
    "bearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "description": f"The client's bearer token. {_FORWARDED}",
    },
    "shuntToken": {
        "type": "apiKey",
        "in": "header",
        "name": "x-shunt-token",
        "description": (
            "A token created on the Tokens screen. Checked by Shunt: unknown or expired "
            "is a 401, and with no database it is a 503. When valid, Shunt sends the "
            "provider key it holds."
        ),
    },
    "adminSession": {
        "type": "apiKey",
        "in": "cookie",
        "name": "shunt_admin",
        "description": "The admin session cookie, set by signing in at `/admin/login`.",
    },
}

# `redirect_slashes=False` (medido no brief da Task 1.3): o 307 padrao de
# rota com barra final perdia qualquer prefixo `/t/<token>/` na Location, e
# nenhuma tela ou rota do app termina com barra.
app = FastAPI(
    title="Shunt",
    version=_version(),
    description=DESCRIPTION,
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
    redirect_slashes=False,
)
app.add_middleware(TokenPrefixMiddleware)


def _openapi() -> dict:
    """O documento padrao, mais o que as rotas citam e o FastAPI nao ve.

    As rotas `/v1` apontam por `$ref` para modelos que nao sao parametro de
    nenhuma funcao (o corpo e lido na mao), entao o FastAPI nao os poe em
    `components`. Sem este registro o Swagger mostraria uma `$ref` pendurada.

    A geracao, o cache e a invalidacao ficam com o `FastAPI.openapi` original:
    ele passa todos os argumentos do app (servers, webhooks, versao do OpenAPI,
    `separate_input_output_schemas`...) e regenera quando uma rota e incluida
    depois da primeira chamada. Medido no FastAPI 0.141: um override que so
    olha `app.openapi_schema` servia o documento velho sem a rota nova. Aqui
    so se acrescenta, e so num documento recem-gerado -- um objeto novo.
    """
    cached = app.openapi_schema
    schema = FastAPI.openapi(app)
    if schema is cached:
        return schema
    components = schema.setdefault("components", {})
    schemas = components.setdefault("schemas", {})
    for model in DOCUMENTED_MODELS:
        body = model.model_json_schema(ref_template="#/components/schemas/{model}")
        for name, definition in body.pop("$defs", {}).items():
            schemas.setdefault(name, definition)
        schemas.setdefault(model.__name__, body)
    # Copia: o documento e publico e mutavel (quem o ajusta depois nao pode
    # alterar a constante do modulo, nem o proximo documento gerado).
    components["securitySchemes"] = copy.deepcopy(SECURITY_SCHEMES)
    for operations in schema["paths"].values():
        for operation in operations.values():
            if not _can_fail_validation(operation):
                operation["responses"].pop("422", None)
    return schema


# O que um parametro `str` sem restricao gera no schema. Medido: `key: str`
# vira `{"type": "string", "title": "Key"}`; com `pattern`/`max_length` entram
# `pattern`/`maxLength`, e ai o 422 e real.
_PLAIN_STRING_KEYS = {"type", "title", "description"}


def _can_fail_validation(operation: dict) -> bool:
    """O FastAPI 0.141 poe 422 em toda rota com parametro declarado, e
    `openapi_extra` so mescla -- nao remove chave. Um 422 so e real quando
    algum parametro pode falhar: corpo validado pelo framework, query, header,
    cookie, ou caminho com tipo ou restricao. Caminho `str` puro nunca falha
    (`/v1/models/{model_id}`, `/api/requests/{request_id}`); as rotas POST de
    `/v1` leem o corpo na mao e nem declaram parametro.
    """
    # Limite conservador: o schema nao diz de onde veio o `requestBody` nem o
    # parametro. Um corpo declarado so em `openapi_extra` e lido na mao (que
    # nunca da 422), ou uma query so documentada ali, contam como "pode
    # falhar". Hoje nenhuma rota junta isso a um parametro declarado, entao
    # nao muda nada; importaria numa rota com `{id}: str` e corpo lido na mao,
    # que ficaria com um 422 que nao devolve.
    if "requestBody" in operation:
        return True
    return any(
        param.get("in") != "path"
        or param.get("schema", {}).get("type") != "string"
        or not set(param.get("schema", {})) <= _PLAIN_STRING_KEYS
        for param in operation.get("parameters", [])
    )


app.openapi = _openapi  # type: ignore[method-assign]


@app.exception_handler(LoginRequired)
async def _login_required_handler(request, exc):
    """Converte o sinal de sessao ausente em um redirect para a tela de login."""
    return login_redirect(request)


@app.exception_handler(TokenRejected)
async def _token_rejected_handler(request, exc):
    """O token recusado responde no envelope do protocolo da rota, nunca `detail`."""
    protocol = protocol_of(request.url.path, request.headers)
    return JSONResponse(status_code=exc.status, content=error_body(protocol, exc.status, exc.message))


app.include_router(v1_router)
# As duas APIs de dados (stats e auditoria) alimentam o painel e a tela de
# requisicoes; ambas exigem sessao, igual as paginas que as consomem.
app.include_router(dashboard_router, dependencies=[Depends(require_admin)])
app.include_router(audit_router, dependencies=[Depends(require_admin)])
app.include_router(admin_auth_router)
app.include_router(admin_dashboard_router)
app.include_router(admin_users_router)
app.include_router(admin_tokens_router)
app.include_router(admin_config_router)

@app.get(
    "/health",
    tags=["System"],
    summary="Liveness check",
    description=(
        "Answers `{\"status\": \"ok\"}` while the process is up, without touching "
        "providers or the database."
    ),
)
async def health() -> dict[str, str]:
    """Liveness check.

    Kept deliberately, not as skeleton leftover: a local proxy that harnesses
    point their `base_url` at needs a fast, dependency-free way for the
    operator (or a supervisor process) to confirm the server is up before
    routing real traffic to it. It intentionally does not touch
    `app.state.settings` or `app.state.pool` -- it must answer even if
    upstream configuration is broken, since that is exactly the situation an
    operator is trying to diagnose.
    """
    return {"status": "ok"}


# A pagina CSS continua lida do disco a cada carga, servida com `no-store`:
# o navegador guarda CSS sem `Cache-Control` pela heuristica dele, e um CSS
# velho em cache deixaria a tela nova com a cara antiga -- defeito medido.
NO_STORE = {"cache-control": "no-store, must-revalidate"}


@app.get("/", include_in_schema=False)
async def root():
    """Redireciona a raiz para o painel, dentro do admin.

    A home nao e mais a tela: o painel vive em `/admin/painel` e exige sessao.
    Quem abre `/` sem logar cai no redirect para o painel, que por sua vez vai
    ao login (o handler de `LoginRequired`). Com sessao, o fluxo e direto.
    """
    return RedirectResponse("/admin/painel", status_code=302)


@app.get("/shunt.css", include_in_schema=False)
async def estilo() -> Response:
    """A identidade visual das telas, num arquivo só.

    Servida com `no-store` como as paginas: ela muda junto com elas, e um CSS
    velho em cache deixaria a tela nova com a cara antiga -- o mesmo defeito
    que ja aconteceu com o HTML.
    """
    return Response(
        ESTILO.read_text(encoding="utf-8"), media_type="text/css", headers=NO_STORE
    )

from app.routers.relay import router as relay_router

app.include_router(relay_router)
