import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.config.config import ENGINE_BOOT_TIMEOUT
from app.config.seed import seed_catalog_if_empty
from app.config.settings import Settings, load_settings_from_db
from app.core.auth import LoginRequired, login_redirect, require_admin
from app.core.observability import configure_logging, set_recorder
from app.core.upstream import UpstreamPool
from app.routers.admin_auth import router as admin_auth_router
from app.routers.admin_config import router as admin_config_router
from app.routers.admin_dashboard import router as admin_dashboard_router
from app.routers.admin_tokens import router as admin_tokens_router
from app.routers.admin_users import router as admin_users_router
from app.routers.audit import router as audit_router
from app.routers.dashboard import router as dashboard_router
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
    # `hasattr` respects state already injected: the route tests and the E2E
    # tests set `state.settings` and `state.pool` before entering the
    # TestClient context.
    if not hasattr(app.state, "settings"):
        app.state.settings = _settings_for(app.state.recorder)
    if not hasattr(app.state, "pool"):
        app.state.pool = UpstreamPool(app.state.settings)
    # Segredo que assina o JWT de sessao do admin. Vem do ambiente para sessoes
    # sobrevivirem a restart; sem ele um segredo aleatorio e gerado no boot, e a
    # sessao morre quando o processo morre -- aceitavel para um proxy local.
    if not hasattr(app.state, "admin_session_secret"):
        app.state.admin_session_secret = (
            os.environ.get("ADMIN_SESSION_SECRET") or secrets.token_hex(32)
        )
    yield
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

app = FastAPI(lifespan=lifespan)


@app.exception_handler(LoginRequired)
async def _login_required_handler(request, exc):
    """Converte o sinal de sessao ausente em um redirect para a tela de login."""
    return login_redirect(request)


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


@app.get("/health")
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


@app.get("/shunt.css")
async def estilo() -> Response:
    """A identidade visual das telas, num arquivo só.

    Servida com `no-store` como as paginas: ela muda junto com elas, e um CSS
    velho em cache deixaria a tela nova com a cara antiga -- o mesmo defeito
    que ja aconteceu com o HTML.
    """
    return Response(
        ESTILO.read_text(encoding="utf-8"), media_type="text/css", headers=NO_STORE
    )
