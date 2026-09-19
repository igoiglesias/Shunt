import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.config.settings import load_settings
from app.core.observability import configure_logging, set_recorder
from app.core.upstream import UpstreamPool
from app.routers.dashboard import router as dashboard_router
from app.routers.v1 import router as v1_router
from app.stats.engine import build_engine
from app.stats.recorder import Recorder


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    # `hasattr` respects state already injected: the route tests and the E2E
    # tests set `state.settings` and `state.pool` before entering the
    # TestClient context.
    if not hasattr(app.state, "settings"):
        app.state.settings = load_settings()
    if not hasattr(app.state, "pool"):
        app.state.pool = UpstreamPool(app.state.settings)
    # A persistencia e opcional: sem `TURSO_DATABASE_URL` o `Recorder` fica com
    # engine nula e `record()` vira um no-op, sem um `if` sequer no caminho da
    # requisicao.
    if not hasattr(app.state, "recorder"):
        app.state.recorder = Recorder(await _engine_or_none())
    set_recorder(app.state.recorder)
    await app.state.recorder.start()
    yield
    await app.state.recorder.aclose()
    await app.state.pool.aclose()


DASHBOARD = Path(__file__).parent / "templates" / "dashboard.html"

# Prazo para o banco responder no boot. Medido: com a URL apontada para uma
# porta morta, `create_all` nao levanta -- ele PENDURA, e o proxy nunca chega a
# servir a primeira requisicao. Um extra que impede o servico de subir deixou
# de ser um extra, entao aqui ele tem relogio.
ENGINE_BOOT_TIMEOUT = 10.0


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

app.include_router(v1_router)
app.include_router(dashboard_router)


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


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    """O painel de uso, servido do disco a cada carga.

    Ler o arquivo por requisicao em vez de na importacao custa microssegundos e
    faz `make dev` recarregar a pagina sem reiniciar o processo. A home era a
    checagem de vida; ela mudou para `/health`, e o README registra a troca.
    """
    return DASHBOARD.read_text(encoding="utf-8")
