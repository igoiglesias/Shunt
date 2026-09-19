from contextlib import asynccontextmanager

from fastapi import FastAPI

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
        app.state.recorder = Recorder(build_engine())
    set_recorder(app.state.recorder)
    await app.state.recorder.start()
    yield
    await app.state.recorder.aclose()
    await app.state.pool.aclose()


app = FastAPI(lifespan=lifespan)

app.include_router(v1_router)
app.include_router(dashboard_router)


@app.get("/")
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
