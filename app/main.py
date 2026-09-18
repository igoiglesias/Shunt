from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config.settings import load_settings
from app.core.upstream import UpstreamPool
from app.routers.v1 import router as v1_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # `hasattr` respects state already injected: the route tests and the E2E
    # tests set `state.settings` and `state.pool` before entering the
    # TestClient context.
    if not hasattr(app.state, "settings"):
        app.state.settings = load_settings()
    if not hasattr(app.state, "pool"):
        app.state.pool = UpstreamPool(app.state.settings)
    yield
    await app.state.pool.aclose()


app = FastAPI(lifespan=lifespan)

app.include_router(v1_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
