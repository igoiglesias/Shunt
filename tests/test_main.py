from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.core.upstream import UpstreamPool
from app.main import app


def test_lifespan_does_not_overwrite_injected_settings_and_pool():
    injected_settings = Settings(providers={}, models={}, routes=[], default_model=None)
    injected_pool = UpstreamPool(injected_settings)
    app.state.settings = injected_settings
    app.state.pool = injected_pool
    try:
        with TestClient(app):
            assert app.state.settings is injected_settings
            assert app.state.pool is injected_pool
    finally:
        del app.state.settings
        del app.state.pool


def test_lifespan_loads_real_settings_and_builds_the_pool_from_them():
    assert not hasattr(app.state, "settings")
    assert not hasattr(app.state, "pool")
    try:
        with TestClient(app):
            assert hasattr(app.state, "settings")
            assert hasattr(app.state, "pool")
            assert isinstance(app.state.settings, Settings)

            # Prove the pool was actually built FROM the loaded settings, not
            # from some other Settings object: a client for a provider
            # declared in the loaded config carries that provider's base_url.
            provider = next(iter(app.state.settings.providers))
            expected_base_url = app.state.settings.providers[provider].base_url
            client = app.state.pool.get(provider)
            assert str(client.base_url).rstrip("/") == expected_base_url.rstrip("/")
    finally:
        for attr in ("settings", "pool"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)
