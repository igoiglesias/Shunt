import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.core.upstream import UpstreamPool
from app.main import app


@pytest.fixture(autouse=True)
def _isolated_app_state():
    """Save and restore `app.state` around each test.

    Without this, the two tests below share the module-level `app` object's
    state: the second test's `assert not hasattr(app.state, "settings")`
    only held because the first test's `finally` block had already deleted
    the attributes it set. Either test run alone -- or in a different order
    -- would fail. Saving and restoring the actual attribute values (not just
    deleting them) also protects any test added later that pre-seeds state
    before entering the TestClient context.
    """
    had_settings = hasattr(app.state, "settings")
    saved_settings = app.state.settings if had_settings else None
    had_pool = hasattr(app.state, "pool")
    saved_pool = app.state.pool if had_pool else None
    for attr in ("settings", "pool"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)
    try:
        yield
    finally:
        for attr in ("settings", "pool"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)
        if had_settings:
            app.state.settings = saved_settings
        if had_pool:
            app.state.pool = saved_pool


def test_health_endpoint_answers_ok_without_touching_state():
    """A checagem de vida mora em `/health` desde que o painel tomou a home."""
    injected_settings = Settings(providers={}, models={}, routes=[], default_model=None)
    injected_pool = UpstreamPool(injected_settings)
    app.state.settings = injected_settings
    app.state.pool = injected_pool
    try:
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    finally:
        del app.state.settings
        del app.state.pool


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
