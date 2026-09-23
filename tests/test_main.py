import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.core.upstream import UpstreamPool
from app.main import app


@pytest.fixture(autouse=True)
def _isolated_app_state():
    """Save and restore `app.state` around each test.

    Without this, the tests below share the module-level `app` object's
    state: the third test's lifespan only calls `_engine_or_none()` when
    `app.state.recorder` is absent, so a recorder left behind by an earlier
    test (engine None, built against the conftest-neutralized env vars)
    makes the boot return empty Settings no matter what the test sets.
    Either test run alone -- or in a different order -- would fail. Saving
    and restoring the actual attribute values (not just deleting them) also
    protects any test added later that pre-seeds state before entering the
    TestClient context.
    """
    saved = {}
    for attr in ("settings", "pool", "recorder"):
        if hasattr(app.state, attr):
            saved[attr] = getattr(app.state, attr)
            delattr(app.state, attr)
    try:
        yield
    finally:
        for attr in ("settings", "pool", "recorder"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)
        for attr, value in saved.items():
            setattr(app.state, attr, value)


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


def test_lifespan_loads_real_settings_and_builds_the_pool_from_them(monkeypatch, tmp_path):
    """Sem banco o catalogo e vazio; com banco o boot semeia e carrega dele."""
    monkeypatch.setenv("TURSO_DATABASE_URL", f"sqlite:///{tmp_path / 'stats.db'}")
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
