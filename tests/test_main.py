import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.core.upstream import UpstreamPool
from app.main import app
from app.routers.admin_config import create_config_version
from app.stats.models import Provider


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


def test_lifespan_has_no_config_watcher_without_a_database():
    """conftest esvazia TURSO_DATABASE_URL: sem banco nao ha o que vigiar."""
    with TestClient(app):
        assert app.state.config_watcher is None


def test_lifespan_starts_and_stops_the_config_watcher_with_a_database(monkeypatch, tmp_path):
    from app.core.config_watcher import ConfigWatcher

    monkeypatch.setenv("TURSO_DATABASE_URL", f"sqlite:///{tmp_path / 'stats.db'}")
    try:
        with TestClient(app):
            watcher = app.state.config_watcher
            assert isinstance(watcher, ConfigWatcher)
            assert watcher._task is not None and not watcher._task.done()
            assert watcher.seen is None  # boot novo: seed nao cria versao
        assert watcher._task is None  # aclose no shutdown, antes do pool
    finally:
        for attr in ("settings", "pool"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)


def test_config_watcher_engine_of_reads_the_recorder_live(monkeypatch, tmp_path):
    """`engine_of` tem de ler `app.state.recorder.engine` a cada chamada, nao
    capturar a engine do boot: o gravador reabre o banco quando ele volta
    (`Recorder.reconnect`), e uma engine congelada no boot deixaria o vigia
    consultando uma conexao morta para sempre."""
    monkeypatch.setenv("TURSO_DATABASE_URL", f"sqlite:///{tmp_path / 'stats.db'}")
    try:
        with TestClient(app):
            watcher = app.state.config_watcher
            engine_at_boot = watcher._engine_of()
            sentinel = object()
            app.state.recorder._engine = sentinel
            assert watcher._engine_of() is sentinel
            app.state.recorder._engine = engine_at_boot
    finally:
        for attr in ("settings", "pool"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)


def test_a_concurrent_edit_between_the_initial_read_and_the_catalog_load_converges(
    monkeypatch, tmp_path
):
    """I-1 (reviewer): se o catalogo fosse carregado ANTES da versao inicial
    ser lida, uma edicao de outro worker que aterrissasse entre os dois
    pontos gravaria a versao NOVA em `seen` com o catalogo VELHO ja
    carregado -- `check_once` nunca mais recarregaria ate a PROXIMA edicao
    real (`version == seen` sempre bateria contra a versao ja vista, medido
    pelo reviewer: `seen: 1 | novo no catalogo: False`). Simula a corrida
    injetando a edicao dentro de `_settings_for` (chamado DEPOIS de
    `read_initial_version()` no lifespan real) e prova que um ciclo do vigia
    converge -- o custo aceito e' so' um reload extra, nao um catalogo preso
    para sempre."""
    import app.main as main_module

    monkeypatch.setenv("TURSO_DATABASE_URL", f"sqlite:///{tmp_path / 'stats.db'}")
    original_settings_for = main_module._settings_for

    def racing_settings_for(recorder):
        settings = original_settings_for(recorder)
        # Edicao de outro worker chegando ENTRE a leitura da versao inicial
        # (ja feita pelo lifespan) e o carregamento do catalogo.
        with Session(recorder.engine) as session:
            session.add(Provider(name="corrida", base_url="http://corrida.test", protocol="openai"))
            session.commit()
            create_config_version(session)
        return settings

    monkeypatch.setattr(main_module, "_settings_for", racing_settings_for)
    try:
        with TestClient(app):
            # Catalogo carregado ANTES da corrida: nao ve o provider novo.
            assert "corrida" not in app.state.settings.providers
            watcher = app.state.config_watcher
            reloaded = asyncio.run(watcher.check_once())
            assert reloaded is True
            assert "corrida" in app.state.settings.providers
    finally:
        for attr in ("settings", "pool"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)


def test_shutdown_still_closes_the_recorder_and_the_pool_when_the_watcher_aclose_raises(
    monkeypatch, tmp_path
):
    """M-1 (ruling do controller): `aclose()` pode relevantar um
    cancelamento mirado no proprio caller do shutdown (ver
    `ConfigWatcher.aclose()`). Sem `try/finally`, essa excecao pularia
    `recorder.aclose()` e `pool.aclose()`, vazando a conexao com o banco e
    os clientes HTTP."""
    from app.core.upstream import UpstreamPool
    from app.stats.recorder import Recorder

    monkeypatch.setenv("TURSO_DATABASE_URL", f"sqlite:///{tmp_path / 'stats.db'}")
    calls: list[str] = []
    original_recorder_aclose = Recorder.aclose
    original_pool_aclose = UpstreamPool.aclose

    async def spy_recorder_aclose(self):
        calls.append("recorder")
        await original_recorder_aclose(self)

    async def spy_pool_aclose(self):
        calls.append("pool")
        await original_pool_aclose(self)

    monkeypatch.setattr(Recorder, "aclose", spy_recorder_aclose)
    monkeypatch.setattr(UpstreamPool, "aclose", spy_pool_aclose)
    try:
        with TestClient(app):

            async def exploding_watcher_aclose():
                calls.append("watcher")
                raise RuntimeError("cancelamento do caller, propagado por aclose()")

            app.state.config_watcher.aclose = exploding_watcher_aclose
        raise AssertionError("o RuntimeError do watcher deveria ter propagado do lifespan")
    except RuntimeError as err:
        assert "cancelamento do caller" in str(err)
    finally:
        assert calls == ["watcher", "recorder", "pool"]
        for attr in ("settings", "pool"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)


def test_lifespan_closes_the_config_watcher_before_the_pool(monkeypatch, tmp_path):
    """Ordem no shutdown: vigia -> recorder -> pool, senao um `apply` em voo
    no vigia pode chamar `pool.update()` num pool ja fechado."""
    from app.core.upstream import UpstreamPool

    monkeypatch.setenv("TURSO_DATABASE_URL", f"sqlite:///{tmp_path / 'stats.db'}")
    calls: list[str] = []
    original_pool_aclose = UpstreamPool.aclose

    async def spy_pool_aclose(self):
        calls.append("pool")
        await original_pool_aclose(self)

    monkeypatch.setattr(UpstreamPool, "aclose", spy_pool_aclose)
    try:
        with TestClient(app):
            original_watcher_aclose = app.state.config_watcher.aclose

            async def spy_watcher_aclose():
                calls.append("watcher")
                await original_watcher_aclose()

            app.state.config_watcher.aclose = spy_watcher_aclose
        assert calls == ["watcher", "pool"]
    finally:
        for attr in ("settings", "pool"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)
