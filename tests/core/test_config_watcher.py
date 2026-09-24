"""O vigia de versao: recarrega quando `config_versions` cresce, e so entao."""

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.core.config_watcher import ConfigWatcher
from app.core.upstream import UpstreamPool
from app.routers.admin_config import create_config_version
from app.stats.models import Base, Provider


def _engine(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'watch.db'}")
    Base.metadata.create_all(engine)
    return engine


def _add_provider(engine, name: str) -> int:
    """Escreve como o admin escreve: linha no catalogo + versao nova."""
    with Session(engine) as session:
        session.add(Provider(name=name, base_url=f"http://{name}.test", protocol="openai"))
        session.commit()
        return create_config_version(session).id


class _Applied:
    def __init__(self) -> None:
        self.settings: list[Settings] = []

    async def __call__(self, settings: Settings) -> None:
        self.settings.append(settings)


def _watcher(engine, applied, url: str = "sqlite:///irrelevante.db") -> ConfigWatcher:
    return ConfigWatcher(engine_of=lambda: engine, apply=applied, url=url, interval=0.01)


async def test_a_new_version_reloads_the_catalog_and_applies_it(tmp_path):
    engine = _engine(tmp_path)
    applied = _Applied()
    watcher = _watcher(engine, applied)
    await watcher.start()
    assert watcher.seen is None  # banco novo: o seed nao cria versao
    assert await watcher.check_once() is False  # None == None: nada a fazer
    version = _add_provider(engine, "p1")
    assert await watcher.check_once() is True
    assert watcher.seen == version
    assert watcher.reloads == 1
    assert "p1" in applied.settings[-1].providers
    await watcher.aclose()


async def test_the_same_version_reloads_nothing(tmp_path):
    engine = _engine(tmp_path)
    version = _add_provider(engine, "p1")
    applied = _Applied()
    watcher = _watcher(engine, applied)
    await watcher.start()
    assert watcher.seen == version  # versao inicial lida no boot
    assert await watcher.check_once() is False
    assert applied.settings == []
    await watcher.aclose()


async def test_a_failed_query_keeps_the_catalog_and_the_next_cycle_tries_again(
    tmp_path, monkeypatch, caplog
):
    engine = _engine(tmp_path)
    applied = _Applied()
    watcher = _watcher(engine, applied)
    await watcher.start()
    _add_provider(engine, "p1")
    real = watcher._read_version
    calls = {"n": 0}

    def flaky(engine_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("banco caiu")
        return real(engine_)

    monkeypatch.setattr(watcher, "_read_version", flaky)
    # `configure_logging` desliga a propagacao do logger `shunt`
    # (app/core/observability.py:182); sem religar, o caplog nao ve nada
    # quando outro teste ja subiu o lifespan.
    monkeypatch.setattr(logging.getLogger("shunt"), "propagate", True)
    with caplog.at_level("WARNING", logger="shunt"):
        assert await watcher.check_once() is False
    assert applied.settings == []
    assert watcher.failures == 1
    assert "RuntimeError: banco caiu" in caplog.text
    assert await watcher.check_once() is True
    assert "p1" in applied.settings[-1].providers
    await watcher.aclose()


async def test_an_apply_that_fails_keeps_seen_unchanged_so_the_next_cycle_retries(tmp_path):
    engine = _engine(tmp_path)
    applied = _Applied()
    watcher = _watcher(engine, applied)
    await watcher.start()
    version = _add_provider(engine, "p1")

    async def explode(settings):
        raise RuntimeError("apply falhou")

    watcher._apply = explode
    assert await watcher.check_once() is False
    assert watcher.seen is None  # nao avanca: o catalogo velho continua valendo
    watcher._apply = applied
    assert await watcher.check_once() is True
    assert watcher.seen == version
    await watcher.aclose()


async def test_an_unreachable_remote_database_is_not_queried(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    applied = _Applied()
    touched: list[str] = []
    monkeypatch.setattr(
        "app.core.config_watcher.reachable", lambda url: touched.append(url) or False
    )
    watcher = _watcher(engine, applied, url="libsql://banco.turso.io")
    await watcher.start()
    assert touched == ["libsql://banco.turso.io"]
    assert watcher.seen is None
    assert await watcher.check_once() is False
    assert watcher.failures == 1
    assert applied.settings == []
    await watcher.aclose()


async def test_without_an_engine_the_cycle_does_nothing_and_counts_no_failure(tmp_path):
    applied = _Applied()
    watcher = ConfigWatcher(engine_of=lambda: None, apply=applied, url="sqlite:///x.db", interval=0.01)
    await watcher.start()
    assert await watcher.check_once() is False
    assert applied.settings == []
    assert watcher.failures == 0
    await watcher.aclose()


async def test_aclose_cancels_the_background_task(tmp_path):
    watcher = _watcher(_engine(tmp_path), _Applied())
    await watcher.start()
    task = watcher._task
    assert task is not None and not task.done()
    await watcher.aclose()
    assert task.done()
    assert watcher._task is None
    await watcher.aclose()  # idempotente


async def test_start_called_twice_keeps_the_first_background_task(tmp_path):
    """`start()` de novo nao pode duplicar o laco: chamar duas vezes tem de
    manter a mesma task, senao dois lacos concorrentes disputariam `seen`."""
    watcher = _watcher(_engine(tmp_path), _Applied())
    await watcher.start()
    first_task = watcher._task
    await watcher.start()
    assert watcher._task is first_task
    await watcher.aclose()


async def test_a_cancellation_during_the_initial_read_is_not_swallowed(tmp_path):
    """O `except` da leitura inicial cobre so `Exception`: um cancelamento
    real (encerramento do processo no meio do boot) tem de propagar, nao
    terminar `start()` como se nada tivesse acontecido."""
    import time

    engine = _engine(tmp_path)
    applied = _Applied()
    watcher = _watcher(engine, applied)

    def slow_read(engine_):
        time.sleep(0.2)

    watcher._read_version = slow_read
    task = asyncio.ensure_future(watcher.start())
    await asyncio.sleep(0.02)
    task.cancel()
    try:
        await task
        raised = False
    except asyncio.CancelledError:
        raised = True
    assert raised
    assert watcher._task is None  # nao chegou a criar o laco de fundo


async def test_an_edit_made_by_one_worker_reaches_another_within_one_cycle(tmp_path):
    """Dois estados sobre o mesmo arquivo: A grava como o admin grava, o laco
    do vigia de B recarrega e o pool de B ja serve o provedor novo."""
    engine_a = _engine(tmp_path)
    url = f"sqlite:///{tmp_path / 'watch.db'}"
    engine_b = create_engine(url)
    state_b = SimpleNamespace(settings=Settings(providers={}, models={}, routes=[]), pool=None)
    state_b.pool = UpstreamPool(state_b.settings)

    async def apply_b(settings: Settings) -> None:
        state_b.settings = settings
        await state_b.pool.update(settings)

    watcher = ConfigWatcher(engine_of=lambda: engine_b, apply=apply_b, url=url, interval=0.01)
    await watcher.start()
    try:
        _add_provider(engine_a, "novo")
        for _ in range(200):
            await asyncio.sleep(0.01)
            if watcher.reloads:
                break
        assert watcher.reloads == 1
        assert "novo" in state_b.settings.providers
        assert str(state_b.pool.get("novo").base_url).rstrip("/") == "http://novo.test"
    finally:
        await watcher.aclose()
        await state_b.pool.aclose()
