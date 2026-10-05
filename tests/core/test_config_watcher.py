"""O vigia de versao: recarrega quando `config_versions` cresce, e so entao."""

import asyncio
import logging
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.core.config_watcher import UNKNOWN_VERSION, ConfigWatcher
from app.core.upstream import UpstreamPool
from app.routers.admin_config import apply_settings, create_config_version
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


# Intervalo longo de proposito: quase todo teste aqui chama `check_once()`
# manualmente. Com um intervalo curto o laco de fundo (criado por `start()`)
# corre em paralelo e pode entrar no mesmo `check_once()` que o teste chama
# na mao -- os dois veem `version != seen(None)` antes de qualquer um gravar
# `seen`, e os dois recarregam (`reloads` vira 2 em vez de 1). Medido: 15
# falhas em 40 rodadas com intervalo 0.01 (reviewer), 0 falhas em 60 rodadas
# com intervalo alto. So o teste que exercita o laco em si usa intervalo
# curto (`ConfigWatcher(..., interval=0.01)` direto, sem este helper).
_INERT_INTERVAL = 3600.0


def _watcher(engine, applied, url: str = "sqlite:///irrelevante.db") -> ConfigWatcher:
    return ConfigWatcher(engine_of=lambda: engine, apply=applied, url=url, interval=_INERT_INTERVAL)


async def _boot(watcher: ConfigWatcher) -> None:
    """Espelha a ordem real do lifespan (app/main.py): versao inicial lida
    ANTES do laco de fundo comecar."""
    await watcher.read_initial_version()
    await watcher.start()


def test_the_unknown_version_sentinel_has_a_stable_repr():
    """O sentinela `seen` de antes da primeira leitura tem `repr` estavel: o
    que aparece em log/erro quando o ciclo mostra a versao atual."""
    assert repr(UNKNOWN_VERSION) == "UNKNOWN_VERSION"


async def test_a_new_version_reloads_the_catalog_and_applies_it(tmp_path):
    engine = _engine(tmp_path)
    applied = _Applied()
    watcher = _watcher(engine, applied)
    await _boot(watcher)
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
    await _boot(watcher)
    assert watcher.seen == version  # versao inicial lida antes do laco comecar
    assert await watcher.check_once() is False
    assert applied.settings == []
    await watcher.aclose()


async def test_a_failed_query_keeps_the_catalog_and_the_next_cycle_tries_again(
    tmp_path, monkeypatch, caplog
):
    engine = _engine(tmp_path)
    applied = _Applied()
    watcher = _watcher(engine, applied)
    await _boot(watcher)
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
    await _boot(watcher)
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
    """`read_initial_version()` tambem falha (banco inalcancavel): `seen`
    fica em `UNKNOWN_VERSION`, nao em `None` -- distincao que importa para o
    caso de recuperacao (ver `test_after_the_initial_read_finds_no_engine_...`
    abaixo)."""
    engine = _engine(tmp_path)
    applied = _Applied()
    touched: list[str] = []
    monkeypatch.setattr(
        "app.core.config_watcher.reachable", lambda url: touched.append(url) or False
    )
    watcher = _watcher(engine, applied, url="libsql://banco.turso.io")
    await watcher.read_initial_version()
    assert touched == ["libsql://banco.turso.io"]
    assert watcher.seen is UNKNOWN_VERSION
    await watcher.start()
    assert await watcher.check_once() is False
    assert watcher.failures == 1
    assert applied.settings == []
    await watcher.aclose()


async def test_an_engine_of_that_raises_counts_as_a_failure_and_retries_next_cycle(tmp_path):
    """`engine_of()` roda dentro do `try` de `check_once()`: se o gravador
    estiver no meio de reabrir o banco e `engine_of()` levantar, o ciclo
    conta como falha e tenta de novo no proximo -- em vez de deixar a
    excecao subir e matar o laco de fundo em silencio, contradizendo a
    docstring de `check_once` ("Nunca levanta")."""
    engine = _engine(tmp_path)
    applied = _Applied()
    watcher = _watcher(engine, applied)
    await _boot(watcher)
    assert watcher.seen is None
    version = _add_provider(engine, "p1")

    calls = {"n": 0}

    def flaky_engine_of():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("banco reabrindo")
        return engine

    watcher._engine_of = flaky_engine_of
    assert await watcher.check_once() is False
    assert watcher.failures == 1
    assert watcher.seen is None
    assert await watcher.check_once() is True
    assert watcher.seen == version
    assert "p1" in applied.settings[-1].providers
    await watcher.aclose()


async def test_without_an_engine_the_cycle_does_nothing_and_counts_no_failure(tmp_path):
    applied = _Applied()
    watcher = ConfigWatcher(
        engine_of=lambda: None, apply=applied, url="sqlite:///x.db", interval=_INERT_INTERVAL
    )
    await _boot(watcher)
    assert await watcher.check_once() is False
    assert applied.settings == []
    assert watcher.failures == 0
    await watcher.aclose()


async def test_after_the_initial_read_finds_no_engine_the_first_successful_cycle_still_reloads(
    tmp_path,
):
    """I-2 (reviewer): banco inalcancavel no boot -- `engine_of()` devolve
    `None` -- deixa `seen` em `UNKNOWN_VERSION`. Quando o gravador reconecta
    e `engine_of()` passa a devolver a engine real, o PRIMEIRO ciclo bem
    sucedido tem de recarregar mesmo que a versao real seja `None` (banco
    alcancavel mas ainda sem nenhuma edicao) -- se `seen` tivesse ficado em
    `None` (o default antigo) em vez de `UNKNOWN_VERSION`, um banco vazio
    apos a reconexao nunca dispararia o catch-up (medido pelo reviewer:
    `check_once apos reconexao: False applied: 0 seen: None`), e o worker
    ficaria com o catalogo desatualizado ate a PROXIMA edicao real."""
    engine_holder: dict[str, object] = {"engine": None}
    applied = _Applied()
    watcher = ConfigWatcher(
        engine_of=lambda: engine_holder["engine"],
        apply=applied,
        url="sqlite:///irrelevante.db",
        interval=_INERT_INTERVAL,
    )
    await watcher.read_initial_version()
    assert watcher.seen is UNKNOWN_VERSION
    assert await watcher.check_once() is False  # ainda sem engine: nada a fazer
    engine_holder["engine"] = _engine(tmp_path)  # "reconexao": banco alcancavel, sem edicoes
    assert await watcher.check_once() is True  # catch-up: version (None) != UNKNOWN_VERSION
    assert watcher.seen is None
    assert applied.settings  # o catalogo (vazio) foi de fato aplicado
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
    await _boot(watcher)
    first_task = watcher._task
    await watcher.start()
    assert watcher._task is first_task
    await watcher.aclose()


async def test_a_cancellation_during_the_initial_read_is_not_swallowed(tmp_path):
    """O `except` de `read_initial_version()` cobre so `Exception`: um
    cancelamento real (encerramento do processo no meio do boot) tem de
    propagar, nao terminar a leitura como se nada tivesse acontecido."""
    import time

    engine = _engine(tmp_path)
    applied = _Applied()
    watcher = _watcher(engine, applied)

    def slow_read(engine_):
        time.sleep(0.2)

    watcher._read_version = slow_read
    task = asyncio.ensure_future(watcher.read_initial_version())
    await asyncio.sleep(0.02)
    task.cancel()
    try:
        await task
        raised = False
    except asyncio.CancelledError:
        raised = True
    assert raised
    assert watcher._task is None  # nao chegou a criar o laco de fundo


async def test_a_cancellation_during_check_once_is_not_swallowed(tmp_path):
    """O `except` de `check_once()` cobre so `Exception`: um cancelamento
    real durante o ciclo (nao um erro de banco) tem de propagar, nao virar
    uma falha comum contada e engolida. `asyncio.timeout` limita a espera:
    se o mutante (`except BaseException`) engolir o cancelamento, a task
    termina sozinha sem levantar nada e o teste falha rapido no assert, sem
    travar a suite."""
    import time

    engine = _engine(tmp_path)
    applied = _Applied()
    watcher = _watcher(engine, applied)
    _add_provider(engine, "p1")

    def slow_read(engine_):
        time.sleep(0.2)

    watcher._read_version = slow_read
    task = asyncio.ensure_future(watcher.check_once())
    await asyncio.sleep(0.02)
    task.cancel()
    async with asyncio.timeout(2):
        try:
            await task
            raised = False
        except asyncio.CancelledError:
            raised = True
    assert raised
    assert watcher.failures == 0  # propagou; nao contou como falha comum


async def test_aclose_while_parked_inside_check_once_does_not_hang(tmp_path):
    """Se o laco de fundo estiver preso dentro de `check_once()` quando
    `aclose()` pede o cancelamento, `aclose()` tem de terminar mesmo assim.

    `aclose()` espera a MESMA task que ele cancelou (`await task` dentro do
    seu proprio `except CancelledError: pass`); um `asyncio.timeout` em volta
    da chamada nao serve de rede de seguranca aqui -- ele cancela a task de
    TESTE, que esta parada bem naquele `await task`, e `aclose()` absorve
    esse cancelamento tambem (e' o mesmo ponto de suspensao), entao o
    `async with` sai sem excecao e nada denuncia o timeout. Por isso o
    `aclose()` roda como uma task PROPRIA e `asyncio.wait(..., timeout=...)`
    so' observa se ela terminou, sem cancelar nada: se o mutante engolir o
    cancelamento em `check_once()`, `_run` nunca sai do `while True`, a task
    de `aclose()` fica em `pending`, e o assert falha rapido em vez de a
    suite travar."""
    import time

    engine = _engine(tmp_path)
    applied = _Applied()
    watcher = ConfigWatcher(
        engine_of=lambda: engine, apply=applied, url="sqlite:///irrelevante.db", interval=0.01
    )
    _add_provider(engine, "p1")

    def slow_read(engine_):
        time.sleep(0.2)

    watcher._read_version = slow_read
    await watcher.start()  # sem leitura inicial: so' cria o laco de fundo
    await asyncio.sleep(0.02)  # deixa o laco de fundo entrar no primeiro check_once, preso no slow_read
    aclose_task = asyncio.ensure_future(watcher.aclose())
    done, pending = await asyncio.wait([aclose_task], timeout=2)
    if pending:
        for t in pending:
            t.cancel()
        raise AssertionError("aclose() nao terminou em 2s: laco preso em check_once")
    assert aclose_task in done
    assert watcher._task is None


async def test_aclose_propagates_cancellation_aimed_at_its_caller(tmp_path):
    """`aclose()` so' pode engolir o cancelamento da task de fundo que ele
    mesmo pediu (`task.cancel()`). Se quem CHAMOU `aclose()` for cancelado
    enquanto `aclose()` esta parado em `await task` -- caso real:
    `asyncio.wait_for(watcher.aclose(), timeout)` estourando o timeout --
    esse segundo cancelamento tem de propagar. Sob o codigo antigo
    (`except CancelledError: pass` sem distinguir de onde veio) ele
    desaparecia e `wait_for` nunca via o timeout: o caller ficava pendurado
    para sempre.

    Para criar uma janela real (medido: cancelar uma task presa em
    `asyncio.sleep` ou em `asyncio.to_thread` resolve quase instantaneo --
    rapido demais para um `asyncio.sleep(0.01)` externo alcancar o `await
    task` de `aclose()` ainda pendurado), a task de fundo e um coroutine
    proprio que, ao ser cancelado, entra num `except CancelledError` e so'
    ENTAO faz um `await asyncio.sleep(0.3)` antes de re-levantar -- limpeza
    lenta de verdade, que mantem `aclose()` genuinamente parado em `await
    task` por uma janela grande o bastante para o cancelamento externo
    chegar dentro dela (medido com um probe isolado antes deste teste:
    `cancelling()=1` dentro do `except` de `aclose()` nesse cenario).

    Roda `aclose()` como task PROPRIA (nao com `asyncio.timeout` em volta de
    um `await` direto -- isso cancelaria a task de TESTE, nao a de
    `aclose()`, e o bug antigo so aparece quando quem e cancelado e a
    propria chamada de `aclose()`) e usa `asyncio.wait(..., timeout=...)`
    para nao travar a suite se o mutante voltar."""

    async def _stubborn_background() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.3)  # limpeza lenta: mantem `aclose()` parado
            raise

    engine = _engine(tmp_path)
    applied = _Applied()
    watcher = _watcher(engine, applied)
    watcher._task = asyncio.ensure_future(_stubborn_background())

    aclose_task = asyncio.ensure_future(watcher.aclose())
    await asyncio.sleep(0.1)  # aclose() ja pediu task.cancel() e esta parado em `await task`
    aclose_task.cancel()  # cancelamento mirado no CALLER de aclose(), nao na task de fundo

    done, pending = await asyncio.wait([aclose_task], timeout=2)
    if pending:
        for t in pending:
            t.cancel()
        raise AssertionError("aclose() nao terminou: mutante pode ter travado a task")
    assert aclose_task in done
    assert aclose_task.cancelled(), "aclose() engoliu o cancelamento do proprio caller"


async def test_an_edit_made_by_one_worker_reaches_another_within_one_cycle(tmp_path):
    """Dois estados sobre o mesmo arquivo: A grava como o admin grava, o laco
    do vigia de B recarrega e o pool de B ja serve o provedor novo."""
    engine_a = _engine(tmp_path)
    url = f"sqlite:///{tmp_path / 'watch.db'}"
    engine_b = create_engine(url)
    state_b = SimpleNamespace(settings=Settings(providers={}, models={}, routes=[]), pool=None)
    state_b.pool = UpstreamPool(state_b.settings)
    # O `apply_settings` de producao, o mesmo que o lifespan liga ao vigia:
    # uma copia aqui provaria o criterio com codigo que nao roda no servidor.
    apply_b = partial(apply_settings, SimpleNamespace(state=state_b))

    watcher = ConfigWatcher(engine_of=lambda: engine_b, apply=apply_b, url=url, interval=0.01)
    await watcher.read_initial_version()  # como o lifespan real: antes do laco comecar
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
