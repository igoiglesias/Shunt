"""O contrato do gravador: nunca bloquear, nunca levantar, nunca morrer.

Cada teste aqui existe por causa da restricao que manda no epico inteiro -- o
banco nao pode custar latencia a API. Sao eles que quebram se alguem trocar a
fila por uma escrita direta.
"""

import asyncio
import time
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.stats.models import ApiToken, RequestEvent
from app.stats.recorder import Recorder


def engine_for(make_engine, tmp_path):
    return make_engine(f"sqlite+pysqlite:///{tmp_path / 'stats.db'}")


def event(**over):
    row = {
        "request_id": "r1",
        "started_at": datetime(2026, 9, 18, 22, 0, tzinfo=UTC),
        "route": "/v1/messages",
        "dialect": "anthropic",
        "stream": False,
        "requested_model": "claude-haiku-4-5",
        "rule": "family",
        "matched": "haiku",
        "provider": "groq",
        "candidate_model": "openai/gpt-oss-120b",
        "status": 200,
        "input_tokens": 10,
        "output_tokens": 5,
        "duration_ms": 100,
    }
    row.update(over)
    return row


def count(engine):
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(RequestEvent))


async def test_a_full_queue_drops_instead_of_waiting(make_engine, tmp_path):
    """Fila cheia significa banco lento. Perder estatistica, nunca segurar requisicao."""
    recorder = Recorder(engine_for(make_engine, tmp_path), max_queue=2)
    started = time.perf_counter()
    for i in range(50):
        recorder.record(event(request_id=f"r{i}"))
    elapsed = time.perf_counter() - started
    assert elapsed < 0.05, f"record() custou {elapsed:.4f}s para 50 eventos"
    assert recorder.dropped == 48


async def test_record_never_raises_even_with_a_dead_engine(make_engine):
    engine = make_engine("sqlite+pysqlite:////nao/existe/stats.db", create_tables=False)
    recorder = Recorder(engine, max_queue=10)
    recorder.record(event())
    assert recorder.dropped == 0


async def test_the_worker_writes_a_batch_in_a_single_commit(make_engine, tmp_path):
    engine = engine_for(make_engine, tmp_path)
    recorder = Recorder(engine, max_queue=1000, batch_size=200, interval=0.01)
    await recorder.start()
    for i in range(200):
        recorder.record(event(request_id=f"r{i}"))
    await recorder.aclose()
    assert count(engine) == 200
    assert recorder.commits == 1, f"gravou em {recorder.commits} commits"


async def test_closing_drains_what_is_still_queued(make_engine, tmp_path):
    engine = engine_for(make_engine, tmp_path)
    recorder = Recorder(engine, max_queue=1000, batch_size=500, interval=10.0)
    await recorder.start()
    for i in range(7):
        recorder.record(event(request_id=f"r{i}"))
    await recorder.aclose()
    assert count(engine) == 7


async def test_a_write_that_explodes_does_not_kill_the_worker(make_engine, tmp_path):
    """Turso fora do ar deixa o painel velho. O proximo lote ainda tem de entrar."""
    engine = engine_for(make_engine, tmp_path)
    recorder = Recorder(engine, max_queue=1000, batch_size=1, interval=0.01)
    await recorder.start()
    recorder.record(event(request_id="quebrado", coluna_que_nao_existe=1))
    await asyncio.sleep(0.1)
    recorder.record(event(request_id="depois"))
    await recorder.aclose()
    assert recorder.failures >= 1
    with Session(engine) as session:
        stored = session.scalars(select(RequestEvent.request_id)).all()
    assert "depois" in stored


async def test_no_engine_means_the_recorder_is_a_no_op(make_engine, tmp_path):
    recorder = Recorder(None)
    await recorder.start()
    recorder.record(event())
    await recorder.aclose()
    assert recorder.enabled is False


@pytest.mark.parametrize("field", ["dropped", "failures", "commits", "queued"])
def test_the_counters_the_panel_shows_exist_from_the_start(field):
    assert getattr(Recorder(None), field) == 0


async def test_the_final_drain_gives_up_instead_of_hanging_the_shutdown(make_engine, tmp_path, monkeypatch, caplog):
    """Desligamento nao pode ficar pendurado esperando um banco que nao responde."""
    engine = engine_for(make_engine, tmp_path)
    monkeypatch.setattr("app.stats.recorder.DRAIN_TIMEOUT", 0.01)
    recorder = Recorder(engine, max_queue=100, batch_size=1, interval=10.0)
    await recorder.start()

    def slow(batch):
        time.sleep(0.2)

    monkeypatch.setattr(recorder, "_write", slow)
    for i in range(5):
        recorder.record(event(request_id=f"r{i}"))
    await recorder.aclose()
    assert "dreno final expirou" in caplog.text


def test_writing_with_no_engine_does_nothing(make_engine, tmp_path):
    recorder = Recorder(None)
    recorder._write([event()])
    assert recorder.commits == 0
    assert recorder.failures == 0


async def test_the_panel_bus_gets_the_event_even_with_no_database(make_engine):
    """Olhar o proxy correndo nao pode exigir Turso configurado."""
    recorder = Recorder(None)
    queue = recorder.subscribe()
    recorder.record(event(request_id="ao-vivo"))
    assert queue.get_nowait()["request_id"] == "ao-vivo"


async def test_a_slow_panel_reader_never_holds_up_the_recorder(make_engine, tmp_path):
    recorder = Recorder(engine_for(make_engine, tmp_path), max_queue=10_000)
    queue = recorder.subscribe()
    started = time.perf_counter()
    for i in range(500):
        recorder.record(event(request_id=f"r{i}"))
    elapsed = time.perf_counter() - started
    assert elapsed < 0.2, f"record() custou {elapsed:.4f}s com um leitor parado"
    assert queue.qsize() == 100, "a fila do assinante tem teto"
    assert recorder.dropped == 0, "o leitor lento nao pode custar evento gravado"


async def test_unsubscribing_stops_the_delivery(make_engine):
    recorder = Recorder(None)
    queue = recorder.subscribe()
    recorder.unsubscribe(queue)
    recorder.unsubscribe(queue)  # idempotente: o SSE fecha uma vez so, mas fecha sempre
    recorder.record(event())
    assert queue.empty()


async def test_a_database_that_failed_at_boot_is_reopened_later(make_engine, tmp_path):
    """Medido num `make prod` com oito workers: tres subiram sem engine.

    Ficariam assim para sempre, e o painel respondia "sem banco" em um terco das
    cargas enquanto os outros gravavam.
    """
    engine = engine_for(make_engine, tmp_path)
    tentativas = []

    def volta():
        tentativas.append(1)
        return engine if len(tentativas) > 1 else None

    recorder = Recorder(None, reconnect=volta, reconnect_seconds=0, interval=0.01)
    assert recorder.enabled is False
    assert recorder.configured is True, "banco declarado nao e banco ausente"
    await recorder.start()
    recorder.record(event(request_id="antes"))
    await asyncio.sleep(0.15)
    recorder.record(event(request_id="depois"))
    await recorder.aclose()
    assert recorder.reconnects == 1
    assert recorder.enabled is True
    assert count(engine) >= 1, "o evento seguinte a reconexao tinha de entrar"


async def test_a_reconnection_that_explodes_is_counted_and_retried(make_engine, tmp_path, caplog):
    def explode():
        raise RuntimeError("banco ainda fora")

    recorder = Recorder(None, reconnect=explode, reconnect_seconds=0, interval=0.01)
    await recorder.start()
    await asyncio.sleep(0.1)
    await recorder.aclose()
    assert recorder.enabled is False
    assert "reconexao falhou" in caplog.text


async def test_without_a_reconnect_the_recorder_stays_a_no_op():
    recorder = Recorder(None)
    assert recorder.configured is False
    await recorder.start()
    recorder.record(event())
    await recorder.aclose()
    assert recorder.reconnects == 0


async def test_reconnection_is_attempted_once_per_interval(make_engine, tmp_path):
    """Banco fora do ar por uma hora nao pode virar uma tentativa por segundo."""
    tentativas = []
    def falha():
        tentativas.append(1)

    recorder = Recorder(None, reconnect=falha, reconnect_seconds=60, interval=0.01)
    await recorder.start()
    await asyncio.sleep(0.12)
    await recorder.aclose()
    assert len(tentativas) == 1, f"tentou {len(tentativas)} vezes em 0,12 s"


async def test_an_open_database_is_never_reopened(make_engine, tmp_path):
    tentativas = []
    recorder = Recorder(
        engine_for(make_engine, tmp_path),
        reconnect=lambda: tentativas.append(1),
        reconnect_seconds=0,
        interval=0.01,
    )
    await recorder.start()
    recorder.record(event())
    await asyncio.sleep(0.1)
    await recorder.aclose()
    assert tentativas == [], "reabriu um banco que ja estava aberto"


async def test_a_recorder_with_no_reconnect_never_tries_to_reopen():
    """Sem banco declarado nao ha o que reabrir: o caminho sai na primeira linha."""
    recorder = Recorder(None)
    recorder._try_reconnect()
    assert recorder.reconnects == 0
    assert recorder.enabled is False


async def test_token_touch_is_written_by_the_worker_not_the_caller(make_engine, tmp_path):
    """`last_used_at` e gravado pelo worker, nunca no caminho da requisicao."""
    engine = engine_for(make_engine, tmp_path)
    with Session(engine) as s:
        s.add(ApiToken(id=3, name="t", token_hash="h" * 64))
        s.commit()
    recorder = Recorder(engine, interval=0.01)
    await recorder.start()
    recorder.touch_token(3)
    with Session(engine) as s:
        assert s.get(ApiToken, 3).last_used_at is None, "o chamador nao pode ter gravado"
    await recorder.aclose()
    with Session(engine) as s:
        assert s.get(ApiToken, 3).last_used_at is not None
    assert count(engine) == 0, "um toque de token nao e um RequestEvent"


async def test_a_token_touch_never_reaches_the_panel_bus(make_engine):
    """Evento interno do gravador nao vira linha ao vivo no painel."""
    recorder = Recorder(None)
    queue = recorder.subscribe()
    recorder.touch_token(1)
    assert queue.empty()


async def test_store_writes_the_row_without_touching_the_panel_bus(make_engine, tmp_path):
    """`store` e a escrita do relay: vai para o banco e PARA.

    O relay nao existe no painel ao vivo (SSE) nem em `recent`: ele e trafego
    de passagem, e publicar no barramento faria cada repasse aparecer na tela
    como se fosse uma requisicao de modelo.
    """
    engine = engine_for(make_engine, tmp_path)
    recorder = Recorder(engine, max_queue=1000, interval=10.0)
    await recorder.start()
    queue = recorder.subscribe()
    recorder.store(event(request_id="relay-1", kind="relay"))
    await recorder.aclose()
    assert count(engine) == 1
    with Session(engine) as session:
        guardado = session.scalar(select(RequestEvent.kind).where(RequestEvent.request_id == "relay-1"))
    assert guardado == "relay"
    assert queue.empty(), "o relay nao pode aparecer no barramento do painel"


# --------------------------------------------------------------------------
# B2: uma linha que viola uma constraint nao derruba o lote inteiro
# --------------------------------------------------------------------------


async def test_one_invalid_line_does_not_lose_the_valid_ones(make_engine, tmp_path):
    """Lote misto: so a linha que violou a constraint e descartada.

    Hoje o lote inteiro e um `commit` so, e uma `IntegrityError` em qualquer
    linha aborta TODAS -- medido em sqlite3 puro, `executemany` com duas linhas
    sendo uma invalida nao insere nem a valida. E defeito em profundidade: o
    migrador de B1 elimina o caso mais comum (NOT NULL legado), mas qualquer
    outra constraint violada por uma linha volta a perder a vizinhanca.
    """
    engine = engine_for(make_engine, tmp_path)
    recorder = Recorder(engine, max_queue=100, batch_size=100, interval=10.0)
    await recorder.start()
    recorder.record(event(request_id="boa-1"))
    # `started_at` e NOT NULL: esta linha viola a constraint de proposito.
    recorder.record(event(request_id="ruim", started_at=None))
    recorder.record(event(request_id="boa-2"))

    await recorder.aclose()

    with Session(engine) as session:
        ids = {
            row.request_id
            for row in session.scalars(select(RequestEvent))
        }

    assert ids == {"boa-1", "boa-2"}, f"linhas validas perdidas: {ids}"
    assert recorder.failures >= 1, "a linha invalida tem que ser contada"
    assert recorder.commits >= 1
