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

from app.stats.models import RequestEvent
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
