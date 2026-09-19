"""O schema do armazem de estatisticas, conferido contra um SQLite de arquivo."""

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.stats.models import RequestEvent


def engine_for(make_engine, tmp_path):
    return make_engine(f"sqlite+pysqlite:///{tmp_path / 'stats.db'}")


def sample(**over):
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
        "error_type": None,
        "input_tokens": 124,
        "output_tokens": 39,
        "ttft_ms": None,
        "duration_ms": 371,
        "attempts": [],
        "fell_back": False,
        "tools_offered": ["read"],
        "tools_called": ["read"],
        "thinking_blocks": 1,
    }
    row.update(over)
    return RequestEvent(**row)


def test_an_event_round_trips_with_every_column(make_engine, tmp_path):
    engine = engine_for(make_engine, tmp_path)
    with Session(engine) as session:
        session.add(sample())
        session.commit()
    with Session(engine) as session:
        stored = session.scalars(select(RequestEvent)).one()
    assert stored.candidate_model == "openai/gpt-oss-120b"
    assert stored.tools_called == ["read"]
    assert stored.attempts == []
    assert stored.duration_ms == 371
    assert stored.id is not None


def test_a_refused_request_stores_with_no_candidate(make_engine, tmp_path):
    """Requisicao recusada e justamente a que se quer contar, entao nada de NOT NULL ali."""
    engine = engine_for(make_engine, tmp_path)
    with Session(engine) as session:
        session.add(
            sample(
                request_id="r2",
                provider=None,
                candidate_model=None,
                status=400,
                error_type="invalid_request_error",
                attempts=["free: credential OPENROUTER_API_KEY is not set"],
            )
        )
        session.commit()
    with Session(engine) as session:
        stored = session.scalars(select(RequestEvent).where(RequestEvent.status == 400)).one()
    assert stored.provider is None
    assert stored.attempts == ["free: credential OPENROUTER_API_KEY is not set"]


def test_events_insert_in_one_batch(make_engine, tmp_path):
    engine = engine_for(make_engine, tmp_path)
    with Session(engine) as session:
        session.add_all([sample(request_id=f"r{i}") for i in range(200)])
        session.commit()
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(RequestEvent)) == 200


def test_the_time_axis_and_the_two_grouping_axes_are_indexed(make_engine, tmp_path):
    """Toda consulta do painel filtra por tempo e agrupa por provedor ou modelo."""
    indexed = {tuple(index.expressions[0].name for _ in [0]) for index in RequestEvent.__table__.indexes}
    columns = {name for (name,) in indexed}
    assert "started_at" in columns
    assert {"provider", "candidate_model"} & columns
