"""Testes da fixture make_engine sem url: motor em memoria com StaticPool."""

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.stats.models import User


def test_make_engine_without_url_uses_static_pool_in_memory(make_engine):
    engine = make_engine()

    assert isinstance(engine.pool, StaticPool)
    assert str(engine.url).startswith("sqlite")


def test_make_engine_without_url_creates_the_tables(make_engine):
    engine = make_engine()

    tables = inspect(engine).get_table_names()
    assert "users" in tables


def test_make_engine_without_url_two_sessions_see_the_same_row(make_engine):
    engine = make_engine()

    with Session(engine) as session:
        session.add(User(username="fulano", password_hash="hash"))
        session.commit()

    with Session(engine) as session:
        stored = session.scalar(select(User).where(User.username == "fulano"))
        assert stored is not None
