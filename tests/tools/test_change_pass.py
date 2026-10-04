"""Tests do script operacional de troca de senha (scripts/change_pass.py)."""

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.security import hash_password, verify_password
from app.stats.models import Base, User
from scripts.change_pass import change_password, main


def _db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/pass.db")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        s.add(User(username="iglesias", password_hash=hash_password("velha")))
        s.commit()
    return engine


def test_change_password_replaces_the_hash_and_the_old_one_stops_working(tmp_path):
    engine = _db(tmp_path)
    report = change_password(str(engine.url), "iglesias", "nova-senha")
    assert report["changed"] is True
    with Session(engine) as s:
        user = s.execute(select(User).where(User.username == "iglesias")).scalar_one()
    assert verify_password("nova-senha", user.password_hash)
    assert not verify_password("velha", user.password_hash)


def test_change_password_reports_the_previous_hash_for_audit(tmp_path):
    engine = _db(tmp_path)
    report = change_password(str(engine.url), "iglesias", "nova-senha")
    assert report["previous_hash"] != user_hash(engine)


def user_hash(engine) -> str:
    with Session(engine) as s:
        return s.execute(select(User).where(User.username == "iglesias")).scalar_one().password_hash


def test_unknown_user_is_reported_not_silently_created(tmp_path):
    engine = _db(tmp_path)
    try:
        change_password(str(engine.url), "ninguem", "x")
    except LookupError as err:
        assert "ninguem" in str(err)
    else:
        raise AssertionError("usuario inexistente deve virar LookupError")
    with Session(engine) as s:
        nomes = [u.username for u in s.execute(select(User)).scalars().all()]
    assert nomes == ["iglesias"]


def test_empty_password_is_refused(tmp_path):
    engine = _db(tmp_path)
    try:
        change_password(str(engine.url), "iglesias", "")
    except ValueError:
        pass
    else:
        raise AssertionError("senha vazia deve ser recusada")
    with Session(engine) as s:
        user = s.execute(select(User).where(User.username == "iglesias")).scalar_one()
    assert verify_password("velha", user.password_hash)


def test_main_returns_nonzero_without_a_password(tmp_path, monkeypatch):
    engine = _db(tmp_path)
    monkeypatch.setenv("TURSO_DATABASE_URL", str(engine.url))
    monkeypatch.delenv("SHUNT_NEW_PASS", raising=False)
    rc = main(["iglesias"])
    assert rc != 0
    with Session(engine) as s:
        user = s.execute(select(User).where(User.username == "iglesias")).scalar_one()
    assert verify_password("velha", user.password_hash)


def test_main_exits_nonzero_for_unknown_user(tmp_path, monkeypatch):
    engine = _db(tmp_path)
    monkeypatch.setenv("TURSO_DATABASE_URL", str(engine.url))
    assert main(["ninguem", "x"]) != 0


def test_main_without_a_database_url_is_an_operational_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TURSO_DATABASE_URL", "")
    assert main(["iglesias", "x"]) != 0


def test_main_takes_the_password_from_the_env_when_omitted(tmp_path, monkeypatch):
    engine = _db(tmp_path)
    monkeypatch.setenv("TURSO_DATABASE_URL", str(engine.url))
    monkeypatch.setenv("SHUNT_NEW_PASS", "do-env")
    rc = main(["iglesias"])
    assert rc == 0
    with Session(engine) as s:
        user = s.execute(select(User).where(User.username == "iglesias")).scalar_one()
    assert verify_password("do-env", user.password_hash)
