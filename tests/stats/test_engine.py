"""A engine do armazem: opcional no boot, e sem token inventado."""

from sqlalchemy import inspect

from app.stats.engine import build_engine, database_url


def test_no_url_means_no_database_and_no_exception(monkeypatch):
    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
    assert database_url() is None
    assert build_engine() is None


def test_the_token_is_appended_as_a_query_parameter(monkeypatch):
    monkeypatch.setenv("TURSO_DATABASE_URL", "sqlite+libsql://exemplo.turso.io")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "token-secreto")
    assert database_url() == "sqlite+libsql://exemplo.turso.io?authToken=token-secreto"


def test_a_url_that_already_carries_the_token_is_left_alone(monkeypatch):
    monkeypatch.setenv("TURSO_DATABASE_URL", "sqlite+libsql://exemplo.turso.io?authToken=ja-tem")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "outro")
    assert database_url().count("authToken=") == 1


def test_a_reachable_url_comes_back_with_the_table_already_created(tmp_path):
    engine = build_engine(f"sqlite+pysqlite:///{tmp_path / 'stats.db'}")
    assert engine is not None
    try:
        assert "request_events" in inspect(engine).get_table_names()
    finally:
        engine.dispose()


def test_a_broken_url_disables_persistence_instead_of_killing_the_boot(caplog):
    """Banco fora do ar no boot nao pode impedir o proxy de servir requisicao."""
    assert build_engine("sqlite+pysqlite:////caminho/que/nao/existe/stats.db") is None
    assert "banco indisponivel" in caplog.text
