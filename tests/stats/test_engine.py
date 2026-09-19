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


def test_an_unreachable_host_is_refused_before_the_driver_touches_it(caplog):
    """Medido: o driver do libsql BLOQUEIA segurando o GIL num host morto.

    Nem uma thread paralela imprime durante a tentativa, o que faria um Turso
    fora do ar congelar o proxy inteiro. Por isso um socket comum, com prazo,
    decide antes de o driver entrar em cena.
    """
    assert build_engine("sqlite+libsql://127.0.0.1:9/nada?authToken=x") is None
    assert "inalcancavel" in caplog.text


def test_a_file_database_never_pays_for_a_network_check(tmp_path):
    from app.stats.engine import reachable

    assert reachable(f"sqlite+pysqlite:///{tmp_path / 'a.db'}") is True


def test_a_reachable_host_passes_the_check():
    """Uma porta que aceita conexao passa; o teste sobe a sua propria."""
    import socket

    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        from app.stats.engine import reachable

        assert reachable(f"sqlite+libsql://127.0.0.1:{port}/x") is True


def test_the_default_port_is_used_when_the_url_names_none():
    from app.stats.engine import _target

    assert _target("sqlite+libsql://exemplo.turso.io/db") == ("exemplo.turso.io", 443)
    assert _target("sqlite+libsql://exemplo.turso.io:8080/db") == ("exemplo.turso.io", 8080)
    # Arquivo local: nao ha host, e tambem nao ha rede a testar.
    assert _target("sqlite+pysqlite:////tmp/a.db") is None
    assert _target("sqlite+pysqlite://") is None
    # Host com dialeto de arquivo continua sendo arquivo, e nao destino de rede.
    assert _target("sqlite+pysqlite://maquina/x.db") is None


def test_a_file_database_is_put_in_wal_with_a_lock_deadline(tmp_path):
    """`make prod` sobe um processo por nucleo; oito gravando no mesmo arquivo.

    Sem WAL o painel nao consegue ler enquanto um worker grava, e sem prazo de
    trava o lote morre com `database is locked` em vez de esperar sua vez.
    """
    from sqlalchemy import text

    engine = build_engine(f"sqlite+pysqlite:///{tmp_path / 'stats.db'}")
    assert engine is not None
    try:
        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA journal_mode")).scalar() == "wal"
            assert connection.execute(text("PRAGMA busy_timeout")).scalar() == 5000
    finally:
        engine.dispose()


def test_two_writers_on_the_same_file_do_not_collide(tmp_path):
    """Dois processos e o caso de `make prod`; duas conexoes bastam para medir."""
    from sqlalchemy.orm import Session

    from tests.stats.test_queries import row

    path = tmp_path / "stats.db"
    first = build_engine(f"sqlite+pysqlite:///{path}")
    second = build_engine(f"sqlite+pysqlite:///{path}")
    assert first is not None and second is not None
    try:
        with Session(first) as one, Session(second) as two:
            one.add(row(request_id="a"))
            one.commit()
            two.add(row(request_id="b"))
            two.commit()
        with Session(first) as reader:
            from sqlalchemy import func, select

            from app.stats.models import RequestEvent

            assert reader.scalar(select(func.count()).select_from(RequestEvent)) == 2
    finally:
        first.dispose()
        second.dispose()


def test_a_destination_that_refuses_the_pragmas_is_not_a_failure(caplog):
    """Turso nao e SQLite em arquivo: PRAGMA recusado e outro destino, nao falha."""
    import logging

    from app.stats.engine import apply_pragmas

    class Recusa:
        def cursor(self):
            raise RuntimeError("PRAGMA nao suportado")

    with caplog.at_level(logging.DEBUG, logger="shunt"):
        apply_pragmas(Recusa())
    assert "PRAGMA recusado" in caplog.text


def test_a_column_added_later_reaches_an_existing_database(tmp_path):
    """Medido: uma coluna nova quebrou TODA leitura da tabela antiga.

    `create_all` cria tabela que falta e ignora tabela que existe, inclusive
    quando ela esta com uma coluna a menos -- e o defeito so aparece no banco
    de quem ja estava usando.
    """
    import sqlite3

    from sqlalchemy import inspect

    from app.stats.engine import add_missing_columns

    caminho = tmp_path / "antigo.db"
    antigo = sqlite3.connect(caminho)
    antigo.execute(
        "CREATE TABLE request_bodies (request_id VARCHAR(64) PRIMARY KEY, prompt TEXT)"
    )
    antigo.commit()
    antigo.close()

    engine = build_engine(f"sqlite+pysqlite:///{caminho}")
    assert engine is not None
    try:
        colunas = {c["name"] for c in inspect(engine).get_columns("request_bodies")}
        assert {"answer", "request_json", "prompt_bytes", "truncated"} <= colunas
        # Rodar de novo nao acrescenta nada: a migracao e idempotente.
        assert add_missing_columns(engine) == []
    finally:
        engine.dispose()


def test_a_database_already_up_to_date_is_left_alone(tmp_path):
    from app.stats.engine import add_missing_columns

    engine = build_engine(f"sqlite+pysqlite:///{tmp_path / 'novo.db'}")
    assert engine is not None
    try:
        assert add_missing_columns(engine) == []
    finally:
        engine.dispose()
