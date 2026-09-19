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
