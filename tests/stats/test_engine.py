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


def test_um_banco_em_uso_ganha_as_colunas_de_projeto_no_boot(tmp_path):
    """Quem ja usa o proxy tem `request_events` sem `project` nem `session_id`."""
    import sqlite3

    from sqlalchemy import inspect

    from app.stats.engine import add_missing_columns

    caminho = tmp_path / "em-uso.db"
    antigo = sqlite3.connect(caminho)
    antigo.execute(
        "CREATE TABLE request_events ("
        " id INTEGER PRIMARY KEY, request_id VARCHAR(64), started_at DATETIME,"
        " route VARCHAR(64), dialect VARCHAR(16), stream BOOLEAN, status INTEGER)"
    )
    antigo.execute(
        "INSERT INTO request_events (request_id, started_at, route, dialect, stream, status)"
        " VALUES ('antiga', '2026-09-01 10:00:00', '/v1/messages', 'anthropic', 0, 200)"
    )
    antigo.commit()
    antigo.close()

    engine = build_engine(f"sqlite+pysqlite:///{caminho}")
    assert engine is not None
    try:
        colunas = {c["name"] for c in inspect(engine).get_columns("request_events")}
        assert {"project", "session_id"} <= colunas
        assert add_missing_columns(engine) == []
        # A linha que ja estava la continua legivel, com o projeto vazio.
        with engine.connect() as conexao:
            from sqlalchemy import text as sql

            linha = conexao.execute(sql("SELECT request_id, project FROM request_events")).one()
        assert linha == ("antiga", None)
    finally:
        engine.dispose()


def test_um_banco_em_uso_ganha_a_coluna_kind_no_boot(tmp_path):
    """Banco criado antes do repasse: `request_events` sem `kind`.

    A coluna e NULLABLE de proposito: a migracao so compila o tipo, sem
    default, e as linhas antigas ficam NULL -- e NULL significa modelo. O
    filtro do painel por isso e `or_(kind IS NULL, kind != 'relay')`, nunca a
    comparacao sozinha (medido em SQLite: `k != 'relay'` exclui o NULL).
    """
    import sqlite3

    from sqlalchemy import inspect

    from app.stats.engine import add_missing_columns

    caminho = tmp_path / "sem-kind.db"
    antigo = sqlite3.connect(caminho)
    antigo.execute(
        "CREATE TABLE request_events ("
        " id INTEGER PRIMARY KEY, request_id VARCHAR(64), started_at DATETIME,"
        " route VARCHAR(64), dialect VARCHAR(16), stream BOOLEAN, status INTEGER)"
    )
    antigo.execute(
        "INSERT INTO request_events (request_id, started_at, route, dialect, stream, status)"
        " VALUES ('antiga', '2026-09-01 10:00:00', '/v1/messages', 'anthropic', 0, 200)"
    )
    antigo.commit()
    antigo.close()

    engine = build_engine(f"sqlite+pysqlite:///{caminho}")
    assert engine is not None
    try:
        colunas = {c["name"] for c in inspect(engine).get_columns("request_events")}
        assert "kind" in colunas
        assert add_missing_columns(engine) == []
        # A linha que ja estava la continua legivel, com o `kind` vazio.
        with engine.connect() as conexao:
            from sqlalchemy import text as sql

            linha = conexao.execute(sql("SELECT request_id, kind FROM request_events")).one()
        assert linha == ("antiga", None)
    finally:
        engine.dispose()


def test_uma_tabela_de_provedores_antiga_ganha_o_limite_de_concorrencia(tmp_path):
    """Banco criado antes do limite por provedor: `providers` sem
    `max_concurrency`. O boot acrescenta a coluna e a linha antiga fica com
    NULL, que e "sem limite" -- o comportamento de antes."""
    import sqlite3

    from sqlalchemy import inspect
    from sqlalchemy import text as sql

    from app.stats.engine import add_missing_columns

    caminho = tmp_path / "provedores-antigos.db"
    antigo = sqlite3.connect(caminho)
    antigo.execute(
        "CREATE TABLE providers (id INTEGER PRIMARY KEY, name VARCHAR(64),"
        " base_url VARCHAR(512), protocol VARCHAR(16), api_key VARCHAR(512),"
        " created_at DATETIME, updated_at DATETIME)"
    )
    antigo.execute(
        "INSERT INTO providers (name, base_url, protocol) VALUES ('local', 'http://l', 'openai')"
    )
    antigo.commit()
    antigo.close()

    engine = build_engine(f"sqlite+pysqlite:///{caminho}")
    assert engine is not None
    try:
        colunas = {c["name"] for c in inspect(engine).get_columns("providers")}
        assert "max_concurrency" in colunas
        assert add_missing_columns(engine) == []
        with engine.connect() as conexao:
            linha = conexao.execute(sql("SELECT name, max_concurrency FROM providers")).one()
        assert linha == ("local", None)
    finally:
        engine.dispose()


def test_um_banco_legado_afrouxa_a_constraint_dos_tokens(tmp_path):
    """B1: banco criado antes de `input_tokens` virar NULLABLE continua NOT NULL.

    `add_missing_columns` so roda ALTER para coluna AUSENTE: quem ja tem a
    coluna com `notnull=1` continua rejeitando NULL, e o `IntegrityError`
    derrubava o lote inteiro no `recorder`. A metadata diz nullable; o banco
    diz notnull; o boot afrouxa so esta direcao.
    """
    import sqlite3

    from sqlalchemy import inspect

    caminho = tmp_path / "legado-tokens.db"
    antigo = sqlite3.connect(caminho)
    antigo.execute(
        "CREATE TABLE request_events ("
        " id INTEGER PRIMARY KEY, request_id VARCHAR(64), started_at DATETIME,"
        " input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL)"
    )
    antigo.execute(
        "INSERT INTO request_events (request_id, started_at, input_tokens, output_tokens)"
        " VALUES ('antiga', '2026-09-01 10:00:00', 100, 100)"
    )
    antigo.commit()
    antigo.close()

    engine = build_engine(f"sqlite+pysqlite:///{caminho}")
    assert engine is not None
    try:
        info = {c["name"]: c for c in inspect(engine).get_columns("request_events")}
        assert info["input_tokens"]["nullable"] is True
        assert info["output_tokens"]["nullable"] is True
        # A linha antiga sobreviveu a reescrita da tabela.
        with engine.connect() as conexao:
            from sqlalchemy import text as sql

            linha = conexao.execute(
                sql("SELECT request_id, input_tokens, output_tokens FROM request_events")
            ).one()
        assert linha == ("antiga", 100, 100)
        # E o NULL, que o legado rejeitava, agora entra.
        with engine.begin() as conexao:
            conexao.execute(
                sql(
                    "INSERT INTO request_events (request_id, started_at, input_tokens)"
                    " VALUES ('muda', '2026-09-01 10:00:00', NULL)"
                )
            )
    finally:
        engine.dispose()


def test_a_metadata_table_missing_from_the_database_is_skipped(tmp_path):
    """Banco VAZIO: toda tabela da metadata esta ausente.

    `create_all` cria o que falta; quem chama `add_missing_columns` num banco
    sem as tabelas tem de sair pela primeira linha do laco. Sem o `continue`,
    um ALTER/PRAGMA numa tabela que nao existe seria erro de boot em vez de
    trabalho pulado.
    """
    from sqlalchemy import create_engine, inspect

    from app.stats.engine import add_missing_columns, relax_strict_columns

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'vazio.db'}")
    try:
        assert inspect(engine).get_table_names() == []
        assert add_missing_columns(engine) == []
        assert relax_strict_columns(engine) == []
    finally:
        engine.dispose()


def test_a_destination_that_refuses_to_relax_is_not_a_boot_failure(tmp_path, monkeypatch, caplog):
    """Turso pode recusar o `PRAGMA writable_schema`; afrouxar e conveniencia.

    A recusa nao pode derrubar o boot: o `recorder` isola a linha que viola, e
    o operador ve o descarte subir no painel. Qualquer excecao dentro do bloco
    vira warning e lista vazia -- a engine sobe igual.
    """
    import sqlite3

    from sqlalchemy import create_engine

    from app.stats import engine as engine_module

    caminho = tmp_path / "legado-recusado.db"
    antigo = sqlite3.connect(caminho)
    antigo.execute(
        "CREATE TABLE request_events ("
        " id INTEGER PRIMARY KEY, input_tokens INTEGER NOT NULL)"
    )
    antigo.execute("INSERT INTO request_events (id, input_tokens) VALUES (1, 100)")
    antigo.commit()
    antigo.close()

    engine = create_engine(f"sqlite+pysqlite:///{caminho}")

    def recusa(connection, table_name, columns):
        raise RuntimeError("PRAGMA writable_schema recusado")

    monkeypatch.setattr(engine_module, "_relax_table", recusa)
    try:
        assert engine_module.relax_strict_columns(engine) == []
    finally:
        engine.dispose()
    assert "nao deu para afrouxar constraints legadas" in caplog.text


def test_relax_table_without_the_table_returns_nothing(tmp_path):
    """PRAGMA numa tabela ausente devolve zero linhas: nao ha o que reescrever."""
    from app.stats.engine import _relax_table, build_engine

    engine = build_engine(f"sqlite+pysqlite:///{tmp_path / 'sem-tabela.db'}")
    assert engine is not None
    try:
        with engine.connect() as connection:
            assert _relax_table(connection, "tabela_que_nao_existe", ["input_tokens"]) == []
    finally:
        engine.dispose()


def test_relax_table_with_no_matching_strict_column_does_nothing(tmp_path):
    """Nenhuma coluna pedida esta em `notnull`: nao ha DDL a reescrever."""
    from app.stats.engine import _relax_table, build_engine

    engine = build_engine(f"sqlite+pysqlite:///{tmp_path / 'sem-strict.db'}")
    assert engine is not None
    try:
        with engine.connect() as connection:
            assert _relax_table(connection, "request_events", ["coluna_inexistente"]) == []
    finally:
        engine.dispose()


def test_uma_tabela_de_modelos_antiga_ganha_a_coluna_effort(tmp_path):
    """Banco criado antes do effort por modelo: `models` sem `effort`. O boot
    acrescenta a coluna e a linha antiga fica com NULL, que e "usar o do
    harness" -- o comportamento de antes."""
    import sqlite3

    from sqlalchemy import inspect
    from sqlalchemy import text as sql

    from app.stats.engine import add_missing_columns

    caminho = tmp_path / "modelos-antigos.db"
    antigo = sqlite3.connect(caminho)
    antigo.execute(
        "CREATE TABLE models (id INTEGER PRIMARY KEY, alias VARCHAR(128),"
        " provider_id INTEGER, upstream_model VARCHAR(256), supports_tools BOOLEAN,"
        " supports_streaming BOOLEAN, supports_vision BOOLEAN, context_window INTEGER,"
        " max_output_tokens INTEGER, is_default BOOLEAN, created_at DATETIME,"
        " updated_at DATETIME)"
    )
    antigo.execute(
        "INSERT INTO models (alias, provider_id, upstream_model, context_window,"
        " max_output_tokens, is_default) VALUES ('velho', 1, 'x', 1024, 128, 0)"
    )
    antigo.commit()
    antigo.close()

    engine = build_engine(f"sqlite+pysqlite:///{caminho}")
    assert engine is not None
    try:
        colunas = {c["name"] for c in inspect(engine).get_columns("models")}
        assert "effort" in colunas
        assert add_missing_columns(engine) == []
        with engine.connect() as conexao:
            linha = conexao.execute(sql("SELECT alias, effort FROM models")).one()
        assert linha == ("velho", None)
    finally:
        engine.dispose()
