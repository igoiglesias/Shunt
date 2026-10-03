"""A engine do armazem, separada de tudo que serve requisicao.

Duas decisoes moram aqui:

- **Engine propria, pool proprio.** O `UpstreamPool` dos provedores e este banco
  nao compartilham cliente, semaforo nem limite de conexao. Banco lento nao pode
  ocupar vaga de provedor: se compartilhassem, uma escrita presa no Turso
  seguraria uma chamada de modelo.
- **Sem URL, sem banco, e o proxy sobe igual.** `build_engine` devolve `None`
  quando `TURSO_DATABASE_URL` nao existe. Persistencia e um extra; trocar um
  extra por um ponto de falha no boot seria um mau negocio.

O dialeto do libsql e SINCRONO -- nao existe driver async para ele. Isso nao
custa nada ao proxy porque quem chama este modulo e o worker de fundo, que roda
fora do caminho da requisicao e pode bloquear a vontade na propria thread.
"""

import logging
import os
import socket
from urllib.parse import urlparse

from sqlalchemy import Engine, create_engine, event, inspect, text

from app.config.config import BUSY_TIMEOUT_MS, REACH_TIMEOUT
from app.stats.models import Base

logger = logging.getLogger("shunt")

# Prazo do teste de alcance e a espera por trava de banco agora vivem em
# `app/config/config.py` (knobs `SHUNT_REACH_TIMEOUT` / `SHUNT_BUSY_TIMEOUT_MS`);
# as medicoes que os criaram estao nos comentarios de la.
DEFAULT_PORTS = {"libsql": 443, "https": 443, "http": 80}


def _target(url: str) -> tuple[str, int] | None:
    """Host e porta de uma URL de rede, ou None quando o destino e um arquivo."""
    parsed = urlparse(url)
    if not parsed.hostname:
        return None
    scheme = parsed.scheme.split("+")[-1]
    if scheme in {"pysqlite", "sqlite"}:
        return None
    return parsed.hostname, parsed.port or DEFAULT_PORTS.get(scheme, 443)


def reachable(url: str, timeout: float = REACH_TIMEOUT) -> bool:
    """Da para abrir um socket ate o banco dentro do prazo?

    Banco em arquivo responde True sem tocar em rede nenhuma.
    """
    target = _target(url)
    if target is None:
        return True
    try:
        with socket.create_connection(target, timeout=timeout):
            return True
    except OSError as err:
        logger.warning("stats: banco inalcancavel em %s:%s (%s)", *target, err)
        return False


def database_url() -> str | None:
    """A URL do Turso montada a partir do ambiente, ou None quando nao ha banco.

    O token vai na query string porque e assim que o dialeto `sqlite+libsql`
    o recebe. Uma URL ja completa no ambiente passa intacta: quem aponta para um
    arquivo local durante um teste nao precisa de token nenhum.
    """
    url = os.environ.get("TURSO_DATABASE_URL")
    if not url:
        return None
    token = os.environ.get("TURSO_AUTH_TOKEN")
    if token and "authToken=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}authToken={token}"
    return url


def apply_pragmas(dbapi_connection: object) -> None:
    """WAL e prazo de trava numa conexao recem-aberta.

    Destino que nao conhece PRAGMA -- o Turso, por exemplo -- recusa, e recusar
    aqui nao e falha: e outro banco. A conexao segue valida.
    """
    try:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        cursor.close()
    except Exception as err:  # noqa: BLE001 - ver docstring.
        logger.debug("stats: PRAGMA recusado (%s: %s)", type(err).__name__, err)


def _tune_for_many_writers(engine: Engine) -> None:
    """WAL e prazo de trava, em toda conexao nova do pool.

    WAL deixa um leitor -- o painel -- ler enquanto um worker grava, o que com o
    journal padrao seria bloqueio mutuo. Vale so para SQLite em arquivo; o
    Turso ignora e nao se importa, entao a falha aqui nao e erro.
    """

    event.listen(engine, "connect", lambda connection, _record: apply_pragmas(connection))


def add_missing_columns(engine: Engine) -> list[str]:
    """Acrescenta ao banco existente as colunas que o modelo ganhou depois.

    `create_all` cria tabela que falta e ignora tabela que existe -- inclusive
    quando ela esta com uma coluna a menos. Medido: uma coluna nova quebrou
    TODA leitura da tabela antiga com `no such column`, e so no banco de quem
    ja estava usando. Migracao de verdade e outra conversa; para um armazem de
    telemetria, ADD COLUMN cobre o caso e nao pede ferramenta nenhuma.

    Devolve o que acrescentou, para o log dizer o que mudou.
    """
    added = []
    inspector = inspect(engine)
    with engine.begin() as connection:
        for table in Base.metadata.sorted_tables:
            if table.name not in inspector.get_table_names():
                continue
            existing = {column["name"] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                kind = column.type.compile(engine.dialect)
                connection.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {column.name} {kind}"))
                added.append(f"{table.name}.{column.name}")
    if added:
        logger.info("stats: colunas acrescentadas ao banco existente: %s", ", ".join(added))
    return added


def relax_strict_columns(engine: Engine) -> list[str]:
    """Afrouxa a constraint NOT NULL de colunas que a metadata diz NULLABLE.

    Cenario real: bancos criados antes de `input_tokens`/`output_tokens` virarem
    nullable (Task 2) continuam com `notnull=1`, e rejeitam o NULL que os
    produtores passam para significar silencio -- um `IntegrityError` que o
    `recorder` tratava descartando o LOTE inteiro. `add_missing_columns` nao
    toca em coluna que ja existe, entao a constraint antiga sobrevivia ao boot.

    O SQLite NAO tem `ALTER TABLE ... ALTER COLUMN ... SET NOT NULL` (medido:
    syntax error). O caminho e reescrever o `CREATE TABLE` registrado em
    `sqlite_master` com `PRAGMA writable_schema`, preservando dados, indices e
    as DEMAIS constraints: a unica direcao e notnull -> nullable, e so para
    colunas que a metadata atual diz nullable e o banco diz notnull.

    Turso/libsql remoto pode recusar qualquer um destes passos. A recusa nao e
    falha: o `recorder` isola a linha que viola (B2), e o operador ve o
    contador de descarte subir no painel em vez de estatistica sumir.
    """
    relaxed: list[str] = []
    inspector = inspect(engine)
    try:
        with engine.connect() as connection:
            for table in Base.metadata.sorted_tables:
                if table.name not in inspector.get_table_names():
                    continue
                info = {column["name"]: column for column in inspector.get_columns(table.name)}
                strict = [
                    column.name
                    for column in table.columns
                    if column.nullable and info.get(column.name, {}).get("nullable") is False
                ]
                if not strict:
                    continue
                relaxed.extend(_relax_table(connection, table.name, strict))
            connection.commit()
    except Exception as err:  # noqa: BLE001 - o destino e outro banco, e a
        # recusa e esperada; afrouxar e conveniencia, nao requisito de boot.
        logger.warning(
            "stats: nao deu para afrouxar constraints legadas (%s: %s)",
            type(err).__name__,
            err,
        )
        return []
    if relaxed:
        logger.info(
            "stats: constraints NOT NULL afrouxadas no banco existente: %s",
            ", ".join(relaxed),
        )
    return relaxed


def _relax_table(connection, table_name: str, columns: list[str]) -> list[str]:
    """Reescreve o `CREATE TABLE` de uma tabela removendo NOT NULL das colunas.

    Devolve as colunas que efetivamente afrouxou. O novo DDL e construido por
    COLUNA a partir do `PRAGMA table_info` (a fonte estruturada do SQLite), e
    nao por replace no texto do CREATE: replace casaria so com a sintaxe exata
    e deixaria escapar formatacao diferente de aspas ou espacos.
    """
    info = connection.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
    if not info:
        return []
    strict_set = set(columns)
    # Ordem do PRAGMA: cid, name, type, notnull, dflt_value, pk.
    afrouxadas = [
        name for _cid, name, _kind, notnull, _default, _pk in info
        if name in strict_set and notnull
    ]
    if not afrouxadas:
        return []
    partes = []
    for _cid, name, kind, notnull, default, pk in info:
        if name in afrouxadas:
            # So a direcao notnull -> nullable: o tipo e o default seguem intactos.
            partes.append(f'"{name}" {kind}{" PRIMARY KEY" if pk else ""}')
        else:
            partes.append(
                f'"{name}" {kind}'
                f'{" NOT NULL" if notnull else ""}'
                f'{" PRIMARY KEY" if pk else ""}'
                f'{f" DEFAULT {default}" if default is not None else ""}'
            )
    novo = f"CREATE TABLE {table_name} ({', '.join(partes)})"
    connection.execute(text("PRAGMA writable_schema=ON"))
    connection.execute(
        text("UPDATE sqlite_master SET sql = :sql WHERE type = 'table' AND name = :name"),
        {"sql": novo, "name": table_name},
    )
    connection.execute(text("PRAGMA writable_schema=OFF"))
    return [f"{table_name}.{name}" for name in afrouxadas]


def build_engine(url: str | None = None) -> Engine | None:
    """A engine pronta e com a tabela criada, ou None quando nao ha URL.

    Falha de conexao no boot nao derruba o processo: o proxy continua servindo
    e o painel mostra a persistencia desligada, que e a informacao que o
    operador precisa para consertar.
    """
    url = url or database_url()
    if not url:
        return None
    if not reachable(url):
        return None
    try:
        engine = create_engine(url, pool_pre_ping=True)
        _tune_for_many_writers(engine)
        Base.metadata.create_all(engine)
        add_missing_columns(engine)
        if relax_strict_columns(engine):
            # `PRAGMA writable_schema` so entra em vigor numa conexao NOVA: o
            # schema de quem abriu antes continua com a constraint antiga
            # (medido: mesmo INSERT falha na conexao antiga e passa na nova).
            # O pool e descartado para as proximas conexoes lerem o schema novo.
            engine.dispose()
    except Exception as err:  # noqa: BLE001 - qualquer falha de banco no boot e
        # a mesma decisao: seguir sem persistencia em vez de nao subir.
        logger.warning("stats: banco indisponivel no boot (%s: %s)", type(err).__name__, err)
        return None
    return engine
