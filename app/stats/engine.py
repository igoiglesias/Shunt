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

from sqlalchemy import Engine, create_engine

from app.stats.models import Base

logger = logging.getLogger("shunt")


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


def build_engine(url: str | None = None) -> Engine | None:
    """A engine pronta e com a tabela criada, ou None quando nao ha URL.

    Falha de conexao no boot nao derruba o processo: o proxy continua servindo
    e o painel mostra a persistencia desligada, que e a informacao que o
    operador precisa para consertar.
    """
    url = url or database_url()
    if not url:
        return None
    try:
        engine = create_engine(url, pool_pre_ping=True)
        Base.metadata.create_all(engine)
    except Exception as err:  # noqa: BLE001 - qualquer falha de banco no boot e
        # a mesma decisao: seguir sem persistencia em vez de nao subir.
        logger.warning("stats: banco indisponivel no boot (%s: %s)", type(err).__name__, err)
        return None
    return engine
