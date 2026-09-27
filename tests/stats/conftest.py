"""Fabrica de engine que se fecha sozinha no fim do teste.

Sem o `dispose`, o pytest acusa `ResourceWarning: unclosed database`: o pool do
SQLAlchemy segura a conexao do SQLite ate ser descartado, e um teste que cria
engine por arquivo temporario deixa uma aberta por teste.

Sem `url`, o factory usa sqlite em memoria com `StaticPool` (mesma conexao
compartilhada entre `Session`s, sem o que cada sessao veria um banco em
memoria vazio e diferente). Medido: `create_all` cai de 104-128 ms em arquivo
para 6-7 ms em memoria. So os 7 arquivos de consulta pura usam essa forma
(`test_queries.py`, `test_search.py`, `test_dossier.py`, `test_analysis.py`,
`test_models.py`, `test_user_models.py`, `test_bodies.py`); os demais
continuam com arquivo:

- Recorder roda em thread (`asyncio.to_thread`, `app/stats/recorder.py:182,
  190`): `test_recorder.py`, `test_wiring.py`, `test_body_wiring.py`,
  `test_*_api.py` e `tests/routers/test_admin_*.py`;
- sobem via URL de boot (`build_engine` sem `StaticPool`): `tests/test_main.py`,
  `tests/config/test_boot.py`, `tests/config/test_seed.py`,
  `tests/core/test_config_watcher.py`, `tests/core/test_token_auth.py` e
  `tests/routers/test_v1_auth.py`;
- reabrem o mesmo arquivo ou migram schema: `tests/stats/test_engine.py`;
- rodam em subprocesso: `tests/e2e/test_analysis_e2e.py` e `tests/browser/*`.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.stats.models import Base


@pytest.fixture
def make_engine():
    created = []

    def factory(url=None, create_tables=True):
        if url is None:
            engine = create_engine(
                "sqlite+pysqlite://",
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
            )
        else:
            engine = create_engine(url)
        created.append(engine)
        if create_tables:
            Base.metadata.create_all(engine)
        return engine

    yield factory
    for engine in created:
        engine.dispose()
