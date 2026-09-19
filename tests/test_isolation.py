"""A suite roda cega para o banco de verdade, e isso e verificado aqui.

Medido antes da protecao existir: uma rodada de `make check` com
`TURSO_DATABASE_URL` no `.env` gravou 166 linhas no `stats.db` do usuario. Os
testes de rota sobem o aplicativo real, o `lifespan` le o `.env`, e a
persistencia conecta no que encontrar.
"""

import os

from fastapi.testclient import TestClient

from app.core.upstream import UpstreamPool
from app.main import app
from app.stats.engine import build_engine, database_url
from tests.core.test_dispatcher import SETTINGS


def test_the_database_variables_are_neutralised_while_the_suite_runs():
    # Vazio e nao ausente: ausente seria repreenchido pelo `.env` na primeira
    # chamada de `load_dotenv`, que todo boot do aplicativo faz.
    assert os.environ.get("TURSO_DATABASE_URL") == ""
    assert os.environ.get("TURSO_AUTH_TOKEN") == ""
    assert database_url() is None
    assert build_engine() is None


def test_booting_the_app_in_a_test_leaves_persistence_off():
    """O `lifespan` de verdade, sem gravador injetado: nao pode achar banco."""
    for attribute in ("settings", "pool", "recorder"):
        if hasattr(app.state, attribute):
            delattr(app.state, attribute)
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert app.state.recorder.enabled is False
    del app.state.recorder
