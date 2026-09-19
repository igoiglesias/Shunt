"""Fabrica de engine que se fecha sozinha no fim do teste.

Sem o `dispose`, o pytest acusa `ResourceWarning: unclosed database`: o pool do
SQLAlchemy segura a conexao do SQLite ate ser descartado, e um teste que cria
engine por arquivo temporario deixa uma aberta por teste.
"""

import pytest
from sqlalchemy import create_engine

from app.stats.models import Base


@pytest.fixture
def make_engine():
    created = []

    def factory(url, create_tables=True):
        engine = create_engine(url)
        created.append(engine)
        if create_tables:
            Base.metadata.create_all(engine)
        return engine

    yield factory
    for engine in created:
        engine.dispose()
