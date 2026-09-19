"""Fixtures compartilhadas, e uma protecao que vale para a suite inteira.

A suite NUNCA pode escrever no banco de verdade. Medido: com
`TURSO_DATABASE_URL` no `.env`, uma rodada de `make check` gravou 166 linhas no
`stats.db` do usuario -- os testes de rota sobem o aplicativo real com
`TestClient`, o `lifespan` le o `.env` e conecta no que estiver la. Nenhum teste
pediu isso, e nenhum teste viu.

A limpeza e `autouse` e de sessao: qualquer teste que suba o aplicativo encontra
o ambiente sem banco, e a persistencia fica desligada a menos que o proprio
teste injete um `Recorder` com engine de arquivo temporario.
"""

import pytest


@pytest.fixture(autouse=True, scope="session")
def never_touch_the_real_database():
    import os

    # VAZIO, e nao removido: `load_dotenv` nao sobrescreve variavel existente,
    # mas repoe a que falta -- e o `.env` do repositorio tem a URL de verdade.
    # Medido: removendo, a suite voltava a escrever no banco do usuario.
    names = ("TURSO_DATABASE_URL", "TURSO_AUTH_TOKEN")
    saved = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = ""
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
