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

import argon2
import pytest
from fastapi.testclient import TestClient

TEST_SHUNT_TOKEN = "token-da-suite-de-testes"
# Id que nenhum banco de teste gera (todos comecam em 1): invalidar o token de
# um teste de revogacao nunca derruba o token da suite.
TEST_SHUNT_TOKEN_ID = 999_999
_ORIGINAL_INIT = TestClient.__init__


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_argon2: mantem o argon2 de producao no teste (desliga a fixture light_argon2)",
    )


@pytest.fixture(autouse=True)
def light_argon2(request, monkeypatch):
    """Troca `app.core.security._hasher` por um argon2 barato em todo teste.

    O default do argon2-cffi custa ~75-145ms por hash+verify, e a suite faz
    muitos logins; sem isso, a suite inteira fica lenta por causa de um
    parametro de seguranca que nao muda o comportamento testado. Funciona
    porque `hash_password`/`verify_password` leem o global `_hasher` na hora
    da chamada (`app/core/security.py:24,35`), entao a troca por
    `monkeypatch.setattr` vale mesmo sem reimportar o modulo.

    O teste-guarda usa `@pytest.mark.real_argon2` para desligar a troca e
    confirmar que `_hasher` de producao continua no default da biblioteca.
    """
    if request.node.get_closest_marker("real_argon2"):
        yield
        return
    from app.core import security

    leve = argon2.PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)
    monkeypatch.setattr(security, "_hasher", leve)
    yield


@pytest.fixture(autouse=True)
def shunt_token_everywhere(monkeypatch):
    """Todo `TestClient` manda `x-shunt-token` por padrao, e o token ja esta no
    cache de validacao: a suite roda sem banco, entao sem este cache semeado
    todo `/v1` responderia 503. Os 32 arquivos que chamam `/v1/` seguem sem
    edicao. Quem testa a AUSENCIA do token pede `without_shunt_token`."""
    from app.core.token_auth import TokenCache, token_hash
    from app.main import app

    cache = TokenCache(ttl=3600)
    cache.put(token_hash(TEST_SHUNT_TOKEN), TEST_SHUNT_TOKEN_ID)
    app.state.token_cache = cache  # cache novo por teste: nada vaza entre testes

    def patched(self, *args, **kwargs):
        headers = dict(kwargs.pop("headers", None) or {})
        headers.setdefault("x-shunt-token", TEST_SHUNT_TOKEN)
        _ORIGINAL_INIT(self, *args, headers=headers, **kwargs)

    monkeypatch.setattr(TestClient, "__init__", patched)
    yield


@pytest.fixture
def without_shunt_token(monkeypatch):
    """O `TestClient` volta a nao mandar token. O cache semeado continua: quem
    manda `TEST_SHUNT_TOKEN` por prefixo ou query ainda e aceito sem banco."""
    monkeypatch.setattr(TestClient, "__init__", _ORIGINAL_INIT)


@pytest.fixture(autouse=True)
def _recorder_not_leaked():
    yield
    # Testes que injetam um `Recorder` com engine fazem isso no `app.state`;
    # sem limpeza o proximo `TestClient` reaproveita o objeto (worker encerrado,
    # engine antiga) e o teste que exige o comportamento "sem banco" (503)
    # passa a ver banco.
    from app.main import app

    if hasattr(app.state, "recorder"):
        del app.state.recorder


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
