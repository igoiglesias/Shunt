"""Token do Shunt: leitura das tres formas, hash, mascara, cache e dependencia."""
import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.core.token_auth import (
    TokenCache,
    TokenConflict,
    TokenRejected,
    _db_lookup,
    _lookup,
    mask_path,
    mask_query,
    read_token,
    require_shunt_token,
    token_hash,
)
from app.main import app
from app.stats.models import ApiToken, Base
from app.stats.recorder import Recorder
from tests.conftest import TEST_SHUNT_TOKEN, TEST_SHUNT_TOKEN_ID


def test_reads_the_same_token_from_any_form():
    assert read_token({"x-shunt-token": "abc"}, None, None) == "abc"
    assert read_token({}, "abc", None) == "abc"
    assert read_token({}, None, "abc") == "abc"
    assert read_token({}, None, None) is None


def test_two_forms_with_different_values_is_a_conflict():
    with pytest.raises(TokenConflict):
        read_token({"x-shunt-token": "abc"}, "xyz", None)
    assert read_token({"x-shunt-token": "abc"}, "abc", "abc") == "abc"


def test_masks_prefix_and_query_everywhere():
    assert mask_path("/t/segredo/v1/messages") == "/t/***/v1/messages"
    assert mask_path("/v1/messages") == "/v1/messages"
    assert mask_query("a=1&token=segredo&b=2") == "a=1&token=***&b=2"


def test_cache_expires_by_injected_clock_and_invalidates_by_id():
    now = [100.0]
    cache = TokenCache(ttl=60, clock=lambda: now[0])
    h = token_hash("abc")
    assert cache.lookup(h) == (False, None)
    cache.put(h, 7)
    assert cache.lookup(h) == (True, 7)
    cache.put(token_hash("nao"), None)
    assert cache.lookup(token_hash("nao")) == (True, None)
    now[0] = 161.0
    assert cache.lookup(h) == (False, None)
    cache.put(h, 7); cache.invalidate(7)
    assert cache.lookup(h) == (False, None)


def _request_with_token_cache(headers: dict, path: str = "/v1/messages") -> Request:
    """Escopo ASGI minimo com o cache de validacao semeado, sem banco.

    `require_shunt_token` so confere no banco quando o valor cai fora do
    cache; com o token ja no cache nenhum teste desta funcao toca o banco."""
    cache = TokenCache(ttl=3600)
    cache.put(token_hash(TEST_SHUNT_TOKEN), TEST_SHUNT_TOKEN_ID)
    app.state.token_cache = cache
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "scheme": "http",
        "server": ("test", 80),
        "client": ("127.0.0.1", 1),
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "app": app,
        "state": {},
    }
    return Request(scope)


def test_the_headers_where_the_token_matched_are_recorded():
    """`token_headers` lista os nomes onde o token do Shunt casou (T5 remove
    esses headers antes de repassar ao upstream). A `authorization` aqui NAO
    casou (e a credencial real do harness) e fica fora da lista."""
    req = _request_with_token_cache({"x-api-key": TEST_SHUNT_TOKEN, "authorization": "Bearer sk-real"})

    asyncio.run(require_shunt_token(req))
    assert req.state.token_headers == frozenset({"x-api-key"})


def test_no_token_in_the_credential_gives_an_empty_token_headers():
    """Sem token na credencial (aqui: so a forma explicita + `x-api-key` que
    nao e token) a lista fica vazia: nao ha nada a remover no repasse."""
    req = _request_with_token_cache({"x-shunt-token": TEST_SHUNT_TOKEN, "x-api-key": "sk-do-harness"})

    asyncio.run(require_shunt_token(req))
    assert req.state.token_headers == frozenset()


def test_a_shunt_token_passed_as_credential_marks_credential_is_token():
    """`x-api-key`/`Authorization` que CASOU com um token do Shunt nao e a
    credencial do provedor: a flag manda o dispatcher injetar a chave
    configurada. Sem a flag, o token viraria credencial e seguiria ao provedor."""
    req = _request_with_token_cache({"x-api-key": TEST_SHUNT_TOKEN})

    token_id = asyncio.run(require_shunt_token(req))
    assert token_id == TEST_SHUNT_TOKEN_ID
    assert req.state.credential_is_token is True


def test_an_explicit_header_token_keeps_credential_is_token_false():
    """O mesmo token apresentado pela forma EXPLICITA (`x-shunt-token`) autentica
    sem marcar a flag: a credencial do cliente, se houver, segue como credencial."""
    req = _request_with_token_cache({"x-shunt-token": TEST_SHUNT_TOKEN})

    token_id = asyncio.run(require_shunt_token(req))
    assert token_id == TEST_SHUNT_TOKEN_ID
    assert req.state.credential_is_token is False


def test_a_token_without_expiry_is_valid():
    import tempfile

    engine = create_engine(f"sqlite:///{tempfile.mkdtemp()}/t.db")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(ApiToken(name="t", token_hash=token_hash("nao-vence")))
        session.commit()
    assert _db_lookup(engine, token_hash("nao-vence")) == 1
    assert _db_lookup(engine, token_hash("que-nao-existe")) is None


def test_a_future_expiry_is_still_valid_even_if_stored_naive():
    """`DateTime(timezone=True)` em SQLite pode vir de volta sem `tzinfo`; a
    comparacao com `now(UTC)` sem normalizar dava `TypeError` -> 500."""
    import tempfile

    engine = create_engine(f"sqlite:///{tempfile.mkdtemp()}/t.db")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            ApiToken(
                name="t",
                token_hash=token_hash("vence-depois"),
                expires_at=datetime.now(UTC).replace(tzinfo=None) + timedelta(days=1),
            )
        )
        session.commit()
    assert _db_lookup(engine, token_hash("vence-depois")) == 1


def test_an_expiry_exactly_at_now_is_still_valid(monkeypatch):
    """Fronteira: `expires == now` ainda vale. Sem relógio fixo o teste passava
    por coincidencia de segundo; o mutante `<` -> `<=` precisava do corte no
    ponto certo para ser visto."""
    import tempfile

    from app.core import token_auth

    fixed = datetime(2026, 9, 24, 12, 0, 0, tzinfo=UTC)

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(token_auth, "datetime", _FakeDatetime)
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp()}/t.db")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            ApiToken(
                name="t",
                token_hash=token_hash("vence-ja"),
                expires_at=fixed.replace(tzinfo=None),
            )
        )
        session.commit()
    assert _db_lookup(engine, token_hash("vence-ja")) == 1


def test_a_past_expiry_is_rejected_even_if_stored_naive():
    import tempfile

    engine = create_engine(f"sqlite:///{tempfile.mkdtemp()}/t.db")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            ApiToken(
                name="t",
                token_hash=token_hash("venceu"),
                expires_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1),
            )
        )
        session.commit()
    assert _db_lookup(engine, token_hash("venceu")) is None


def _engine_that_cannot_open(tmp_path):
    """Engine cujo "arquivo" e a propria pasta: abre, mas toda consulta levanta."""
    return create_engine(f"sqlite:///{tmp_path}")


def test_a_dead_database_with_a_cache_hit_still_authenticates(tmp_path):
    """Cache valido e a resposta: o banco fora do ar nao vira 500. `recorder`
    com engine quebrado simula o banco que caiu depois do boot."""
    scope = _request_with_token_cache({"x-shunt-token": TEST_SHUNT_TOKEN}).scope
    request = Request(scope)
    request.app.state.token_cache = TokenCache(ttl=3600)
    request.app.state.token_cache.put(token_hash(TEST_SHUNT_TOKEN), TEST_SHUNT_TOKEN_ID)
    request.app.state.recorder = Recorder(_engine_that_cannot_open(tmp_path))
    token_id = asyncio.run(require_shunt_token(request))
    assert token_id == TEST_SHUNT_TOKEN_ID


def test_a_dead_database_rejects_an_explicit_unknown_token_not_crashes(tmp_path):
    """Forma explicita (strict) sem cache: banco fora do ar tem a resposta do
    banco ausente (R4) -- 503, nunca um 500 de `OperationalError`."""
    scope = _request_with_token_cache({"x-shunt-token": "desconhecido"}, path="/v1/messages").scope
    request = Request(scope)
    request.app.state.token_cache = TokenCache(ttl=3600)
    request.app.state.recorder = Recorder(_engine_that_cannot_open(tmp_path))
    with pytest.raises(TokenRejected) as exc:
        asyncio.run(require_shunt_token(request))
    assert exc.value.status == 503


def test_a_dead_database_counts_a_non_explicit_credential_as_not_a_token(tmp_path):
    """`x-api-key` que nao acerta o cache (strict=False): banco fora do ar conta
    como "nao e token" e segue como credencial, sem 503 e sem 500."""
    engine = _engine_that_cannot_open(tmp_path)
    cache = TokenCache(ttl=3600)
    request = _request_with_token_cache({"x-api-key": "sk-do-harness"})
    request.app.state.token_cache = cache
    request.app.state.recorder = Recorder(engine)
    found = asyncio.run(_lookup(request, token_hash("sk-do-harness"), strict=False))
    assert found is None
