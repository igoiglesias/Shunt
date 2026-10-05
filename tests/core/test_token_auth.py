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
    require_shunt_token_or_transparent,
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


def test_purge_drops_only_the_expired_entries():
    """`purge` varre o cache e remove so o que venceu: a entrada viva sobrevive.

    Sem isso, o cache de validacao enche e o `put` no teto comecaria a
    descartar entradas vivas por FIFO."""
    now = [100.0]
    cache = TokenCache(ttl=60, clock=lambda: now[0])
    cache.put(token_hash("vivo"), 1)  # vence em 160
    now[0] = 200.0
    cache.put(token_hash("novo"), 2)  # vence em 260
    now[0] = 230.0
    cache.purge()
    assert cache.lookup(token_hash("vivo")) == (False, None)  # expirado: sai no purge
    assert cache.lookup(token_hash("novo")) == (True, 2)  # viva: o purge nao toca


def test_put_at_the_ceiling_purges_the_expired_before_evicting():
    """No teto: `put` chama `purge` e so depois decide evictar. Com uma entrada
    expirada presente, o purge abre espaco e a entrada VIVA e preservada."""
    now = [100.0]
    cache = TokenCache(ttl=60, clock=lambda: now[0], max_entries=2)
    cache.put(token_hash("vivo"), 1)
    now[0] = 200.0
    cache.put(token_hash("morto"), 2)  # ja ocupa uma vaga
    # Cache cheio (2). O `put` novo dispara purge: o expirado sai, sobra vaga.
    cache.put(token_hash("novo"), 3)
    assert cache.lookup(token_hash("vivo")) == (False, None)  # expirou pelo ttl
    # A entrada expirada saiu sem evictar nenhuma viva pelo FIFO.
    assert cache.lookup(token_hash("novo")) == (True, 3)


def test_put_at_the_ceiling_with_nothing_expired_evicts_one_entry():
    """No teto sem nada expirado: o purge nao libera nada e o `put` evicta
    UMA entrada (a mais antiga) para abrir a vaga -- o cache nunca passa do
    teto."""
    now = [100.0]
    cache = TokenCache(ttl=3600, clock=lambda: now[0], max_entries=2)
    cache.put(token_hash("a"), 1)
    cache.put(token_hash("b"), 2)
    cache.put(token_hash("c"), 3)  # teto: purge nao remove nada, evicta "a"
    # "a" foi evictado (FIFO pela ordem de insercao do dict).
    assert cache.lookup(token_hash("a")) == (False, None)
    assert cache.lookup(token_hash("b")) == (True, 2)
    assert cache.lookup(token_hash("c")) == (True, 3)


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


def _request_without_recorder(headers: dict) -> Request:
    """Request sem `token_cache` semeado no alvo (miss garantido) e sem
    `app.state.recorder`: o Shunt sem banco, o caso que cai no ramo `engine is
    None` de `_lookup`."""
    request = _request_with_token_cache(headers)
    request.app.state.token_cache = TokenCache(ttl=3600)
    request.app.state.token_cache.put(token_hash(TEST_SHUNT_TOKEN), TEST_SHUNT_TOKEN_ID)
    if hasattr(request.app.state, "recorder"):
        delattr(request.app.state, "recorder")
    return request


def test_lookup_without_engine_strict_raises_503():
    """Forma explicita (strict=True) sem banco e sem cache: nao ha como conferir,
    e a resposta e 503 (R4) -- nunca seguir fingindo que o valor nao e token."""
    request = _request_without_recorder({})
    with pytest.raises(TokenRejected) as exc:
        asyncio.run(_lookup(request, token_hash("desconhecido"), strict=True))
    assert exc.value.status == 503
    assert exc.value.message == "shunt has no database to check the token"


def test_lookup_without_engine_non_strict_returns_none():
    """Credencial do caller (strict=False) sem banco: conta como "nao e token"
    e devolve None, em vez de derrubar com 503 todo pedido de um Shunt sem
    banco (o 401 em /api/oauth/usage medido em producao)."""
    request = _request_without_recorder({})
    assert asyncio.run(_lookup(request, token_hash("sk-do-harness"), strict=False)) is None


def test_require_shunt_token_without_any_credential_is_401():
    """`require_shunt_token` sem nenhuma credencial (nem explicita, nem
    `x-api-key`/`Authorization`): 401 pedindo a forma de envio do token."""
    request = _request_without_recorder({})
    with pytest.raises(TokenRejected) as exc:
        asyncio.run(require_shunt_token(request))
    assert exc.value.status == 401
    assert "missing shunt token" in exc.value.message


def _request_with_fresh_database(tmp_path, headers: dict, token_values: tuple = ()) -> Request:
    """Request com banco REAL (sqlite em tmp) sem nenhum token semeado no
    cache: o lookup cai no banco de verdade, e o caminho e o do producao."""
    engine = create_engine(f"sqlite:///{tmp_path}/vazio.db")
    Base.metadata.create_all(engine)
    for value in token_values:
        with Session(engine) as session:
            session.add(ApiToken(name="t", token_hash=token_hash(value)))
            session.commit()
    request = _request_with_token_cache(headers)
    request.app.state.token_cache = TokenCache(ttl=3600)
    request.app.state.recorder = Recorder(engine)
    return request


def test_require_shunt_token_with_an_invalid_explicit_header_is_401(tmp_path):
    """`x-shunt-token` EXPLICITO que nao existe no banco: 401 "unknown or
    expired", e nao o 503 do banco ausente -- a recusa e por valor invalido."""
    request = _request_with_fresh_database(tmp_path, {"x-shunt-token": "token-que-nao-existe"})
    with pytest.raises(TokenRejected) as exc:
        asyncio.run(require_shunt_token(request))
    assert exc.value.status == 401
    assert exc.value.message == "unknown or expired shunt token"


def test_require_shunt_token_or_transparent_without_any_credential_is_401():
    """A dependencia do relay sem credencial nenhuma: 401 nomeando tambem as
    formas aceitas (`x-api-key`/`Authorization`) para o bypass transparente."""
    request = _request_without_recorder({})
    with pytest.raises(TokenRejected) as exc:
        asyncio.run(require_shunt_token_or_transparent(request))
    assert exc.value.status == 401
    assert exc.value.message.startswith("missing shunt token or caller credentials")


def test_require_shunt_token_or_transparent_with_an_invalid_explicit_header_is_401(tmp_path):
    """Token Shunt explicito invalido no relay: 401 "unknown or expired", o
    mesmo contrato do v1."""
    request = _request_with_fresh_database(tmp_path, {"x-shunt-token": "token-que-nao-existe"})
    with pytest.raises(TokenRejected) as exc:
        asyncio.run(require_shunt_token_or_transparent(request))
    assert exc.value.status == 401
    assert exc.value.message == "unknown or expired shunt token"


def test_or_transparent_reuses_the_validated_explicit_token_in_the_credential(tmp_path):
    """`x-api-key` com o MESMO valor do token explicito ja validado: nao ha
    segundo lookup, o header entra em `token_headers` (para ser removido no
    repasse), mas a flag de credencial fica FALSA -- ali ja e o token."""
    request = _request_with_fresh_database(
        tmp_path, {"x-shunt-token": "tk-1", "x-api-key": "tk-1"}, token_values=("tk-1",)
    )
    token_id = asyncio.run(require_shunt_token_or_transparent(request))
    assert token_id == 1
    assert request.state.credential_is_token is False
    assert request.state.token_headers == frozenset({"x-api-key"})


def test_or_transparent_accepts_a_shunt_token_hidden_in_the_credential(tmp_path):
    """`x-api-key` que NAO e o token explicito mas CASOU no banco como token
    Shunt: autentica pelo token (a flag de credencial sobe para o dispatcher
    injetar a chave do catalogo, e nao repassar o token como segredo)."""
    request = _request_with_fresh_database(tmp_path, {"x-api-key": "tk-1"}, token_values=("tk-1",))
    token_id = asyncio.run(require_shunt_token_or_transparent(request))
    assert token_id == 1
    assert request.state.credential_is_token is True
    assert request.state.token_headers == frozenset({"x-api-key"})


def test_require_shunt_token_on_an_exempt_path_skips_everything(monkeypatch):
    """Path em `EXEMPT_PATHS`: nem header, nem banco, nem cache -- autenticacao
    desligada e `shunt_token_id` 0 (isento)."""
    from app.core import token_auth

    monkeypatch.setattr(token_auth, "EXEMPT_PATHS", frozenset({"/v1/messages"}))
    request = _request_without_recorder({})  # sem credencial nenhuma: o path isento salva
    token_id = asyncio.run(require_shunt_token(request))
    assert token_id == 0
    assert request.state.shunt_token_id is None
    assert request.state.token_headers == frozenset()


def test_require_shunt_token_or_transparent_on_an_exempt_path_skips_everything(monkeypatch):
    """Idem para a dependencia do relay: path isento volta 0 sem checar nada."""
    from app.core import token_auth

    monkeypatch.setattr(token_auth, "EXEMPT_PATHS", frozenset({"/v1/messages"}))
    request = _request_without_recorder({})
    token_id = asyncio.run(require_shunt_token_or_transparent(request))
    assert token_id == 0
    assert request.state.shunt_token_id is None
    assert request.state.token_headers == frozenset()


def test_require_shunt_token_conflicting_forms_is_400():
    """Duas formas EXPLICITAS com valores diferentes: 400 antes de qualquer
    validacao (o cliente mandou o token em mais de um lugar errado)."""
    request = _request_without_recorder({"x-shunt-token": "abc"})
    request.scope["query_string"] = b"token=xyz"
    with pytest.raises(TokenRejected) as exc:
        asyncio.run(require_shunt_token(request))
    assert exc.value.status == 400
    assert "more than one form" in exc.value.message


def test_require_shunt_token_or_transparent_conflicting_forms_is_400():
    """Mesma regra para a dependencia do relay."""
    request = _request_without_recorder({"x-shunt-token": "abc"})
    request.scope["query_string"] = b"token=xyz"
    with pytest.raises(TokenRejected) as exc:
        asyncio.run(require_shunt_token_or_transparent(request))
    assert exc.value.status == 400
    assert "more than one form" in exc.value.message
