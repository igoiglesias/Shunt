"""Token do Shunt: leitura das tres formas, hash, mascara, cache e dependencia."""
import pytest
from starlette.requests import Request

from app.core.token_auth import (
    TokenCache,
    TokenConflict,
    mask_path,
    mask_query,
    read_token,
    require_shunt_token,
    token_hash,
)
from app.main import app
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


def test_a_shunt_token_passed_as_credential_marks_credential_is_token():
    """`x-api-key`/`Authorization` que CASOU com um token do Shunt nao e a
    credencial do provedor: a flag manda o dispatcher injetar a chave
    configurada. Sem a flag, o token viraria credencial e seguiria ao provedor."""
    req = _request_with_token_cache({"x-api-key": TEST_SHUNT_TOKEN})
    import asyncio

    token_id = asyncio.run(require_shunt_token(req))
    assert token_id == TEST_SHUNT_TOKEN_ID
    assert req.state.credential_is_token is True


def test_an_explicit_header_token_keeps_credential_is_token_false():
    """O mesmo token apresentado pela forma EXPLICITA (`x-shunt-token`) autentica
    sem marcar a flag: a credencial do cliente, se houver, segue como credencial."""
    req = _request_with_token_cache({"x-shunt-token": TEST_SHUNT_TOKEN})
    import asyncio

    token_id = asyncio.run(require_shunt_token(req))
    assert token_id == TEST_SHUNT_TOKEN_ID
    assert req.state.credential_is_token is False
