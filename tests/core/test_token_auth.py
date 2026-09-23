"""Token do Shunt: leitura das tres formas, hash, mascara e cache de validacao."""
import pytest

from app.core.token_auth import TokenCache, TokenConflict, mask_path, mask_query, read_token, token_hash


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
