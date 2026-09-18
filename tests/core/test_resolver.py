import pytest

from app.config.settings import ModelConfig, ProviderConfig, Settings
from app.core.resolver import UnknownProviderError, resolve


def build(routes, default_model=None):
    return Settings(
        providers={
            "openrouter": ProviderConfig(base_url="https://openrouter.ai/api/v1",
                                         protocol="openai", api_key_env=None),
            "anthropic": ProviderConfig(base_url="https://api.anthropic.com",
                                        protocol="anthropic", api_key_env=None),
            "local": ProviderConfig(base_url="http://localhost:8080/v1",
                                    protocol="openai", api_key_env=None),
        },
        models={
            "free": ModelConfig(provider="openrouter", model="vendor/free",
                                context_window=64000, max_output_tokens=8192),
            "qwen": ModelConfig(provider="local", model="qwen3-8b",
                                context_window=32768, max_output_tokens=4096),
            "cheap": ModelConfig(provider="openrouter", model="vendor/cheap",
                                 context_window=64000, max_output_tokens=8192),
        },
        routes=routes,
        default_model=default_model,
    )


def test_exact_match_wins_over_family():
    settings = build([("claude-opus-4-5", ["qwen"]), ("opus", ["free"])])
    result = resolve("claude-opus-4-5", settings)
    assert result.rule == "exact"
    assert [c.alias for c in result.chain] == ["qwen"]


def test_family_match_uses_substring():
    settings = build([("opus", ["qwen"])])
    result = resolve("claude-3-opus-20240229", settings)
    assert result.rule == "family"
    assert result.matched == "opus"


def test_first_matching_pattern_wins():
    settings = build([("opus", ["qwen"]), ("claude-opus", ["free"])])
    assert resolve("claude-opus-4-5", settings).chain[0].alias == "qwen"


def test_default_model_comes_right_after_the_rule_candidate():
    settings = build([("opus", ["qwen", "free"])], default_model="cheap")
    chain = [c.alias for c in resolve("claude-opus-4-5", settings).chain]
    assert chain == ["qwen", "cheap", "free"]


def test_no_matching_route_falls_back_to_default_rule():
    settings = build([("haiku", ["free"])], default_model="cheap")
    result = resolve("claude-opus-4-5", settings)
    assert result.rule == "default"
    assert [c.alias for c in result.chain] == ["cheap"]


def test_chain_deduplicates_candidate_that_equals_the_default_model():
    settings = build([("opus", ["cheap", "free"])], default_model="cheap")
    chain = [c.alias for c in resolve("claude-opus-4-5", settings).chain]
    assert chain == ["cheap", "free"]


def test_no_rule_and_no_default_falls_back_to_transparent():
    settings = build([("haiku", ["free"])])
    result = resolve("claude-fable-5-1", settings)
    assert result.rule == "transparent"
    assert result.chain[0].transparent is True
    assert result.chain[0].provider == "anthropic"


def test_transparent_infers_provider_from_model_name():
    settings = build([])
    assert resolve("claude-fable-5-1", settings).chain[0].provider == "anthropic"
    assert resolve("vendor/algum-modelo", settings).chain[0].provider == "openrouter"


def test_transparent_candidate_keeps_the_requested_name_as_the_model():
    settings = build([])
    candidate = resolve("claude-fable-5-1", settings).chain[0]
    assert candidate.model == "claude-fable-5-1"
    assert candidate.protocol == "anthropic"


def test_transparent_rejects_name_with_no_declared_provider():
    settings = Settings(
        providers={"local": ProviderConfig(base_url="http://localhost:8080/v1",
                                           protocol="openai", api_key_env=None)},
        models={}, routes=[], default_model=None,
    )
    with pytest.raises(UnknownProviderError, match="claude-fable-5-1"):
        resolve("claude-fable-5-1", settings)
