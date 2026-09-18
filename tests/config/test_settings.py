import pytest
from pydantic import ValidationError

from app.config.settings import ModelConfig, ProviderConfig, Settings


def make_settings(**over):
    base = dict(
        providers={"openrouter": ProviderConfig(
            base_url="https://openrouter.ai/api/v1",
            protocol="openai",
            api_key_env="OPENROUTER_API_KEY",
        )},
        models={"free": ModelConfig(
            provider="openrouter", model="vendor/free",
            context_window=64000, max_output_tokens=8192,
        )},
        routes=[("haiku", ["free"])],
        default_model=None,
    )
    base.update(over)
    return Settings(**base)


def test_model_pointing_to_unknown_provider_fails_at_boot():
    with pytest.raises(ValidationError, match="unknown provider"):
        make_settings(models={"free": ModelConfig(
            provider="nao-existe", model="x",
            context_window=1000, max_output_tokens=100,
        )})


def test_route_pointing_to_unknown_model_fails_at_boot():
    with pytest.raises(ValidationError, match="unknown model"):
        make_settings(routes=[("haiku", ["inexistente"])])


def test_api_key_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-teste")
    assert make_settings().api_key("openrouter") == "sk-or-teste"


def test_provider_without_api_key_env_returns_none():
    settings = make_settings(providers={"local": ProviderConfig(
        base_url="http://localhost:8080/v1", protocol="openai", api_key_env=None,
    )}, models={}, routes=[])
    assert settings.api_key("local") is None
