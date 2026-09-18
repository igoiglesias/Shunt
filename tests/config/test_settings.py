import pytest
from pydantic import ValidationError

from app.config.settings import ModelConfig, ProviderConfig, Settings, load_settings


def make_settings(**over):
    base = {
        "providers": {
            "openrouter": ProviderConfig(
                base_url="https://openrouter.ai/api/v1",
                protocol="openai",
                api_key_env="OPENROUTER_API_KEY",
            )
        },
        "models": {
            "free": ModelConfig(
                provider="openrouter",
                model="vendor/free",
                context_window=64000,
                max_output_tokens=8192,
            )
        },
        "routes": [("haiku", ["free"])],
        "default_model": None,
    }
    base.update(over)
    return Settings(**base)


def test_model_pointing_to_unknown_provider_fails_at_boot():
    with pytest.raises(ValidationError, match="unknown provider"):
        make_settings(
            models={
                "free": ModelConfig(
                    provider="nao-existe",
                    model="x",
                    context_window=1000,
                    max_output_tokens=100,
                )
            }
        )


def test_route_pointing_to_unknown_model_fails_at_boot():
    with pytest.raises(ValidationError, match="unknown model"):
        make_settings(routes=[("haiku", ["inexistente"])])


def test_default_model_pointing_to_unknown_model_fails_at_boot():
    with pytest.raises(ValidationError, match="default_model"):
        make_settings(default_model="inexistente")


def test_load_settings_builds_valid_settings_from_the_real_config():
    settings = load_settings()
    assert "free" in settings.models
    assert settings.models["free"].provider in settings.providers


def test_load_settings_does_not_swallow_the_validator(monkeypatch):
    from app.config import config

    monkeypatch.setattr(config, "routes", [("haiku", ["inexistente"])])
    with pytest.raises(ValidationError, match="unknown model"):
        load_settings()


def test_route_with_empty_pattern_fails_at_boot():
    with pytest.raises(ValidationError, match="empty"):
        make_settings(routes=[("", ["free"])])


def test_route_with_empty_candidate_list_fails_at_boot():
    with pytest.raises(ValidationError, match="empty candidate list"):
        make_settings(routes=[("haiku", [])])


def test_api_key_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-teste")
    assert make_settings().api_key("openrouter") == "sk-or-teste"


def test_provider_without_api_key_env_returns_none():
    settings = make_settings(
        providers={
            "local": ProviderConfig(
                base_url="http://localhost:8080/v1",
                protocol="openai",
                api_key_env=None,
            )
        },
        models={},
        routes=[],
    )
    assert settings.api_key("local") is None
