"""Validacao de referencias do `Settings` (`_check_references`).

A validacao confere a referencia contra o que existe no proprio objeto: um
modelo que aponta para provedor ausente, um padrao de rota vazio (que
engoliria todo o trafego) e um `default_model` inexistente. Cada guarda
levanta `ValueError` na construcao, antes de qualquer requisicao chegar ao
dispatcher.
"""

import pytest

from app.config.settings import ModelConfig, ProviderConfig, Settings


def _provider() -> ProviderConfig:
    return ProviderConfig(base_url="http://x", protocol="openai", api_key=None)


def _model(provider: str) -> ModelConfig:
    return ModelConfig(
        provider=provider, model="x", context_window=8192, max_output_tokens=1024
    )


def test_settings_unknown_provider_raises():
    """Model apontando para provedor inexistente levanta ValueError."""
    with pytest.raises(ValueError, match="unknown provider"):
        Settings(
            providers={"p1": _provider()},
            models={"m1": _model("unknown")},
            routes=[],
        )


def test_settings_empty_route_pattern_raises():
    """Pattern de rota vazio levanta ValueError (casa todo modelo)."""
    with pytest.raises(ValueError, match="must not be empty"):
        Settings(
            providers={"p1": _provider()},
            models={},
            routes=[("", ["m1"])],
        )


def test_settings_route_empty_candidates_raises():
    """Rota com lista de candidatos vazia levanta ValueError."""
    with pytest.raises(ValueError, match="empty candidate list"):
        Settings(
            providers={"p1": _provider()},
            models={},
            routes=[("pattern", [])],
        )


def test_settings_unknown_default_model_raises():
    """default_model apontando para modelo inexistente levanta ValueError."""
    with pytest.raises(ValueError, match="unknown model"):
        Settings(
            providers={"p1": _provider()},
            models={"m1": _model("p1")},
            routes=[],
            default_model="unknown",
        )


def test_settings_api_key_returns_the_provider_key():
    """`api_key` le a chave bruta do provedor no catalogo (do banco)."""
    settings = Settings(
        providers={
            "p1": ProviderConfig(
                base_url="http://x", protocol="openai", api_key="sk-bruta"
            )
        },
        models={},
        routes=[],
    )
    assert settings.api_key("p1") == "sk-bruta"
