"""Fixtures do E2E: o aplicativo real, com o pool apontado ao provedor falso."""

import pytest
from fastapi.testclient import TestClient

from app.config.settings import ModelCaps, ModelConfig, ProviderConfig, Settings
from app.core.upstream import UpstreamPool
from app.main import app as shunt_app
from tests.e2e.fake_provider import FakeProvider, ScriptedTransport

E2E_SETTINGS = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="http://fake.openrouter/v1",
            protocol="openai",
            api_key="fake-or-key",
        ),
        "anthropic": ProviderConfig(
            base_url="http://fake.anthropic",
            protocol="anthropic",
            api_key="fake-ant-key",
        ),
        "local": ProviderConfig(
            base_url="http://fake.local/v1", protocol="openai", api_key="fake-local-key"
        ),
    },
    models={
        "free": ModelConfig(
            provider="openrouter",
            model="vendor/free",
            supports=ModelCaps(tools=True, streaming=True, vision=False),
            context_window=64000,
            max_output_tokens=8192,
        ),
        "qwen": ModelConfig(
            provider="local",
            model="qwen3-8b",
            supports=ModelCaps(tools=True, streaming=True, vision=False),
            context_window=32768,
            max_output_tokens=4096,
        ),
        "mudo": ModelConfig(
            provider="openrouter",
            model="vendor/mudo",
            supports=ModelCaps(tools=False, streaming=True, vision=False),
            context_window=64000,
            max_output_tokens=8192,
        ),
    },
    routes=[("opus", ["qwen", "free"]), ("haiku", ["free"])],
    default_model=None,
)


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def settings():
    return E2E_SETTINGS.model_copy(deep=True)


@pytest.fixture
def shunt(provider, settings):
    shunt_app.state.settings = settings
    shunt_app.state.pool = UpstreamPool(settings, transport=ScriptedTransport(provider))
    with TestClient(shunt_app) as client:
        yield client
