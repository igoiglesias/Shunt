import os
from typing import Literal

from pydantic import BaseModel, model_validator


class ProviderConfig(BaseModel):
    base_url: str
    protocol: Literal["openai", "anthropic"]
    api_key_env: str | None = None


class ModelCaps(BaseModel):
    tools: bool = True
    streaming: bool = True
    vision: bool = False


class ModelConfig(BaseModel):
    provider: str
    model: str
    supports: ModelCaps = ModelCaps()
    context_window: int
    max_output_tokens: int


class Settings(BaseModel):
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    routes: list[tuple[str, list[str]]]
    default_model: str | None = None

    @model_validator(mode="after")
    def _check_references(self) -> "Settings":
        for alias, model in self.models.items():
            if model.provider not in self.providers:
                raise ValueError(f"model {alias!r} points to unknown provider {model.provider!r}")
        known = set(self.models)
        for pattern, candidates in self.routes:
            for candidate in candidates:
                if candidate not in known:
                    raise ValueError(f"route {pattern!r} points to unknown model {candidate!r}")
        if self.default_model is not None and self.default_model not in known:
            raise ValueError(f"default_model points to unknown model {self.default_model!r}")
        return self

    def api_key(self, provider: str) -> str | None:
        env = self.providers[provider].api_key_env
        return os.environ.get(env) if env else None


def load_settings() -> Settings:
    from dotenv import load_dotenv

    from app.config import config

    load_dotenv()
    return Settings(
        providers={k: ProviderConfig(**v) for k, v in config.providers.items()},
        models={k: ModelConfig(**v) for k, v in config.models.items()},
        routes=config.routes,
        default_model=config.default_model,
    )
