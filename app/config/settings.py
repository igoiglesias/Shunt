import os
from typing import Literal

from pydantic import BaseModel, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session


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
    def _check_references(self) -> Settings:
        for alias, model in self.models.items():
            if model.provider not in self.providers:
                raise ValueError(f"model {alias!r} points to unknown provider {model.provider!r}")
        known = set(self.models)
        for pattern, candidates in self.routes:
            if pattern == "":
                raise ValueError(
                    "route pattern must not be empty; it is a substring of every "
                    "model name and would swallow all traffic ahead of every other rule"
                )
            if not candidates:
                raise ValueError(f"route {pattern!r} has an empty candidate list")
        if self.default_model is not None and self.default_model not in known:
            raise ValueError(f"default_model points to unknown model {self.default_model!r}")
        return self

    def api_key(self, provider: str) -> str | None:
        env = self.providers[provider].api_key_env
        return os.environ.get(env) if env else None


def load_settings() -> Settings:
    """Carrega settings do arquivo de configuracao estatico (fallback sem banco)."""
    from dotenv import load_dotenv

    from app.config import config

    load_dotenv()
    return Settings(
        providers={k: ProviderConfig.model_validate(v) for k, v in config.providers.items()},
        models={k: ModelConfig.model_validate(v) for k, v in config.models.items()},
        routes=config.routes,
        default_model=config.default_model,
    )


def load_settings_from_db(session: Session) -> Settings:
    """Carrega settings do banco de dados."""
    from app.stats.models import Model, Provider, Route, RouteCandidate

    providers_db = session.execute(select(Provider).order_by(Provider.id)).scalars().all()
    providers = {
        p.name: ProviderConfig(
            base_url=p.base_url,
            protocol=p.protocol,  # type: ignore[arg-type]
            api_key_env=p.api_key_env,
        )
        for p in providers_db
    }

    models_db = session.execute(select(Model).order_by(Model.id)).scalars().all()
    models = {}
    default_model = None
    for m in models_db:
        models[m.alias] = ModelConfig(
            provider=m.provider.name,
            model=m.upstream_model,
            supports=ModelCaps(
                tools=m.supports_tools,
                streaming=m.supports_streaming,
                vision=m.supports_vision,
            ),
            context_window=m.context_window,
            max_output_tokens=m.max_output_tokens,
        )
        if m.is_default:
            default_model = m.alias

    routes_db = session.execute(select(Route).order_by(Route.order_index)).scalars().all()
    routes = []
    for r in routes_db:
        candidates_db = session.execute(
            select(RouteCandidate)
            .where(RouteCandidate.route_id == r.id)
            .order_by(RouteCandidate.order_index)
        ).scalars().all()
        candidates = [c.model_alias for c in candidates_db]
        routes.append((r.pattern, candidates))

    return Settings(
        providers=providers,
        models=models,
        routes=routes,
        default_model=default_model,
    )
