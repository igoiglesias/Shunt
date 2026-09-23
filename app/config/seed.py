"""Catálogo de roteamento e seed único no primeiro boot.

O catálogo vive exclusivamente no banco em operação. Este módulo carrega o
catálogo padrão e o insere nas tabelas se estiverem vazias (primeira subida).
As chaves dos provedores vêm das variáveis de ambiente APENAS no momento do
seed; em runtime a chave é lida do banco (coluna Provider.api_key).
"""

import os
from typing import Any

from app.config.settings import ModelCaps, ModelConfig, ProviderConfig, Settings

CATALOG: dict[str, Any] = {
    "providers": {
        "local": {
            "base_url": "http://127.0.0.1:8080/v1",
            "protocol": "openai",
            "api_key_env": "LOCAL_API_KEY",
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "protocol": "openai",
            "api_key_env": "OPENROUTER_API_KEY",
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "protocol": "openai",
            "api_key_env": "GROQ_API_KEY",
        },
    },
    "models": {
        "qwen-local": {
            "provider": "local",
            "model": "qwen3.8-27b",
            "supports": {"tools": True, "streaming": True, "vision": True},
            "context_window": 163840,
            "max_output_tokens": 16384,
        },
        "open-free": {
            "provider": "openrouter",
            "model": "openrouter/free",
            "supports": {"tools": True, "streaming": True, "vision": False},
            "context_window": 262144,
            "max_output_tokens": 8192,
        },
        "open-nemotron-ultra": {
            "provider": "openrouter",
            "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "supports": {"tools": True, "streaming": True, "vision": False},
            "context_window": 262144,
            "max_output_tokens": 16384,
        },
        "open-deepseek-v4.1-flash": {
            "provider": "openrouter",
            "model": "deepseek/deepseek-v4.1-flash",
            "supports": {"tools": True, "streaming": True, "vision": True},
            "context_window": 262144,
            "max_output_tokens": 16384,
        },
        "open-gpt-oss-120": {
            "provider": "openrouter",
            "model": "openai/gpt-oss-120b",
            "supports": {"tools": True, "streaming": True, "vision": False},
            "context_window": 131072,
            "max_output_tokens": 16384,
        },
        "groq-free": {
            "provider": "groq",
            "model": "groq/compound",
            "supports": {"tools": True, "streaming": True, "vision": False},
            "context_window": 131072,
            "max_output_tokens": 8192,
        },
    },
    "routes": [
        ("groq", ["groq-free"]),
        ("local", ["qwen-local"]),
        ("nemotron", ["open-nemotron-ultra"]),
        ("deepseek", ["open-deepseek-v4.1-flash"]),
        ("gpt-oss", ["open-gpt-oss-120"]),
        ("free", ["open-free", "groq-free", "open-nemotron-ultra"]),
        ("fable", ["open-nemotron-ultra", "open-free", "groq-free"]),
        ("haiku", ["groq-free", "open-free", "open-nemotron-ultra"]),
        ("sonnet", ["open-free", "open-nemotron-ultra", "groq-free"]),
        ("opus", ["qwen-local", "open-nemotron-ultra", "open-free"]),
    ],
    "default_model": "open-gpt-oss-120",
}


def catalog_settings() -> Settings:
    """Constrói Settings a partir do catálogo, lendo chaves do env."""
    providers = {}
    for name, data in CATALOG["providers"].items():
        env_name = data["api_key_env"]
        key = os.environ.get(env_name)
        providers[name] = ProviderConfig(
            base_url=data["base_url"],
            protocol=data["protocol"],
            api_key=key,
        )

    models = {}
    for alias, data in CATALOG["models"].items():
        supports_data = data["supports"]
        models[alias] = ModelConfig(
            provider=data["provider"],
            model=data["model"],
            supports=ModelCaps(
                tools=supports_data["tools"],
                streaming=supports_data["streaming"],
                vision=supports_data["vision"],
            ),
            context_window=data["context_window"],
            max_output_tokens=data["max_output_tokens"],
        )

    routes = CATALOG["routes"]
    default_model = CATALOG["default_model"]

    return Settings(
        providers=providers,
        models=models,
        routes=routes,
        default_model=default_model,
    )


def _backfill_legacy_api_keys(session) -> None:
    """Bancos antigos guardam em `api_key_env` o NOME da variavel de ambiente.

    A coluna legacy so existe em schemas antigos -- no banco novo ela nem
    existe, entao a consulta e SQL puro e o teste de coluna vem antes. A
    chave bruta e resolvida UMA vez daqui; em runtime a fonte e `api_key`.
    """
    from sqlalchemy import inspect, text

    bind = session.get_bind()
    columns = {c["name"] for c in inspect(bind).get_columns("providers")}
    if "api_key_env" not in columns:
        return
    rows = session.execute(
        text("SELECT id, api_key_env FROM providers WHERE api_key IS NULL AND api_key_env IS NOT NULL")
    ).fetchall()
    for row in rows:
        key = os.environ.get(row.api_key_env)
        if key:
            session.execute(
                text("UPDATE providers SET api_key = :k WHERE id = :i"),
                {"k": key, "i": row.id},
            )


def seed_catalog_if_empty(engine):
    """Semeia (ou completa) o catalogo padrao no boot. Idempotente.

    Porta: as tres tabelas (providers, models, routes) nao vazias significam
    catalogo dono do operador -- o seed nao toca. Em qualquer outro caso o
    catalogo esta incompleto (por exemplo, um provider criado por QA sem
    rotas: toda requisicao cai em "sem candidato") e as partes padrao
    ausentes entram; linhas existentes sobrevivem por nome de provider e
    alias de modelo.
    """
    from sqlalchemy.orm import Session

    from app.stats.models import Model, Provider, Route, RouteCandidate

    with Session(engine) as session:
        _backfill_legacy_api_keys(session)
        complete = (
            session.query(Provider).count() > 0
            and session.query(Model).count() > 0
            and session.query(Route).count() > 0
        )
        if complete:
            session.commit()
            return

        provider_ids = {p.name: p.id for p in session.query(Provider)}
        for name, data in CATALOG["providers"].items():
            if name in provider_ids:
                continue
            p = Provider(
                name=name,
                base_url=data["base_url"],
                protocol=data["protocol"],
                api_key=os.environ.get(data["api_key_env"]),
            )
            session.add(p)
            session.flush()
            provider_ids[name] = p.id

        existing_aliases = {m.alias for m in session.query(Model)}
        for alias, data in CATALOG["models"].items():
            if alias in existing_aliases:
                continue
            supports = data["supports"]
            session.add(
                Model(
                    alias=alias,
                    provider_id=provider_ids[data["provider"]],
                    upstream_model=data["model"],
                    supports_tools=supports["tools"],
                    supports_streaming=supports["streaming"],
                    supports_vision=supports["vision"],
                    context_window=data["context_window"],
                    max_output_tokens=data["max_output_tokens"],
                    is_default=(alias == CATALOG["default_model"]),
                )
            )

        if session.query(Route).count() == 0:
            for order_idx, (pattern, candidates) in enumerate(CATALOG["routes"]):
                r = Route(pattern=pattern, order_index=order_idx)
                session.add(r)
                session.flush()
                for cand_idx, model_alias in enumerate(candidates):
                    session.add(
                        RouteCandidate(
                            route_id=r.id,
                            model_alias=model_alias,
                            order_index=cand_idx,
                        )
                    )

        session.commit()