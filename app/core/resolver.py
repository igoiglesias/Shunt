from dataclasses import dataclass

from app.config.settings import Settings

# prefixo do nome do modelo -> nome do provedor esperado na tabela `providers`
PROVIDER_HINTS: tuple[tuple[str, str], ...] = (
    ("claude-", "anthropic"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
)


class UnknownProviderError(Exception):
    pass


@dataclass(frozen=True)
class Candidate:
    alias: str | None
    provider: str
    model: str
    protocol: str
    transparent: bool = False


@dataclass(frozen=True)
class Resolution:
    rule: str
    matched: str | None
    chain: list[Candidate]


def _candidate(alias: str, settings: Settings) -> Candidate:
    model = settings.models[alias]
    return Candidate(
        alias=alias,
        provider=model.provider,
        model=model.model,
        protocol=settings.providers[model.provider].protocol,
    )


def _transparent(requested: str, settings: Settings) -> Candidate:
    provider = None
    for prefix, name in PROVIDER_HINTS:
        if requested.startswith(prefix):
            provider = name
            break
    if provider is None and "/" in requested:
        provider = "openrouter"
    if provider is None or provider not in settings.providers:
        raise UnknownProviderError(
            f"no provider declared for model {requested!r}; "
            f"add it to `providers` or create a route for it"
        )
    return Candidate(alias=None, provider=provider, model=requested,
                     protocol=settings.providers[provider].protocol, transparent=True)


def _chain(aliases: list[str], settings: Settings) -> list[Candidate]:
    ordered = list(aliases)
    if settings.default_model and settings.default_model not in ordered[:1]:
        ordered = ordered[:1] + [settings.default_model] + [
            a for a in ordered[1:] if a != settings.default_model
        ]
    seen: set[str] = set()
    result = []
    for alias in ordered:
        if alias not in seen:
            seen.add(alias)
            result.append(_candidate(alias, settings))
    return result


def resolve(requested: str, settings: Settings) -> Resolution:
    for pattern, aliases in settings.routes:
        if pattern == requested:
            return Resolution("exact", pattern, _chain(aliases, settings))
    for pattern, aliases in settings.routes:
        if pattern in requested:
            return Resolution("family", pattern, _chain(aliases, settings))
    if settings.default_model:
        return Resolution("default", None, _chain([settings.default_model], settings))
    return Resolution("transparent", None, [_transparent(requested, settings)])
