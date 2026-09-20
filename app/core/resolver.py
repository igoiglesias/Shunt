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


def _candidate(alias: str, settings: Settings, requested: str) -> Candidate:
    """Candidato configurado em models; transparente (alias como model ID) quando ausente."""
    if alias in settings.models:
        model = settings.models[alias]
        return Candidate(
            alias=alias,
            provider=model.provider,
            model=model.model,
            protocol=settings.providers[model.provider].protocol,
        )
    # alias ausente em models: transparente, model=alias, provider derivado do requested
    return _transparent(requested, settings, model=alias)


def _transparent(requested: str, settings: Settings, model: str | None = None) -> Candidate:
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
    return Candidate(
        alias=None,
        provider=provider,
        model=model if model is not None else requested,
        protocol=settings.providers[provider].protocol,
        transparent=True,
    )


def _chain(aliases: list[str], settings: Settings, requested: str) -> list[Candidate]:
    ordered = list(aliases)
    if settings.default_model:
        ordered = (
            ordered[:1]
            + [settings.default_model]
            + [a for a in ordered[1:] if a != settings.default_model]
        )
    seen: set[str] = set()
    result = []
    for alias in ordered:
        if alias not in seen:
            seen.add(alias)
            result.append(_candidate(alias, settings, requested))
    return result


def last_resort(requested: str, settings: Settings) -> list[Candidate]:
    """O degrau de baixo: quem atende quando nenhum candidato da cadeia cabe.

    A ordem e a que o operador declarou:

    1. o `default_model`, mesmo que ele tambem nao caiba -- e a escolha escrita
       para "quando nada serve";
    2. sem `default_model`, o modelo que o harness pediu, em modo transparente:
       se ninguem no catalogo da conta, quem pediu resolve com o proprio
       provedor.

    Lista vazia quando nao ha nem um nem outro: ai o chamador devolve o erro
    que ja devolvia, e nao uma chamada a esmo.
    """
    if settings.default_model:
        return [_candidate(settings.default_model, settings, requested)]
    try:
        return [_transparent(requested, settings)]
    except UnknownProviderError:
        return []


def resolve(requested: str, settings: Settings) -> Resolution:
    for pattern, aliases in settings.routes:
        if pattern == requested:
            return Resolution("exact", pattern, _chain(aliases, settings, requested))
    for pattern, aliases in settings.routes:
        if pattern in requested:
            return Resolution("family", pattern, _chain(aliases, settings, requested))
    if settings.default_model:
        return Resolution("default", None, _chain([settings.default_model], settings, requested))
    return Resolution("transparent", None, [_transparent(requested, settings)])
