from dataclasses import dataclass

from app.config.settings import Settings
from app.core.official_hosts import OFFICIAL_HOSTS

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
    """Destino transparente do nome pedido.

    Medida (bug do harness): o seed nao declara "anthropic" nem "openai", e um
    Claude Code com `claude-*` sem rota morria em `UnknownProviderError` -- o
    erro que a spec R1 proibia, porque o destino de um transparente SEM
    entrada no catalogo e o host oficial embutido em codigo, nao o catalogo.
    Provider declarado continua valendo: e a escolha do operador.
    """
    provider = None
    for prefix, name in PROVIDER_HINTS:
        if requested.startswith(prefix):
            provider = name
            break
    if provider is None and "/" in requested:
        provider = "openrouter"
    if provider is None:
        raise UnknownProviderError(
            f"no provider declared for model {requested!r}; "
            f"add it to `providers` or create a route for it"
        )
    if provider in settings.providers:
        protocol = settings.providers[provider].protocol
    else:
        official = OFFICIAL_HOSTS.get(provider)
        if official is None:
            raise UnknownProviderError(
                f"no provider declared for model {requested!r}; "
                f"add it to `providers` or create a route for it"
            )
        protocol = official.protocol
    return Candidate(
        alias=None,
        provider=provider,
        model=model if model is not None else requested,
        protocol=protocol,
        transparent=True,
    )


def _chain(aliases: list[str], settings: Settings, requested: str) -> list[Candidate]:
    """Cadeia da rota mais o degrau de baixo, que sempre fecha a cadeia.

    O degrau de baixo vem de `last_resort` e aparece UMA vez, no fim: se ele
    ja esta na rota (com ou sem alias), a copia antiga some do lugar em que
    estava, em vez de duplicar. Sem default e sem provedor deduzivel do nome,
    nao ha degrau: a cadeia da rota sai como foi declarada.
    """
    tail = last_resort(requested, settings)
    tail_ids = {(c.alias, c.model) for c in tail}
    result: list[Candidate] = []
    seen: set[str] = set()
    for alias in aliases:
        if alias in seen:
            continue
        seen.add(alias)
        candidate = _candidate(alias, settings, requested)
        if (candidate.alias, candidate.model) in tail_ids:
            continue  # o degrau de baixo ja o atende; ele mora no fim
        result.append(candidate)
    result.extend(tail)
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
