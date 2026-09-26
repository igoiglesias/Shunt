import pytest

from app.config.settings import ModelConfig, ProviderConfig, Settings
from app.core.resolver import UnknownProviderError, last_resort, resolve


def build(routes, default_model=None):
    return Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://openrouter.ai/api/v1", protocol="openai", api_key=None
            ),
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.com", protocol="anthropic", api_key=None
            ),
            "local": ProviderConfig(
                base_url="http://localhost:8080/v1", protocol="openai", api_key=None
            ),
            "openai": ProviderConfig(
                base_url="https://api.openai.com/v1", protocol="openai", api_key=None
            ),
        },
        models={
            "free": ModelConfig(
                provider="openrouter",
                model="vendor/free",
                context_window=64000,
                max_output_tokens=8192,
            ),
            "qwen": ModelConfig(
                provider="local", model="qwen3-8b", context_window=32768, max_output_tokens=4096
            ),
            "cheap": ModelConfig(
                provider="openrouter",
                model="vendor/cheap",
                context_window=64000,
                max_output_tokens=8192,
            ),
        },
        routes=routes,
        default_model=default_model,
    )


def test_exact_match_wins_over_family():
    settings = build([("claude-opus-4-5", ["qwen"]), ("opus", ["free"])])
    result = resolve("claude-opus-4-5", settings)
    assert result.rule == "exact"
    assert [c.alias for c in result.chain] == ["qwen", None]  # o None e o transparente original


def test_family_match_uses_substring():
    settings = build([("opus", ["qwen"])])
    result = resolve("claude-3-opus-20240229", settings)
    assert result.rule == "family"
    assert result.matched == "opus"


def test_first_matching_pattern_wins():
    settings = build([("opus", ["qwen"]), ("claude-opus", ["free"])])
    assert resolve("claude-opus-4-5", settings).chain[0].alias == "qwen"


def test_default_model_closes_the_chain_after_the_route_members():
    """O `default_model` e o ultimo degrau de toda a cadeia, nao o segundo."""
    settings = build([("opus", ["qwen", "free"])], default_model="cheap")
    chain = [c.alias for c in resolve("claude-opus-4-5", settings).chain]
    assert chain == ["qwen", "free", "cheap"]


def test_no_matching_route_falls_back_to_default_rule():
    settings = build([("haiku", ["free"])], default_model="cheap")
    result = resolve("claude-opus-4-5", settings)
    assert result.rule == "default"
    assert [c.alias for c in result.chain] == ["cheap"]


def test_seen_set_dedups_repeated_alias_within_a_route():
    settings = build([("opus", ["free", "cheap", "free"])])
    chain = [c.alias for c in resolve("claude-opus-4-5", settings).chain]
    assert chain == ["free", "cheap", None]  # o None e o transparente original


def test_no_rule_and_no_default_falls_back_to_transparent():
    settings = build([("haiku", ["free"])])
    result = resolve("claude-fable-5-1", settings)
    assert result.rule == "transparent"
    assert result.chain[0].transparent is True
    assert result.chain[0].provider == "anthropic"


def test_transparent_infers_provider_from_model_name():
    settings = build([])
    assert resolve("claude-fable-5-1", settings).chain[0].provider == "anthropic"
    assert resolve("vendor/algum-modelo", settings).chain[0].provider == "openrouter"


def test_transparent_candidate_keeps_the_requested_name_as_the_model():
    settings = build([])
    candidate = resolve("claude-fable-5-1", settings).chain[0]
    assert candidate.model == "claude-fable-5-1"
    assert candidate.protocol == "anthropic"


def test_transparent_infers_openai_provider_from_gpt_and_o_series_prefixes():
    settings = build([])
    assert resolve("gpt-4o", settings).chain[0].provider == "openai"
    assert resolve("o1-preview", settings).chain[0].provider == "openai"
    assert resolve("o3-mini", settings).chain[0].provider == "openai"


def test_transparent_uses_official_host_when_provider_not_declared():
    """Provedor derivado do nome fora do catalogo: o transparente usa o host
    oficial embutido, em vez de levantar (RAIZ DO BUG que matava o harness)."""
    settings = Settings(
        providers={
            "local": ProviderConfig(
                base_url="http://localhost:8080/v1", protocol="openai", api_key=None
            )
        },
        models={},
        routes=[],
        default_model=None,
    )
    result = resolve("claude-fable-5-1", settings)
    assert result.rule == "transparent"
    candidate = result.chain[0]
    assert candidate.transparent is True
    assert candidate.alias is None
    assert candidate.provider == "anthropic"
    assert candidate.protocol == "anthropic"
    assert candidate.model == "claude-fable-5-1"


def test_transparent_still_rejects_a_name_with_no_deducible_provider():
    """Sem prefixo dedutivel nem host oficial do nome, nao existe destino: o
    400 que o chamador ja devolvia continua valendo."""
    settings = Settings(
        providers={
            "local": ProviderConfig(
                base_url="http://localhost:8080/v1", protocol="openai", api_key=None
            )
        },
        models={},
        routes=[],
        default_model=None,
    )
    with pytest.raises(UnknownProviderError, match="um-modelo-qualquer"):
        resolve("um-modelo-qualquer", settings)


def test_unknown_model_with_no_route_and_no_default_is_official_transparent():
    """Caso a do usuario: modelo inexistente, sem rota, sem default e sem
    "anthropic" no catalogo -> transparente no host oficial da Anthropic."""
    settings = Settings(
        providers={
            "local": ProviderConfig(
                base_url="http://localhost:8080/v1", protocol="openai", api_key=None
            ),
            "openrouter": ProviderConfig(
                base_url="https://openrouter.ai/api/v1", protocol="openai", api_key=None
            ),
        },
        models={},
        routes=[],
        default_model=None,
    )
    result = resolve("claude-sonnet-4-5", settings)
    assert result.rule == "transparent"
    candidate = result.chain[0]
    assert candidate.transparent is True
    assert candidate.provider == "anthropic"
    assert candidate.protocol == "anthropic"
    assert candidate.model == "claude-sonnet-4-5"


def test_family_route_tail_is_official_transparent_when_provider_undeclared():
    """Caso b do usuario: rota de familia casando, sem default e sem
    "anthropic" no catalogo -> o ULTIMO da cadeia e o transparente oficial
    com o model pedido."""
    settings = Settings(
        providers={
            "local": ProviderConfig(
                base_url="http://localhost:8080/v1", protocol="openai", api_key=None
            )
        },
        models={
            "qwen": ModelConfig(
                provider="local", model="qwen3-8b", context_window=32768, max_output_tokens=4096
            )
        },
        routes=[("haiku", ["qwen"])],
        default_model=None,
    )
    result = resolve("claude-haiku-4-5-20251001", settings)
    assert result.rule == "family"
    chain = result.chain
    assert chain[0].alias == "qwen"
    last = chain[-1]
    assert last.transparent is True
    assert last.alias is None
    assert last.provider == "anthropic"
    assert last.protocol == "anthropic"
    assert last.model == "claude-haiku-4-5-20251001"


# -- Candidato transparente na cadeia de rota ----------------------------------

def test_settings_constructor_accepts_unconfigured_route_candidate():
    """Settings nao rejeita rota com candidato ausente em models."""
    settings = build([("fable", ["opus"])])
    assert "opus" not in settings.models


def test_route_candidate_not_in_models_becomes_transparent():
    """Alias ausente em models: transparente, model=alias, provider do requested."""
    settings = build([("fable", ["opus"])])
    result = resolve("claude-fable-5-1", settings)
    assert result.chain[0].transparent is True
    assert result.chain[0].model == "opus"
    assert result.chain[0].provider == "anthropic"
    assert result.chain[0].alias is None


def test_mixed_chain_configured_then_transparent():
    """Candidato configurado primeiro, transparente ocupa sua posicao na cadeia."""
    settings = build([("fable", ["qwen", "opus"])])
    chain = resolve("claude-fable-5-1", settings).chain
    assert chain[0].alias == "qwen"
    assert chain[0].transparent is False
    assert chain[1].alias is None
    assert chain[1].transparent is True
    assert chain[1].model == "opus"


def test_transparent_candidate_openai_provider():
    """Requested gpt-* -> provider=openai para alias nao configurado."""
    settings = build([("gpt", ["gpt-4o"])])
    result = resolve("gpt-4o-2024", settings)
    assert result.chain[0].transparent is True
    assert result.chain[0].model == "gpt-4o"
    assert result.chain[0].provider == "openai"


def test_transparent_candidate_no_route_match_falls_back_to_requested_name():
    """Sem rota que casa, fallback transparente usa requested, nao o alias."""
    settings = build([("fable", ["opus"])])
    result = resolve("claude-haiku-4-5", settings)
    assert result.rule == "transparent"
    assert result.chain[0].model == "claude-haiku-4-5"
    assert result.chain[0].provider == "anthropic"


# -- Historia E: quem atende quando nada na cadeia cabe ------------------------


def test_last_resort_is_the_declared_default_model():
    """O `default_model` e a escolha do operador para "quando nada serve", e
    por isso ele e chamado mesmo sem caber."""
    settings = build([("opus", ["free"])], default_model="qwen")
    chain = last_resort("claude-opus-5", settings)
    assert [c.alias for c in chain] == ["qwen"]
    assert chain[0].transparent is False


def test_last_resort_without_a_default_is_the_model_the_harness_asked_for():
    """Sem default declarado, quem pediu resolve com o proprio provedor."""
    settings = build([("opus", ["free"])], default_model=None)
    chain = last_resort("claude-opus-5", settings)
    assert chain[0].transparent is True
    assert chain[0].model == "claude-opus-5"
    assert chain[0].provider == "anthropic"


def test_last_resort_is_empty_when_no_provider_can_be_guessed():
    """Sem default e sem provedor deduzivel nao ha degrau nenhum: o chamador
    devolve o 400 que ja devolvia."""
    settings = build([("opus", ["free"])], default_model=None)
    assert last_resort("um-modelo-qualquer", settings) == []


# -- O degrau de baixo fecha toda a cadeia resolvida ---------------------------


def test_family_chain_ends_with_the_transparent_original_when_no_default():
    """Sem `default_model`, o ultimo degrau e o proprio modelo pedido,
    transparente, com provedor derivado do nome."""
    settings = build([("haiku", ["qwen"])], default_model=None)
    chain = resolve("claude-haiku-4-5-20251001", settings).chain
    assert chain[0].alias == "qwen"
    last = chain[-1]
    assert last.transparent is True
    assert last.alias is None
    assert last.model == "claude-haiku-4-5-20251001"
    assert last.provider == "anthropic"
    assert last.protocol == "anthropic"


def test_family_chain_ends_with_the_default_when_declared():
    """Com `default_model` declarado, ele fecha a cadeia -- nem que nenhum
    alias da rota seja ele, e nao ocupa lugar no meio."""
    settings = build([("opus", ["qwen", "free"])], default_model="cheap")
    chain = resolve("claude-opus-4-5", settings).chain
    assert [c.alias for c in chain] == ["qwen", "free", "cheap"]
    last = chain[-1]
    assert last.provider == "openrouter"
    assert last.model == "vendor/cheap"
    assert last.transparent is False


def test_default_that_is_a_route_member_moved_to_the_end_without_a_copy():
    """O default membro da rota aparece UMA vez, no fim: a copia antiga do
    meio da cadeia some, sem duplicata."""
    settings = build([("opus", ["cheap", "free"])], default_model="cheap")
    chain = [c.alias for c in resolve("claude-opus-4-5", settings).chain]
    assert chain == ["free", "cheap"]
    assert chain.count("cheap") == 1


def test_chain_unchanged_when_no_default_and_no_provider_can_be_guessed():
    """Sem default e sem provedor deduzivel do nome, nao ha ultimo degrau:
    a cadeia da rota sai intacta, sem elemento fantasma."""
    settings = build([("qualquer", ["qwen"])], default_model=None)
    chain = [c.alias for c in resolve("um-modelo-qualquer", settings).chain]
    assert chain == ["qwen"]


def test_declared_route_order_is_kept_before_the_tail():
    """A ordem declarada da rota manda ate o penultimo; so o fim e reservado
    ao degrau de baixo."""
    settings = build([("opus", ["free", "qwen"])], default_model="cheap")
    chain = [c.alias for c in resolve("claude-opus-4-5", settings).chain]
    assert chain == ["free", "qwen", "cheap"]
