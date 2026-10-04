"""Testes unitarios de `_missing_credential`: quem pode ser chamado sem chave.

A regra e curta, mas dois dos seus ramos so tinham cobertura E2E:
o candidato transparente pode ser chamado sem credencial de servidor (usa a do
cliente), e o candidato declarado sem `api_key` e pulado com uma mensagem que o
rastro do dispatcher loga.
"""

import httpx
import respx

from app.config.settings import ModelConfig, ProviderConfig, Settings
from app.core.dispatcher import ShuntRequest, _missing_credential, dispatch
from app.core.resolver import Candidate
from app.core.upstream import UpstreamPool

BODY = {
    "model": "opus",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "oi"}],
}


def ok_payload(model):
    return {
        "id": "chatcmpl-1",
        "model": model,
        "choices": [{"message": {"content": "pronto"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _settings(api_key: str | None) -> Settings:
    return Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key=api_key
            ),
        },
        models={
            "free": ModelConfig(
                provider="openrouter",
                model="vendor/free",
                context_window=64000,
                max_output_tokens=8192,
            ),
        },
        routes=[("opus", ["free"])],
        default_model=None,
    )


def _declared_candidate() -> Candidate:
    return Candidate(
        alias="free",
        provider="openrouter",
        model="vendor/free",
        protocol="openai",
    )


def test_a_transparent_candidate_is_never_missing_a_credential():
    # Transparente usa a chave do CLIENTE, nao a do servidor: a ausencia dela
    # nao pode impedir a chamada. E o unico modo em que o proxy mesmo nao tem
    # credencial nenhuma, e a regra de chave ausente nao pode mata-lo.
    candidate = Candidate(
        alias=None,
        provider="openrouter",
        model="vendor/free",
        protocol="openai",
        transparent=True,
    )
    assert _missing_credential(candidate, _settings(None)) is None


def test_a_configured_provider_has_no_missing_credential():
    assert _missing_credential(_declared_candidate(), _settings("sk-configurada")) is None


def test_an_unset_credential_reports_the_provider_by_name():
    missing = _missing_credential(_declared_candidate(), _settings(None))
    assert missing == "provider openrouter sem chave configurada"


def test_an_empty_string_credential_counts_as_unset():
    # `not key` cobre a string vazia, nao so None: uma chave vazia nao existe
    # para ser mandada, e tem o mesmo tratamento de uma coluna nunca populada.
    assert _missing_credential(_declared_candidate(), _settings("")) == (
        "provider openrouter sem chave configurada"
    )


@respx.mock
async def test_the_trace_line_names_the_provider_whose_credential_is_unset():
    # O rastro e a unica evidencia que o operador tem do pulo: a mensagem do
    # `_missing_credential` e aninhada sob "credential ... is not set" e tem
    # que carregar o NOME do provedor, nao so o label do modelo.
    settings = _settings(None)
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_payload("vendor/free"))
    )
    pool = UpstreamPool(settings)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), settings, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 0, "candidato sem chave nao pode ser chamado"
    assert result.trace == [
        "vendor/free: credential provider openrouter sem chave configurada is not set"
    ]
    assert "provider openrouter sem chave configurada" in result.body["error"]["message"]
