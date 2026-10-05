import json
import re

import httpx
import pytest
import respx
from pydantic import ValidationError

from app.config.settings import ModelCaps, ModelConfig, ProviderConfig, Settings
from app.core import dispatcher
from app.core.dispatcher import ShuntRequest, _exception_text, dispatch, outbound_headers
from app.core.resolver import Candidate
from app.core.upstream import UpstreamPool

SETTINGS = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
        )
    },
    models={
        "free": ModelConfig(
            provider="openrouter", model="vendor/free", context_window=64000, max_output_tokens=8192
        ),
        "cheap": ModelConfig(
            provider="openrouter",
            model="vendor/cheap",
            context_window=64000,
            max_output_tokens=8192,
        ),
    },
    routes=[("opus", ["free", "cheap"])],
    default_model=None,
)

BODY = {
    "model": "claude-opus-4-5",
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


@respx.mock
async def test_first_candidate_answers_and_response_is_translated():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_payload("vendor/free"))
    )
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 200
    assert result.body["content"] == [{"type": "text", "text": "pronto"}]
    assert result.body["model"] == "claude-opus-4-5"
    assert result.real_model == "vendor/free"


@respx.mock
async def test_400_skips_to_the_next_candidate_without_retrying():
    route = respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "contexto"}}),
            httpx.Response(200, json=ok_payload("vendor/cheap")),
        ]
    )
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 2
    assert result.real_model == "vendor/cheap"


@respx.mock
async def test_400_trace_line_carries_the_upstream_reason():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            400, json={"error": {"message": "max_tokens exceeds the model limit"}}
        )
    )
    pool = UpstreamPool(SETTINGS)
    body = {**BODY, "model": "my-opus"}
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 400
    # a linha de rastro nao pode dizer so "400": o motivo que o provedor devolveu
    # e o que diagnostica a recusa, e ele ja esta na mao (last_message).
    joined = " | ".join(result.trace)
    assert "max_tokens exceeds the model limit" in joined
    assert any("400" in line and "max_tokens exceeds" in line for line in result.trace)


@respx.mock
async def test_transport_error_retries_the_same_candidate():
    route = respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.ConnectError("recusou"),
            httpx.Response(200, json=ok_payload("vendor/free")),
        ]
    )
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 2
    assert result.real_model == "vendor/free"


@respx.mock
async def test_openai_in_anthropic_out_translates_both_ways():
    settings = Settings(
        providers={
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.test", protocol="anthropic", api_key="sk-teste"
            )
        },
        models={
            "fable": ModelConfig(
                provider="anthropic",
                model="claude-fable-5-1",
                context_window=200000,
                max_output_tokens=8192,
            )
        },
        routes=[("gpt-4o", ["fable"])],
        default_model=None,
    )
    respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-fable-5-1",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "pronto"}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
            },
        )
    )
    pool = UpstreamPool(settings)
    try:
        result = await dispatch(
            ShuntRequest(
                "openai", {"model": "gpt-4o", "messages": [{"role": "user", "content": "oi"}]}, {}
            ),
            settings,
            pool,
        )
    finally:
        await pool.aclose()
    assert result.body["object"] == "chat.completion"
    assert result.body["choices"][0]["message"]["content"] == "pronto"
    assert result.body["model"] == "gpt-4o"


@respx.mock
async def test_exhausted_chain_returns_the_last_error_and_names_every_candidate():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "nao deu"}})
    )
    pool = UpstreamPool(SETTINGS)
    body = {**BODY, "model": "my-opus"}
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 400
    assert result.body["type"] == "error"
    assert result.body["error"]["type"] == "invalid_request_error"
    assert "free" in result.body["error"]["message"]
    assert "cheap" in result.body["error"]["message"]


# --- credenciais: o ponto mais sensível do proxy -----------------------------

KEYED = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key="sk-da-config"
        )
    },
    models={
        "free": ModelConfig(
            provider="openrouter", model="vendor/free", context_window=64000, max_output_tokens=8192
        )
    },
    routes=[("opus", ["free"])],
    default_model=None,
)

TRANSPARENT = Settings(
    providers={
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test",
            protocol="anthropic",
            api_key="sk-da-config",
        )
    },
    models={},
    routes=[],
    default_model=None,
)

CLIENT_HEADERS = {
    "authorization": "Bearer oauth-da-assinatura",
    "x-api-key": "sk-do-cliente",
    "anthropic-beta": "oauth-2026-01-01",
    "anthropic-version": "2023-06-01",
    "host": "localhost:8080",
    "content-length": "999",
    "accept-encoding": "gzip",
}


@respx.mock
async def test_routed_mode_never_forwards_any_inbound_header(monkeypatch):
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_payload("vendor/free"))
    )
    pool = UpstreamPool(KEYED)
    try:
        await dispatch(ShuntRequest("anthropic", BODY, dict(CLIENT_HEADERS)), KEYED, pool)
    finally:
        await pool.aclose()
    sent = route.calls[0].request.headers
    assert sent["authorization"] == "Bearer sk-da-config"
    assert "x-api-key" not in sent
    assert "anthropic-beta" not in sent
    assert "sk-do-cliente" not in str(dict(sent))
    assert "oauth-da-assinatura" not in str(dict(sent))


@respx.mock
async def test_transparent_mode_passes_inbound_headers_verbatim_minus_three(monkeypatch):
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-sonnet-9",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            },
        )
    )
    pool = UpstreamPool(TRANSPARENT)
    body = {**BODY, "model": "claude-sonnet-9"}
    try:
        result = await dispatch(
            ShuntRequest("anthropic", body, dict(CLIENT_HEADERS)), TRANSPARENT, pool
        )
    finally:
        await pool.aclose()
    sent = route.calls[0].request.headers
    assert sent["authorization"] == "Bearer oauth-da-assinatura"
    assert sent["x-api-key"] == "sk-do-cliente"
    assert sent["anthropic-beta"] == "oauth-2026-01-01"
    assert sent["anthropic-version"] == "2023-06-01"
    assert sent["host"] == "api.anthropic.test"
    assert sent["content-length"] != "999"
    assert "sk-da-config" not in str(dict(sent))
    assert result.status == 200
    assert result.real_model == "claude-sonnet-9"


def test_outbound_headers_routed_uses_x_api_key_for_an_anthropic_provider(monkeypatch):
    candidate = Candidate(alias=None, provider="anthropic", model="m", protocol="anthropic")
    headers = outbound_headers(
        ShuntRequest("anthropic", BODY, dict(CLIENT_HEADERS)), candidate, TRANSPARENT
    )
    assert headers["x-api-key"] == "sk-da-config"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in headers
    assert headers["content-type"] == "application/json"


def test_outbound_headers_routed_without_a_configured_key_sends_no_credential():
    """Sem chave configurada nao ha credencial para enviar: o rastro de
    cabecalhos fica com o minimo."""
    keyless = Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key=None
            )
        },
        models={},
        routes=[],
        default_model=None,
    )
    candidate = Candidate(
        alias="free", provider="openrouter", model="vendor/free", protocol="openai"
    )
    headers = outbound_headers(
        ShuntRequest("anthropic", BODY, dict(CLIENT_HEADERS)), candidate, keyless
    )
    assert headers == {"content-type": "application/json"}


def test_outbound_headers_transparent_drops_exactly_the_listed_headers():
    candidate = Candidate(
        alias=None, provider="anthropic", model="m", protocol="anthropic", transparent=True
    )
    headers = outbound_headers(
        ShuntRequest(
            "anthropic",
            BODY,
            {
                "Host": "x",
                "Content-Length": "1",
                "Accept-Encoding": "gzip",
                "X-Api-Key": "sk",
                "X-Shunt-Token": "tok",
                "Cookie": "shunt_admin=abc",
            },
        ),
        candidate,
        TRANSPARENT,
    )
    # x-shunt-token e cookie NAO saem; a credencial do cliente (x-api-key) sai.
    assert headers == {"X-Api-Key": "sk"}


def test_transport_credentials_in_authorization_alone_stay_the_transparent_credential():
    """So `authorization` na requisicao (OAuth de assinatura, sem `x-api-key`):
    continua sendo a credencial do transparente, e a chave configurada nao
    substitui."""
    candidate = Candidate(
        alias=None, provider="anthropic", model="m", protocol="anthropic", transparent=True
    )
    headers = outbound_headers(
        ShuntRequest("anthropic", BODY, {"authorization": "Bearer oauth-x"}), candidate, TRANSPARENT
    )
    assert headers["authorization"] == "Bearer oauth-x"
    assert "x-api-key" not in headers


# --- transparente com provedor FORA do catalogo: o host oficial embutido -----
#
# O seed nao declara "anthropic" nem "openai". Antes do mapa embutido, o
# candidato transparente `claude-*` levantava KeyError em
# `settings.providers[candidate.provider]` e o harness recebia um 500.

NO_ANTHROPIC_DECLARED = Settings(
    providers={
        "local": ProviderConfig(
            base_url="http://localhost:8080/v1", protocol="openai", api_key=None
        )
    },
    models={},
    routes=[],
    default_model=None,
)

OFFICIAL_ANTHROPIC_OK = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "ok"}],
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def test_provider_config_helper_falls_back_to_the_embedded_official_host():
    """Candidato transparente com provider fora do catalogo: o helper devolve o
    ProviderConfig do host oficial embutido (base_url certa, sem chave)."""
    from app.core.dispatcher import _provider_config

    candidate = Candidate(
        alias=None, provider="anthropic", model="claude-haiku-4-5", protocol="anthropic",
        transparent=True,
    )
    config = _provider_config(candidate, NO_ANTHROPIC_DECLARED)
    # Medido: o base do Anthropic fica SEM /v1 porque `PATHS` ja carrega o
    # prefixo no path (`/v1/messages`); com /v1 no base o httpx 0.28 fazia
    # `/v1/v1/messages`. O resultado e o host oficial certo na URL final.
    assert config.base_url == "https://api.anthropic.com"
    assert config.protocol == "anthropic"
    assert config.api_key is None


def test_provider_config_helper_prefers_the_declared_entry():
    """Provider declarado no catalogo: a entrada do operador manda, o mapa
    embutido nem olha."""
    from app.core.dispatcher import _provider_config

    candidate = Candidate(
        alias=None, provider="anthropic", model="claude-a", protocol="anthropic",
        transparent=True,
    )
    config = _provider_config(candidate, TRANSPARENT)
    assert config.base_url == "https://api.anthropic.test"
    assert config.api_key == "sk-da-config"


def test_provider_config_helper_raises_for_a_provider_outside_catalog_and_officials():
    """Linha 315: transparente cujo provider NAO esta no catalogo NEM no mapa
    de hosts oficiais embutido nao tem destino -- o helper levanta `KeyError`
    com o nome do provedor, em vez de inventar uma URL. (O dispatcher so chega
    aqui com um provedor declarado ou derivavel; este e o guard do ultimo
    `raise`.)"""
    from app.core.dispatcher import _provider_config

    candidate = Candidate(
        alias=None, provider="groq", model="vendor/x", protocol="openai", transparent=True
    )
    # "groq" nao esta em NO_ANTHROPIC_DECLARED.providers nem em OFFICIAL_HOSTS
    with pytest.raises(KeyError) as exc:
        _provider_config(candidate, NO_ANTHROPIC_DECLARED)
    assert exc.value.args == ("groq",)


def test_the_missing_credential_note_is_the_provider_without_a_key():
    """Linha 384: candidato declarado (nao transparente) cujo provedor nao tem
    chave configurada -- o rastro nomeia o provedor, em vez de uma mensagem
    generica."""
    from app.core.dispatcher import _missing_credential

    settings = Settings(
        providers={
            "local": ProviderConfig(
                base_url="http://localhost:8080/v1", protocol="openai", api_key=None
            )
        },
        models={
            "local-m": ModelConfig(
                provider="local", model="local/m", context_window=64000, max_output_tokens=8192
            )
        },
        routes=[],
        default_model=None,
    )
    candidate = Candidate(alias="local-m", provider="local", model="local/m", protocol="openai")
    note = _missing_credential(candidate, settings)
    assert note == "provider local sem chave configurada"


def test_tools_offered_skips_non_dict_and_unnamed_entries():
    """A lista de ferramentas pode vir malformada: entrada nao-dict e nome ausente
    sao ignorados, e os dois dialetos (top-level e `function.name`) leem o nome."""
    from app.core.dispatcher import tools_offered

    req = ShuntRequest(
        "anthropic",
        {
            "model": "claude-opus-4-5",
            "tools": [
                "lixo-que-nao-e-dict",
                None,
                {"name": "grep", "input_schema": {}},
                {"type": "function", "function": {"name": "bash", "parameters": {}}},
                {"type": "function", "function": {}},
                {},
            ],
        },
        {},
    )
    assert tools_offered(req) == ["grep", "bash"]


def test_tools_offered_is_empty_without_a_tools_key():
    from app.core.dispatcher import tools_offered

    req = ShuntRequest("anthropic", {**BODY}, {})
    assert tools_offered(req) == []


def test_the_tools_called_and_thinking_are_read_from_anthropic_bodies():
    """Corpo Anthropic: blocos `tool_use` (nome str), `thinking` e entrada
    malformada (nao-dict, nome ausente) no mesmo `content`."""
    from app.core.dispatcher import _tools_called

    body = {
        "content": [
            "lixo-que-nao-e-dict",
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "name": "grep"},
            {"type": "tool_use"},
            {"type": "thinking", "thinking": "..."},
        ]
    }
    assert _tools_called(body) == (["grep"], 1)


def test_the_tools_called_and_thinking_are_read_from_openai_bodies():
    """Corpo OpenAI: `tool_calls` e `reasoning_content` dentro de cada choice,
    com choice sem message e `function` malformada ignorados."""
    from app.core.dispatcher import _tools_called

    body = {
        "choices": [
            {"message": "lixo-que-nao-e-dict"},
            {
                "message": {
                    "reasoning_content": "...",
                    "tool_calls": [
                        {"function": {"name": "bash"}},
                        {"function": None},
                        {},
                        {"function": {}},
                    ],
                }
            },
        ]
    }
    assert _tools_called(body) == (["bash"], 1)


@respx.mock
async def test_transparent_claude_request_reaches_the_official_anthropic_host():
    """E2E no nivel do dispatcher: settings SEM "anthropic", candidato
    transparente claude- -> a chamada sai para https://api.anthropic.com/v1
    com os headers do cliente verbatim, sem 500."""
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=OFFICIAL_ANTHROPIC_OK)
    )
    body = {**BODY, "model": "claude-haiku-4-5"}
    pool = UpstreamPool(NO_ANTHROPIC_DECLARED)
    try:
        result = await dispatch(
            ShuntRequest("anthropic", body, dict(CLIENT_HEADERS)),
            NO_ANTHROPIC_DECLARED,
            pool,
        )
    finally:
        await pool.aclose()
    assert route.call_count == 1
    sent = route.calls[0].request
    assert str(sent.url) == "https://api.anthropic.com/v1/messages"
    assert sent.headers["x-api-key"] == "sk-do-cliente"
    assert sent.headers["authorization"] == "Bearer oauth-da-assinatura"
    assert result.status == 200
    assert result.real_model == "claude-haiku-4-5"
    assert result.body["content"] == [{"type": "text", "text": "ok"}]


# --- T4: token do Shunt nao e credencial de provedor -------------------------
# --- T5: a credencial do harness so vale no destino nativo (spec R2) --------


def test_transparent_skip_reason_is_the_shunt_token_when_it_was_the_only_credential():
    """T4(a): a unica credencial apresentada era o token do Shunt -- nao ha
    chave de provedor que sair, e o candidato transparente e pulado com motivo."""
    from app.core.dispatcher import _transparent_skip_reason

    candidate = Candidate(
        alias=None, provider="anthropic", model="claude-sonnet-5", protocol="anthropic",
        transparent=True,
    )
    req = ShuntRequest(
        "anthropic", BODY, {"x-api-key": "token-do-shunt"}, credential_is_token=True
    )
    reason = _transparent_skip_reason(candidate, req, NO_ANTHROPIC_DECLARED)
    assert reason is not None
    assert "shunt token" in reason


def test_transparent_is_eligible_when_the_provider_is_declared_with_a_key():
    """T4(c): a unica credencial era o token do Shunt, MAS o provedor e
    declarado com chave proprio -- aquela substitui, a chamada segue e nao ha
    pulo (o valor do token fica com o proxy)."""
    from app.core.dispatcher import _transparent_skip_reason

    settings = Settings(
        providers={
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.test", protocol="anthropic", api_key="sk-do-provedor"
            )
        },
        models={},
        routes=[],
        default_model=None,
    )
    candidate = Candidate(
        alias=None, provider="anthropic", model="claude-sonnet-5", protocol="anthropic",
        transparent=True,
    )
    req = ShuntRequest(
        "anthropic", BODY, {"x-api-key": "token-do-shunt"}, credential_is_token=True
    )
    assert _transparent_skip_reason(candidate, req, settings) is None


def test_transparent_skip_reason_is_wrong_destination_for_a_foreign_protocol():
    """T5: Claude Code (protocolo anthropic) pedindo gpt-*: a credencial
    anthropic nao serve no destino openai -- candidato in elegivel."""
    from app.core.dispatcher import _transparent_skip_reason

    candidate = Candidate(
        alias=None, provider="openai", model="gpt-5", protocol="openai", transparent=True
    )
    req = ShuntRequest("anthropic", BODY, {"x-api-key": "sk-do-cliente"})
    reason = _transparent_skip_reason(candidate, req, NO_ANTHROPIC_DECLARED)
    assert reason is not None
    assert "openai" in reason  # o destino nativo da credencial entra no motivo


def test_transparent_is_eligible_in_its_native_destination_with_a_real_credential():
    from app.core.dispatcher import _transparent_skip_reason

    candidate = Candidate(
        alias=None, provider="anthropic", model="claude-sonnet-5", protocol="anthropic",
        transparent=True,
    )
    req = ShuntRequest("anthropic", BODY, {"x-api-key": "sk-do-cliente"})
    assert _transparent_skip_reason(candidate, req, NO_ANTHROPIC_DECLARED) is None


def test_the_skip_reason_does_not_apply_to_routed_candidates():
    """Candidato de rota declarado nao passa pelo filtro: o comportamento
    existente (credencial da config) e preservado."""
    from app.core.dispatcher import _transparent_skip_reason

    candidate = Candidate(
        alias="free", provider="openrouter", model="vendor/free", protocol="openai"
    )
    req = ShuntRequest("anthropic", BODY, {"x-api-key": "token-do-shunt"}, credential_is_token=True)
    assert _transparent_skip_reason(candidate, req, SETTINGS) is None


@respx.mock
async def test_transparent_with_only_the_shunt_token_is_skipped_and_answers_400():
    """T4(b): cadeia so com o transparente e a unica credencial era o token do
    Shunt: nenhum upstream e chamado, e o erro sai no envelope do protocolo do
    chamador dizendo o que falta."""
    route = respx.post("https://api.anthropic.com/v1/messages")
    body = {**BODY, "model": "claude-sonnet-5"}
    req = ShuntRequest(
        "anthropic", body, {"x-api-key": "token-do-shunt"}, credential_is_token=True
    )
    pool = UpstreamPool(NO_ANTHROPIC_DECLARED)
    try:
        result = await dispatch(req, NO_ANTHROPIC_DECLARED, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 0
    assert result.status == 400
    assert result.real_model is None
    assert result.body["type"] == "error"
    assert "shunt token" in result.body["error"]["message"]


@respx.mock
async def test_anthropic_caller_gpt5_no_default_returns_400():
    """T5(a), o exemplo canonico: Claude Code pedindo gpt-5 sem default. O
    transparente openai NAO e elegivel para a credencial anthropic: pulado,
    sem outro candidato, 400 no envelope Anthropic dizendo o que falta."""
    route = respx.post("https://api.openai.com/v1/chat/completions")
    body = {**BODY, "model": "gpt-5"}
    pool = UpstreamPool(NO_ANTHROPIC_DECLARED)
    try:
        result = await dispatch(
            ShuntRequest("anthropic", body, {"x-api-key": "sk-do-cliente"}),
            NO_ANTHROPIC_DECLARED,
            pool,
        )
    finally:
        await pool.aclose()
    assert route.call_count == 0
    assert result.status == 400
    assert result.body["type"] == "error"
    assert result.body["error"]["type"] == "invalid_request_error"
    # o erro aponta o caminho: credencial do destino certo ou default_model
    assert "gpt-5" in result.body["error"]["message"]


@respx.mock
async def test_openai_caller_claude_model_no_default_returns_400_in_openai_envelope():
    """T5(b), simetrico: cliente OpenAI pedindo claude-* sem default. O
    transparente anthropic nao e elegivel para a credencial openai: 400 no
    envelope OpenAI (sem `type` na raiz, `error.param` presente)."""
    route = respx.post("https://api.anthropic.com/v1/messages")
    body = {**BODY, "model": "claude-sonnet-5"}
    pool = UpstreamPool(NO_ANTHROPIC_DECLARED)
    try:
        result = await dispatch(
            ShuntRequest("openai", body, {"authorization": "Bearer sk-do-cliente"}, endpoint="chat"),
            NO_ANTHROPIC_DECLARED,
            pool,
        )
    finally:
        await pool.aclose()
    assert route.call_count == 0
    assert result.status == 400
    assert "type" not in result.body
    assert result.body["error"]["type"] == "invalid_request_error"
    assert "claude-sonnet-5" in result.body["error"]["message"]


# --- escolha de caminho, corpo e tradução ------------------------------------

ANTHROPIC_SETTINGS = Settings(
    providers={
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test", protocol="anthropic", api_key="sk-teste"
        )
    },
    models={
        "fable": ModelConfig(
            provider="anthropic",
            model="claude-fable-5-1",
            context_window=200000,
            max_output_tokens=8192,
        )
    },
    routes=[("gpt-4o", ["fable"]), ("claude-opus-4-5", ["fable"])],
    default_model=None,
)


@respx.mock
async def test_endpoint_without_a_path_on_the_provider_protocol_is_skipped():
    route = respx.post("https://api.anthropic.test/v1/messages")
    pool = UpstreamPool(ANTHROPIC_SETTINGS)
    try:
        # `claude-opus-4-5` casa na rota exact e deriva para "anthropic" (ja
        # declarado em ANTHROPIC_SETTINGS): a cadeia inteira fica anthropic, e
        # nenhum dos dois candidatos tem path de /embeddings. Usar um nome
        # `gpt-*` traria um degrau transparente openai, que APOIA /embeddings e
        # encobriria o descarte por path.
        result = await dispatch(
            ShuntRequest(
                "openai", {"model": "claude-opus-4-5", "input": "oi"}, {}, endpoint="embeddings"
            ),
            ANTHROPIC_SETTINGS,
            pool,
        )
    finally:
        await pool.aclose()
    assert route.call_count == 0
    assert result.status == 502
    assert "endpoint not supported" in result.body["error"]["message"]
    assert result.trace == [
        "claude-fable-5-1: endpoint not supported",
        "claude-opus-4-5: endpoint not supported",
    ]


@respx.mock
async def test_the_output_cap_comes_from_the_candidate_not_from_the_client():
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-fable-5-1",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            },
        )
    )
    pool = UpstreamPool(ANTHROPIC_SETTINGS)
    try:
        await dispatch(
            ShuntRequest(
                "openai",
                {
                    "model": "gpt-4o",
                    "max_tokens": 999999,
                    "messages": [{"role": "user", "content": "oi"}],
                },
                {},
            ),
            ANTHROPIC_SETTINGS,
            pool,
        )
    finally:
        await pool.aclose()
    import json as _json

    sent = _json.loads(route.calls[0].request.content)
    assert sent["max_tokens"] == 8192
    assert sent["model"] == "claude-fable-5-1"


@respx.mock
async def test_same_protocol_passes_the_body_through_rewriting_only_the_model():
    upstream = {
        "id": "msg_1",
        "model": "claude-fable-5-1",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, json=upstream)
    )
    pool = UpstreamPool(ANTHROPIC_SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), ANTHROPIC_SETTINGS, pool)
    finally:
        await pool.aclose()
    import json as _json

    sent = _json.loads(route.calls[0].request.content)
    assert sent == {**BODY, "model": "claude-fable-5-1"}
    assert result.body == upstream


# --- filtro de capacidade, prazo e erros -------------------------------------

NO_TOOLS = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
        )
    },
    models={
        "free": ModelConfig(
            provider="openrouter",
            model="vendor/free",
            supports=ModelCaps(tools=False),
            context_window=64000,
            max_output_tokens=8192,
        )
    },
    routes=[("opus", ["free"])],
    default_model=None,
)


@respx.mock
async def test_a_chain_emptied_by_the_capability_filter_returns_400_without_calling_anyone():
    route = respx.post("https://api.test/v1/chat/completions")
    pool = UpstreamPool(NO_TOOLS)
    # "my-opus": casa na rota de familia, mas o prefixo nao deriva provedor --
    # sem degrau transparente no fim da cadeia, que encobriria o 400 do filtro.
    body = {**BODY, "model": "my-opus", "tools": [{"name": "grep", "input_schema": {"type": "object"}}]}
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), NO_TOOLS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 0
    assert result.status == 400
    assert result.real_model is None
    assert "no candidate can serve this request" in result.body["error"]["message"]
    # A linha de descarte nomeia pelo MODELO, igual a linha de tentativa
    # (`label = candidate.model`): nomear pelo alias fazia um candidato so
    # aparecer como dois modelos no mesmo rastro.
    assert result.trace == ["vendor/free: no tool support"]


class _Clock:
    """Relógio monotônico falso: devolve os valores dados, depois repete o último."""

    def __init__(self, *values):
        self._values = list(values)

    def monotonic(self):
        return self._values.pop(0) if len(self._values) > 1 else self._values[0]


@respx.mock
async def test_the_total_deadline_cuts_the_chain_short(monkeypatch):
    from app.core import dispatcher

    # Quatro instantes, nao tres: `dispatch` le o relogio uma vez antes de
    # comecar, para a duracao que vai na linha de log. Os tres seguintes sao
    # os que a decisao de prazo usa -- limite montado, primeira tentativa
    # dentro dele, segunda ja fora.
    monkeypatch.setattr(dispatcher, "time", _Clock(0.0, 0.0, 1.0, 10_000.0))
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "nao deu"}})
    )
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 1
    # `cheap` nunca foi chamado: o rastro diz isso, nao que ele estourou prazo.
    assert "vendor/cheap: not tried, deadline exceeded" in result.trace
    assert "vendor/free: 400 nao deu (attempt 1)" in result.trace


@respx.mock
async def test_429_retries_the_same_candidate_when_retry_after_fits_the_budget(monkeypatch):
    from app.core import dispatcher

    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    route = respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "1"}, json={}),
            httpx.Response(200, json=ok_payload("vendor/free")),
        ]
    )
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 2
    assert result.real_model == "vendor/free"


@respx.mock
async def test_an_unparseable_retry_after_is_treated_as_absent_and_the_candidate_is_skipped():
    route = respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, json={}),
            httpx.Response(200, json=ok_payload("vendor/cheap")),
        ]
    )
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 2
    assert result.real_model == "vendor/cheap"


@respx.mock
async def test_a_non_json_error_body_becomes_the_message_verbatim(monkeypatch):
    from app.core import dispatcher

    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(403, text="cota estourada")
    )
    pool = UpstreamPool(SETTINGS)
    body = {**BODY, "model": "my-opus"}
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 403
    assert "cota estourada" in result.body["error"]["message"]


@respx.mock
async def test_a_candidate_that_only_ever_fails_transport_is_retried_to_the_limit(monkeypatch):
    from app.core import dispatcher

    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    route = respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("recusou")
    )
    pool = UpstreamPool(SETTINGS)
    body = {**BODY, "model": "my-opus"}
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == dispatcher.MAX_ATTEMPTS * 2
    assert result.status == 502
    assert "recusou" in result.body["error"]["message"]
    assert result.real_model is None


@respx.mock
async def test_backoff_is_never_awaited_after_the_last_attempt(monkeypatch):
    """`if attempt < MAX_ATTEMPTS: await asyncio.sleep(backoff(attempt))` --
    the strict `<` means the final attempt of each candidate is never
    followed by a sleep. A mutation to `<=` would add one extra backoff() call
    per candidate; nothing about `route.call_count` or the result would
    change (retries are driven by the outer `for attempt in range(...)` loop,
    not by how many times we slept), so only counting `backoff()` calls
    directly catches it."""
    from app.core import dispatcher

    calls: list[int] = []
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: calls.append(attempt) or 0.0)
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("recusou")
    )
    pool = UpstreamPool(SETTINGS)
    body = {**BODY, "model": "my-opus"}
    try:
        await dispatch(ShuntRequest("anthropic", body, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    # Two candidates ("free", "cheap"), each retried MAX_ATTEMPTS times but
    # slept between attempts only MAX_ATTEMPTS - 1 times.
    assert len(calls) == (dispatcher.MAX_ATTEMPTS - 1) * 2


def test_shunt_request_defaults_to_the_messages_endpoint():
    assert ShuntRequest("anthropic", BODY, {}).endpoint == "messages"


def test_chain_exhausted_is_an_exception():
    from app.core.dispatcher import ChainExhausted

    assert issubclass(ChainExhausted, Exception)
    with pytest.raises(ChainExhausted):
        raise ChainExhausted("vazio")


@respx.mock
async def test_a_json_content_type_with_a_broken_body_falls_back_to_the_raw_text():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            400, content=b"{nao e json", headers={"content-type": "application/json"}
        )
    )
    pool = UpstreamPool(SETTINGS)
    body = {**BODY, "model": "my-opus"}
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 400
    assert "{nao e json" in result.body["error"]["message"]


@respx.mock
async def test_a_json_error_body_that_is_not_an_object_falls_back_to_the_raw_text():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json=["explodiu"])
    )
    pool = UpstreamPool(SETTINGS)
    body = {**BODY, "model": "my-opus"}
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 400
    assert "explodiu" in result.body["error"]["message"]


MIXED = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
        )
    },
    models={
        "free": ModelConfig(
            provider="openrouter",
            model="vendor/free",
            supports=ModelCaps(tools=False),
            context_window=64000,
            max_output_tokens=8192,
        ),
        "cheap": ModelConfig(
            provider="openrouter",
            model="vendor/cheap",
            context_window=64000,
            max_output_tokens=8192,
        ),
        "backup": ModelConfig(
            provider="openrouter",
            model="vendor/backup",
            context_window=64000,
            max_output_tokens=8192,
        ),
    },
    routes=[("opus", ["free", "cheap", "backup"])],
    default_model=None,
)


@respx.mock
async def test_a_successful_result_still_reports_what_happened_before_it():
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "nao deu"}}),
            httpx.Response(200, json=ok_payload("vendor/backup")),
        ]
    )
    pool = UpstreamPool(MIXED)
    body = {**BODY, "tools": [{"name": "grep", "input_schema": {"type": "object"}}]}
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), MIXED, pool)
    finally:
        await pool.aclose()
    assert result.real_model == "vendor/backup"
    assert result.trace == ["vendor/free: no tool support", "vendor/cheap: 400 nao deu (attempt 1)"]


# --- rodada 1 de correcoes ---------------------------------------------------

VISION = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
        )
    },
    models={
        "cego": ModelConfig(
            provider="openrouter",
            model="vendor/cego",
            supports=ModelCaps(vision=False),
            context_window=64000,
            max_output_tokens=8192,
        ),
        "enxerga": ModelConfig(
            provider="openrouter",
            model="vendor/enxerga",
            supports=ModelCaps(vision=True),
            context_window=64000,
            max_output_tokens=8192,
        ),
    },
    routes=[("opus", ["cego", "enxerga"])],
    default_model=None,
)

ANTHROPIC_IMAGE_BODY = {
    "model": "claude-opus-4-5",
    "max_tokens": 64,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
                },
                {"type": "text", "text": "o que e isso?"},
            ],
        }
    ],
}


@respx.mock
async def test_an_anthropic_image_drops_a_candidate_without_vision():
    """C1: o probe usa o payload do primeiro candidato. Quando ele fala o mesmo
    protocolo do cliente o corpo chega sem traducao, e a grafia Anthropic da
    imagem precisa contar como exigencia de visao — senao a captura de tela vai
    parar num modelo cego."""
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_payload("vendor/enxerga"))
    )
    pool = UpstreamPool(VISION)
    try:
        result = await dispatch(ShuntRequest("anthropic", ANTHROPIC_IMAGE_BODY, {}), VISION, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 1
    assert result.real_model == "vendor/enxerga"
    assert result.trace == ["vendor/cego: no vision support"]


ANTHROPIC_PAIR = Settings(
    providers={
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test", protocol="anthropic", api_key="sk-teste"
        )
    },
    models={
        "a": ModelConfig(
            provider="anthropic", model="claude-a", context_window=200000, max_output_tokens=8192
        ),
        "b": ModelConfig(
            provider="anthropic", model="claude-b", context_window=200000, max_output_tokens=8192
        ),
    },
    routes=[("opus", ["a", "b"])],
    default_model=None,
)


@respx.mock
async def test_a_2xx_body_that_is_not_json_falls_through_to_the_next_candidate():
    """I1: um gateway interposto devolvendo HTML com status 200 nao pode virar
    500 para o usuario; e falha deste candidato, como qualquer outra."""
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        side_effect=[
            httpx.Response(
                200, content=b"<html>desculpe</html>", headers={"content-type": "text/html"}
            ),
            httpx.Response(
                200,
                json={
                    "id": "msg_2",
                    "model": "claude-b",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {},
                },
            ),
        ]
    )
    pool = UpstreamPool(ANTHROPIC_PAIR)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), ANTHROPIC_PAIR, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 2
    assert result.status == 200
    assert result.real_model == "claude-b"
    assert "claude-a: unreadable body (attempt 1)" in result.trace


@respx.mock
async def test_every_2xx_body_that_is_not_json_exhausts_the_chain_with_502():
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200, content=b"<html>desculpe</html>", headers={"content-type": "text/html"}
        )
    )
    pool = UpstreamPool(ANTHROPIC_PAIR)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), ANTHROPIC_PAIR, pool)
    finally:
        await pool.aclose()
    # Tres candidatos desde que o degrau de baixo entrou na cadeia resolvida:
    # `a`, `b` e o modelo pedido em modo transparente. Todos respondem a mesma
    # coisa, entao a ultima falha e a que sai.
    assert route.call_count == 3
    assert result.status == 502
    assert "not a usable JSON object" in result.body["error"]["message"]


@respx.mock
async def test_a_bare_string_error_envelope_becomes_the_message():
    """I2: o envelope nativo do Ollama e `{"error": "..."}` — string, nao objeto.
    Ler `.get` dela levantava AttributeError e matava a requisicao inteira."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(404, json={"error": "model not found"})
    )
    pool = UpstreamPool(SETTINGS)
    body = {**BODY, "model": "my-opus"}
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 404
    # Equality, not `in`: falling through to `response.text` would also contain
    # the phrase, and would have let the unhandled-string regression back in.
    assert result.body["error"]["message"].startswith("model not found - tried:")


@respx.mock
async def test_an_error_object_without_a_message_falls_back_to_the_raw_text():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"code": "bad"}})
    )
    pool = UpstreamPool(SETTINGS)
    body = {**BODY, "model": "my-opus"}
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 400
    assert '"code"' in result.body["error"]["message"]


MIXED_PROTOCOLS = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
        ),
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test", protocol="anthropic", api_key="sk-teste"
        ),
    },
    models={
        "openai_um": ModelConfig(
            provider="openrouter",
            model="vendor/free",
            context_window=200000,
            max_output_tokens=8192,
        ),
        "anthropic_um": ModelConfig(
            provider="anthropic", model="claude-a", context_window=200000, max_output_tokens=8192
        ),
    },
    routes=[("opus", ["openai_um", "anthropic_um"])],
    default_model=None,
)


@respx.mock
async def test_a_translator_failure_skips_the_candidate_instead_of_killing_the_chain():
    """I3: mais de 4 stop sequences faz `anthropic_request_to_openai` levantar.
    Isso e um defeito daquele candidato, nao do pedido: o candidato Anthropic
    logo abaixo leva o corpo sem traducao nenhuma."""
    openai_route = respx.post("https://api.test/v1/chat/completions")
    anthropic_route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-a",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            },
        )
    )
    body = {**BODY, "stop_sequences": ["a", "b", "c", "d", "e"]}
    pool = UpstreamPool(MIXED_PROTOCOLS)
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), MIXED_PROTOCOLS, pool)
    finally:
        await pool.aclose()
    assert openai_route.call_count == 0
    assert anthropic_route.call_count == 1
    assert result.status == 200
    assert result.real_model == "claude-a"
    assert any("vendor/free: request translation failed" in line for line in result.trace)


BLIND_THEN_SEEING = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
        ),
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test", protocol="anthropic", api_key="sk-teste"
        ),
    },
    models={
        "cego": ModelConfig(
            provider="openrouter",
            model="vendor/cego",
            supports=ModelCaps(vision=False),
            context_window=200000,
            max_output_tokens=8192,
        ),
        "enxerga": ModelConfig(
            provider="anthropic",
            model="claude-a",
            supports=ModelCaps(vision=True),
            context_window=200000,
            max_output_tokens=8192,
        ),
    },
    routes=[("opus", ["cego", "enxerga"])],
    default_model=None,
)


@respx.mock
async def test_the_probe_degrades_to_the_untranslated_body_instead_of_raising():
    """I3, metade do probe: o probe usa o primeiro candidato, que aqui nao
    consegue traduzir (5 stop sequences). Se ele levantasse, nenhum candidato
    seria tentado. Caindo para o corpo cru, o filtro ainda ve a imagem e
    descarta o modelo cego antes de qualquer chamada."""
    openai_route = respx.post("https://api.test/v1/chat/completions")
    anthropic_route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-a",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            },
        )
    )
    body = {**ANTHROPIC_IMAGE_BODY, "stop_sequences": ["a", "b", "c", "d", "e"]}
    pool = UpstreamPool(BLIND_THEN_SEEING)
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), BLIND_THEN_SEEING, pool)
    finally:
        await pool.aclose()
    assert openai_route.call_count == 0
    assert anthropic_route.call_count == 1
    assert result.real_model == "claude-a"
    # O probe falhou e deixou rastro: sem essa linha, um `chain[0]` descartado
    # depois pelo filtro apagaria toda a evidencia de que ele falhou.
    assert result.trace[0].startswith("probe (vendor/cego): request translation failed:")
    assert result.trace[1:] == ["vendor/cego: no vision support"]


PATH_SETTINGS = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
        ),
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test", protocol="anthropic", api_key="sk-teste"
        ),
    },
    models={
        "openai_um": ModelConfig(
            provider="openrouter",
            model="vendor/free",
            context_window=200000,
            max_output_tokens=8192,
        ),
        "anthropic_um": ModelConfig(
            provider="anthropic", model="claude-a", context_window=200000, max_output_tokens=8192
        ),
    },
    routes=[("rota-openai", ["openai_um"]), ("rota-anthropic", ["anthropic_um"])],
    default_model=None,
)

PATH_CASES = [
    ("openai", "chat", "rota-openai", "https://api.test/v1/chat/completions"),
    ("openai", "messages", "rota-openai", "https://api.test/v1/chat/completions"),
    ("openai", "completions", "rota-openai", "https://api.test/v1/completions"),
    ("openai", "embeddings", "rota-openai", "https://api.test/v1/embeddings"),
    ("anthropic", "messages", "rota-anthropic", "https://api.anthropic.test/v1/messages"),
    ("anthropic", "chat", "rota-anthropic", "https://api.anthropic.test/v1/messages"),
]


def test_the_path_cases_cover_every_row_of_paths():
    from app.core import dispatcher

    assert {(p, e) for p, e, _, _ in PATH_CASES} == set(dispatcher.PATHS)


@pytest.mark.parametrize(("protocol", "endpoint", "model", "url"), PATH_CASES)
@respx.mock
async def test_each_protocol_endpoint_pair_lands_on_its_own_path(protocol, endpoint, model, url):
    route = respx.post(url).mock(return_value=httpx.Response(200, json={}))
    pool = UpstreamPool(PATH_SETTINGS)
    client_protocol = "anthropic" if protocol == "anthropic" else "openai"
    try:
        await dispatch(
            ShuntRequest(
                client_protocol,
                {"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "oi"}]},
                {},
                endpoint=endpoint,
            ),
            PATH_SETTINGS,
            pool,
        )
    finally:
        await pool.aclose()
    assert route.call_count == 1
    assert str(route.calls[0].request.url) == url


@respx.mock
async def test_openai_in_openai_out_passes_through_rewriting_only_the_model():
    upstream = ok_payload("vendor/free")
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=upstream)
    )
    body = {"model": "opus", "max_tokens": 32, "messages": [{"role": "user", "content": "oi"}]}
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("openai", body, {}, endpoint="chat"), SETTINGS, pool)
    finally:
        await pool.aclose()
    import json as _json

    assert _json.loads(route.calls[0].request.content) == {**body, "model": "vendor/free"}
    assert result.body == upstream


@respx.mock
async def test_the_pass_through_branch_also_clamps_max_tokens_to_the_candidate_cap():
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-a",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            },
        )
    )
    pool = UpstreamPool(ANTHROPIC_PAIR)
    try:
        await dispatch(
            ShuntRequest("anthropic", {**BODY, "max_tokens": 999999}, {}), ANTHROPIC_PAIR, pool
        )
    finally:
        await pool.aclose()
    import json as _json

    assert _json.loads(route.calls[0].request.content)["max_tokens"] == 8192


@respx.mock
async def test_transparent_mode_never_rewrites_the_clients_max_tokens():
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-sonnet-9",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            },
        )
    )
    pool = UpstreamPool(TRANSPARENT)
    body = {**BODY, "model": "claude-sonnet-9", "max_tokens": 999999}
    try:
        await dispatch(ShuntRequest("anthropic", body, {}), TRANSPARENT, pool)
    finally:
        await pool.aclose()
    import json as _json

    # Byte-for-byte equality, not just the max_tokens field: this is what
    # proves a same-protocol pass-through does not translate the body at all.
    assert _json.loads(route.calls[0].request.content) == body


def test_payload_without_an_alias_falls_back_to_the_default_cap():
    from app.core import dispatcher

    payload = dispatcher._payload(
        ShuntRequest("anthropic", {**BODY, "max_tokens": 999999}, {}),
        Candidate(alias=None, provider="openrouter", model="m", protocol="openai"),
        SETTINGS,
    )
    assert payload["max_tokens"] == dispatcher.DEFAULT_MAX_OUTPUT_TOKENS


def test_transparent_drop_also_covers_the_body_describing_headers():
    from app.core import dispatcher

    candidate = Candidate(
        alias=None, provider="anthropic", model="m", protocol="anthropic", transparent=True
    )
    headers = outbound_headers(
        ShuntRequest(
            "anthropic",
            BODY,
            {"Content-Encoding": "gzip", "Transfer-Encoding": "chunked", "X-Api-Key": "sk"},
        ),
        candidate,
        TRANSPARENT,
    )
    assert headers == {"X-Api-Key": "sk"}
    assert {"content-encoding", "transfer-encoding"} <= dispatcher.TRANSPARENT_DROP


@respx.mock
async def test_a_2xx_that_is_not_200_is_reported_with_its_own_status():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(201, json=ok_payload("vendor/free"))
    )
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 201
    assert result.real_model == "vendor/free"


def test_chain_exhausted_documents_that_dispatch_never_raises_it():
    from app.core.dispatcher import ChainExhausted

    assert ChainExhausted.__doc__ is not None
    assert "never raised" in ChainExhausted.__doc__.lower()


GATEWAY = Settings(
    providers={
        "anthropic": ProviderConfig(
            base_url="https://gateway.exemplo/v1", protocol="openai", api_key="sk-da-config"
        )
    },
    models={},
    routes=[],
    default_model=None,
)


@respx.mock
async def test_transparent_across_protocols_translates_both_halves(monkeypatch):
    """Um `provider` chamado `anthropic` pode perfeitamente declarar
    `protocol: "openai"` — e apontar nomes `claude-*` para um gateway
    compativel com a OpenAI e algo normal de querer. `PATHS` ja re-enderecou a
    requisicao para `/chat/completions`; mandar o corpo Anthropic cru para la e
    um 400 garantido. As duas metades traduzem."""
    route = respx.post("https://gateway.exemplo/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_payload("claude-opus-4-5"))
    )
    body = {
        "model": "claude-opus-4-5",
        "max_tokens": 999999,
        "system": "seja breve",
        "messages": [
            {"role": "user", "content": "oi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {"path": "/"}}
                ],
            },
        ],
    }
    pool = UpstreamPool(GATEWAY)
    try:
        result = await dispatch(
            ShuntRequest("anthropic", body, dict(CLIENT_HEADERS)), GATEWAY, pool
        )
    finally:
        await pool.aclose()
    import json as _json

    sent = _json.loads(route.calls[0].request.content)
    # Ida traduzida para a forma OpenAI: `system` vira mensagem, e o bloco
    # `tool_use` vira `tool_calls` com `arguments` em string JSON.
    assert "system" not in sent
    assert sent["messages"][0] == {"role": "system", "content": "seja breve"}
    assert sent["messages"][-1]["tool_calls"][0]["function"] == {
        "name": "ls",
        "arguments": '{"path": "/"}',
    }
    assert sent["model"] == "claude-opus-4-5"
    # Volta traduzida de volta para a forma Anthropic.
    assert result.status == 200
    assert result.body["type"] == "message"
    assert result.body["content"] == [{"type": "text", "text": "pronto"}]


@respx.mock
async def test_transparent_translation_uses_the_clients_cap_not_our_default(monkeypatch):
    """O que continua transparente e nao contribuirmos configuracao nem
    credencial nossa. Passar `_cap()` aqui devolveria DEFAULT_MAX_OUTPUT_TOKENS
    para `alias=None` e cortaria os 999999 do cliente para 4096 em silencio."""
    route = respx.post("https://gateway.exemplo/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_payload("claude-opus-4-5"))
    )
    body = {
        "model": "claude-opus-4-5",
        "max_tokens": 999999,
        "messages": [{"role": "user", "content": "oi"}],
    }
    pool = UpstreamPool(GATEWAY)
    try:
        await dispatch(ShuntRequest("anthropic", body, {}), GATEWAY, pool)
    finally:
        await pool.aclose()
    import json as _json

    assert _json.loads(route.calls[0].request.content)["max_tokens"] == 999999


@respx.mock
async def test_transparent_credentials_go_verbatim_to_the_declared_base_url(monkeypatch):
    route = respx.post("https://gateway.exemplo/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_payload("claude-opus-4-5"))
    )
    body = {
        "model": "claude-opus-4-5",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "oi"}],
    }
    pool = UpstreamPool(GATEWAY)
    try:
        await dispatch(ShuntRequest("anthropic", body, dict(CLIENT_HEADERS)), GATEWAY, pool)
    finally:
        await pool.aclose()
    sent = route.calls[0].request.headers
    assert sent["x-api-key"] == "sk-do-cliente"
    assert sent["authorization"] == "Bearer oauth-da-assinatura"
    assert "sk-da-config" not in str(dict(sent))


def test_transparent_cap_falls_back_to_the_default_when_the_client_sends_none():
    from app.core import dispatcher

    assert (
        dispatcher._transparent_cap(ShuntRequest("anthropic", {"messages": []}, {}))
        == dispatcher.DEFAULT_MAX_OUTPUT_TOKENS
    )
    assert (
        dispatcher._transparent_cap(ShuntRequest("anthropic", {"max_tokens": "muitos"}, {}))
        == dispatcher.DEFAULT_MAX_OUTPUT_TOKENS
    )
    assert dispatcher._transparent_cap(ShuntRequest("anthropic", {"max_tokens": 777}, {})) == 777


@respx.mock
@pytest.mark.parametrize("payload", [["lista"], None, 7, "texto"])
async def test_a_2xx_whose_json_is_not_an_object_falls_through_to_the_next_candidate(payload):
    """Passa pelo `response.json()` e so entao quebra dentro do tradutor, que
    faz `resp.get(...)` num nao-dicionario. Mesma classe do corpo ilegivel."""
    route = respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=payload),
            httpx.Response(200, json=ok_payload("vendor/cheap")),
        ]
    )
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 2
    assert result.real_model == "vendor/cheap"
    assert "vendor/free: unreadable body (attempt 1)" in result.trace


@respx.mock
async def test_a_non_numeric_max_tokens_is_left_for_the_provider_to_reject():
    """A pinca do passthrough coage com `int()`. Um valor nao numerico e
    deixado como veio de proposito: o provedor responde o proprio 400, em vez
    de a gente pular o candidato por um erro de coercao."""
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(400, json={"error": {"message": "max_tokens invalido"}})
    )
    pool = UpstreamPool(ANTHROPIC_PAIR)
    try:
        result = await dispatch(
            ShuntRequest("anthropic", {**BODY, "max_tokens": "muitos"}, {}), ANTHROPIC_PAIR, pool
        )
    finally:
        await pool.aclose()
    import json as _json

    assert _json.loads(route.calls[0].request.content)["max_tokens"] == "muitos"
    assert result.status == 400
    assert "max_tokens invalido" in result.body["error"]["message"]
    # `a`, `b` e o degrau de baixo transparente com o max_tokens do cliente.
    assert route.call_count == 3


def test_a_transparent_candidate_with_an_alias_still_keeps_the_clients_max_tokens():
    """`resolver._transparent` so produz `alias=None`, mas a regra nao e "sem
    alias" e sim "em modo transparente nao reescrevemos o numero do cliente".
    Um candidato transparente construido com alias prova qual das duas o codigo
    esta aplicando — o mesmo par que `test_capabilities` ja fixa no filtro."""
    from app.core import dispatcher

    candidate = Candidate(
        alias="free",
        provider="openrouter",
        model="vendor/free",
        protocol="anthropic",
        transparent=True,
    )
    payload = dispatcher._payload(
        ShuntRequest("anthropic", {**BODY, "max_tokens": 999999}, {}), candidate, SETTINGS
    )
    assert payload["max_tokens"] == 999999
    assert payload["model"] == "vendor/free"


# --------------------------------------------------------------------------
# Envelope de erro: a forma segue o protocolo de QUEM PERGUNTOU
# --------------------------------------------------------------------------


def test_the_error_envelope_for_an_anthropic_caller_is_anthropic_shaped():
    from app.core.dispatcher import error_body

    assert error_body("anthropic", 429, "devagar") == {
        "type": "error",
        "error": {"type": "rate_limit_error", "message": "devagar"},
    }


def test_the_error_envelope_for_an_openai_caller_is_openai_shaped():
    """No `type` na raiz e `param`/`code` presentes: e o que a biblioteca
    oficial da OpenAI le para montar sua excecao."""
    from app.core.dispatcher import error_body

    assert error_body("openai", 429, "devagar") == {
        "error": {
            "message": "devagar",
            "type": "rate_limit_error",
            "param": None,
            "code": None,
        }
    }


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (422, "invalid_request_error"),
        (429, "rate_limit_error"),
        (418, "invalid_request_error"),  # 4xx desconhecido: culpa do pedido
        (500, "server_error"),
        (502, "server_error"),  # 5xx desconhecido: culpa do servidor
    ],
)
def test_the_openai_error_type_follows_the_status(status, expected):
    from app.core.dispatcher import error_body

    assert error_body("openai", status, "x")["error"]["type"] == expected


@respx.mock
async def test_an_exhausted_chain_answers_an_openai_client_in_its_own_dialect():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "cota"}})
    )
    body = {"model": "opus", "max_tokens": 32, "messages": [{"role": "user", "content": "oi"}]}
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("openai", body, {}, endpoint="chat"), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 429
    assert "type" not in result.body
    assert result.body["error"]["type"] == "rate_limit_error"
    assert "cota" in result.body["error"]["message"]


async def test_an_empty_chain_answers_an_openai_client_in_its_own_dialect():
    """O segundo lugar onde `dispatch` monta erro: nenhum candidato passou pelo
    filtro de capacidades, antes de qualquer requisicao."""
    settings = Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
            )
        },
        models={
            "free": ModelConfig(
                provider="openrouter",
                model="vendor/free",
                supports=ModelCaps(tools=False),
                context_window=64000,
                max_output_tokens=8192,
            )
        },
        routes=[("opus", ["free"])],
        default_model=None,
    )
    body = {
        "model": "opus",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "oi"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    }
    pool = UpstreamPool(settings)
    try:
        result = await dispatch(ShuntRequest("openai", body, {}, endpoint="chat"), settings, pool)
    finally:
        await pool.aclose()
    assert result.status == 400
    assert "type" not in result.body
    assert result.body["error"]["type"] == "invalid_request_error"
    assert "no candidate can serve this request" in result.body["error"]["message"]


def test_exception_text_names_a_transport_failure_with_no_exception_object():
    """`classify` pode dizer falha sem excecao em maos; a mensagem nao pode sair vazia."""
    assert _exception_text(None) == "transport failure"
    assert _exception_text(httpx.ReadTimeout("")) == "ReadTimeout"
    assert _exception_text(httpx.ConnectError("recusou")) == "ConnectError: recusou"


# -- Historia E: a escada quando nenhum candidato cabe -------------------------

TINY = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
        ),
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test", protocol="anthropic", api_key="sk-teste"
        ),
    },
    models={
        "curto": ModelConfig(
            provider="openrouter",
            model="vendor/curto",
            context_window=50,
            max_output_tokens=16,
        ),
        "grande": ModelConfig(
            provider="openrouter",
            model="vendor/grande",
            context_window=200000,
            max_output_tokens=8192,
        ),
    },
    routes=[("opus", ["curto"])],
    default_model=None,
)

BIG_BODY = {
    "model": "claude-opus-4-5",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "x" * 8000}],
}


@respx.mock
async def test_the_default_model_is_called_even_when_it_does_not_fit_either():
    """`default_model` e a escolha do operador para "quando nada serve", e por
    isso ele e chamado mesmo sem caber -- a alternativa seria um 400 onde o
    operador ja disse o que fazer."""
    settings = TINY.model_copy(update={"default_model": "curto"})
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_payload("vendor/curto"))
    )
    pool = UpstreamPool(settings)
    try:
        result = await dispatch(ShuntRequest("anthropic", BIG_BODY, {}), settings, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 1
    assert result.status == 200
    assert result.real_model == "vendor/curto"
    assert any("nothing in the chain fits" in line for line in result.trace)


@respx.mock
async def test_a_chain_emptied_by_size_goes_transparent_without_a_default():
    """Ninguem no catalogo dando conta, quem pediu resolve com o proprio
    provedor -- e o proxy nao inventa um candidato."""
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
    )
    pool = UpstreamPool(TINY)
    try:
        result = await dispatch(ShuntRequest("anthropic", BIG_BODY, {}), TINY, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 1
    assert result.status == 200
    assert result.real_model == "claude-opus-4-5"
    # O transparente agora e o ultimo degrau da propria cadeia, e nao um
    # candidato que a escada do dispatcher inventa depois de esvaziar tudo: o
    # filtro nunca descarta um transparente, entao ele chega aqui como membro
    # normal e a nota "taken anyway" nao aparece.
    assert any("context window too small" in line for line in result.trace)
    assert not any("taken anyway" in line for line in result.trace)


@respx.mock
async def test_a_chain_emptied_by_a_missing_capability_still_returns_400():
    """A escada e so para tamanho. Cair no default por falta de ferramenta
    mandaria o pedido a um modelo que tambem nao sabe chamar ferramenta."""
    settings = NO_TOOLS.model_copy(update={"default_model": "free"})
    route = respx.post("https://api.test/v1/chat/completions")
    pool = UpstreamPool(settings)
    body = {**BODY, "tools": [{"name": "grep", "input_schema": {"type": "object"}}]}
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), settings, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 0
    assert result.status == 400


@respx.mock
async def test_a_route_with_no_candidates_is_refused_by_the_configuration():
    """A escada so e alcancavel porque a cadeia nunca chega vazia.

    Quem garante isso e a validacao do catalogo, e nao o despachante: sem esta
    recusa, uma rota sem candidato viraria uma chamada ao `default_model` por
    baixo dos panos em vez de um erro de configuracao.
    """
    with pytest.raises(ValidationError):
        Settings(
            providers=TINY.providers,
            models=TINY.models,
            routes=[("opus", [])],
            default_model="grande",
        )


# --- um candidato, um nome ------------------------------------------------
#
# Um alias existe para o operador escrever a rota; o MODELO e o que aparece
# como resultado. Enquanto o rastro usa um e o resultado usa o outro, o mesmo
# candidato fisico le como dois modelos diferentes na mesma linha do tempo.

ALIAS_DIFFERS = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
        )
    },
    models={
        "qwen-local": ModelConfig(
            provider="openrouter",
            model="qwen3.8-27b",
            context_window=64000,
            max_output_tokens=8192,
        )
    },
    routes=[("opus", ["qwen-local"])],
    default_model=None,
)


@respx.mock
async def test_the_attempt_label_is_the_model_not_the_alias():
    """O rotulo canonico e o MODELO. Com o alias, a linha de tentativa diz
    `qwen-local` e o campo de resultado diz `qwen3.8-27b`: na tela sao dois
    modelos, e o operador nao consegue ligar o pulo ao candidato que o
    respondeu."""
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "nao deu"}})
    )
    pool = UpstreamPool(ALIAS_DIFFERS)
    body = {**BODY, "model": "my-opus"}
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), ALIAS_DIFFERS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 1
    assert result.trace == ["qwen3.8-27b: 400 nao deu (attempt 1)"]


@respx.mock
async def test_the_probe_note_names_the_model_not_the_alias(monkeypatch):
    """A nota do probe nomeia o candidato cujo payload nao pode ser renderizado.
    Nomeando pelo alias, ela e a linha de tentativa apontam para dois nomes."""
    settings = ALIAS_DIFFERS.model_copy(
        update={
            "models": {
                **ALIAS_DIFFERS.models,
                "cego": ModelConfig(
                    provider="openrouter",
                    model="vendor/cego",
                    supports=ModelCaps(vision=False),
                    context_window=64000,
                    max_output_tokens=8192,
                ),
            },
            "routes": [("opus", ["qwen-local", "cego"])],
        }
    )
    real = dispatcher._payload

    def explode(req, candidate, settings):
        if candidate.model == "qwen3.8-27b":
            raise ValueError("shape estranho")
        return real(req, candidate, settings)

    monkeypatch.setattr(dispatcher, "_payload", explode)
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_payload("vendor/cego"))
    )
    pool = UpstreamPool(settings)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), settings, pool)
    finally:
        await pool.aclose()
    assert result.trace[0] == (
        "probe (qwen3.8-27b): request translation failed: shape estranho"
    )


@respx.mock
async def test_the_last_resort_note_names_the_model_not_the_alias():
    """A escada de ultimo recurso entra no mesmo rastro. Apresentando-se pelo
    alias, o degrau de baixo aparece com um terceiro nome.

    As duas janelas sao apertadas de proposito: com uma delas cabendo, o
    candidato chega como membro normal da cadeia e a escada nem dispara.
    """
    settings = Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
            )
        },
        models={
            "curto": ModelConfig(
                provider="openrouter",
                model="vendor/curto",
                context_window=50,
                max_output_tokens=16,
            ),
            "qwen-local": ModelConfig(
                provider="openrouter",
                model="qwen3.8-27b",
                context_window=100,
                max_output_tokens=16,
            ),
        },
        routes=[("opus", ["curto"])],
        default_model="qwen-local",
    )
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_payload("qwen3.8-27b"))
    )
    pool = UpstreamPool(settings)
    try:
        result = await dispatch(ShuntRequest("anthropic", BIG_BODY, {}), settings, pool)
    finally:
        await pool.aclose()
    assert result.trace[-1] == "qwen3.8-27b: taken anyway, nothing in the chain fits"


def test_an_empty_resolution_closes_the_ladder_with_the_default_model():
    """Defensiva de `_chain_for`: a cadeia chega vazia e a escada fecha no
    `default_model`, com a linha "taken anyway" no rastro.

    Nao e alcancavel pela API publica: `resolve` nunca devolve cadeia vazia
    (uma rota sem candidato nao passa da validacao --
    test_a_route_with_no_candidates_is_refused_by_the_configuration), e o
    filtro nunca descarta transparente. Coberto com um `Resolution` direto.
    """
    from app.core.resolver import Resolution

    settings = TINY.model_copy(update={"default_model": "grande"})
    resolution = Resolution("family", "opus", [])
    req = ShuntRequest("anthropic", BIG_BODY, {})
    chain, trace = dispatcher._chain_for(req, settings, resolution, BIG_BODY)
    assert [c.alias for c in chain] == ["grande"]
    assert "taken anyway, nothing in the chain fits" in "; ".join(trace)


@respx.mock
async def test_nothing_fits_so_the_default_is_tried_and_the_upstream_422_comes_back(
    monkeypatch,
):
    """O sintoma da producao, documentado: "context window too small" seguido
    de 422. NADA na cadeia cabe (nem o `default_model`): o proxy ainda tenta
    o `default_model` -- e a escolha do operador para "quando nada serve" --
    e o 4xx do upstream volta no envelope do chamador, nao um 400 nosso.

    A escada do `_chain_for` (589-593) NAO entra aqui: com `default_model`
    declarado, `resolve` o anexa ao fim da cadeia e a linha 571 impede que ele
    seja descartado por tamanho, entao o `kept` nunca fica vazio. O que corre
    e o caminho normal: o `curto` cai no rastro (574) e o `grande` chega com
    "taken anyway" (658-659), mas segue para o upstream."""
    settings = Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key="sk-teste"
            )
        },
        models={
            "curto": ModelConfig(
                provider="openrouter",
                model="vendor/curto",
                context_window=50,
                max_output_tokens=16,
            ),
            "tambem_curto": ModelConfig(
                provider="openrouter",
                model="vendor/tambem_curto",
                context_window=100,
                max_output_tokens=16,
            ),
        },
        routes=[("opus", ["curto"])],
        default_model="tambem_curto",
    )
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(422, json={"error": {"message": "prompt too long"}})
    )
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    pool = UpstreamPool(settings)
    try:
        result = await dispatch(ShuntRequest("anthropic", {**BIG_BODY, "model": "my-opus"}, {}), settings, pool)
    finally:
        await pool.aclose()
    # So o default_model foi chamado: o `curto` nem tentou.
    assert route.call_count == 1
    assert result.status == 422
    assert result.body["error"]["type"] == "invalid_request_error"
    assert "prompt too long" in result.body["error"]["message"]
    joined = " | ".join(result.trace)
    # a linha de descarte por tamanho aparece ANTES da tentativa do default
    assert "vendor/curto: context window too small" in joined
    # e traz OS NUMEROS do porquê (needed > context_window), como em
    # capabilities.py -- sem eles o operador nao sabe por quanto passou.
    assert re.search(
        r"vendor/curto: context window too small \(\d+ > \d+\)", joined
    ), f"descarte por tamanho sem numeros no rastro: {joined!r}"
    assert "vendor/tambem_curto: taken anyway, nothing in the chain fits" in joined
    assert joined.index("context window too small") < joined.index("taken anyway")


@respx.mock
async def test_a_transparent_candidate_without_a_default_is_tried_despite_the_window_note(
    monkeypatch,
):
    """Transparente SEM default, tentado no laço sem caber: a nota
    `(estimate > inferred window)` entra no rastro e a chamada SEGUE -- o
    upstream decide com o 4xx dele, nao o proxy.

    A chave esta em `_chain_for` devolver o transparente (o filtro nunca o
    descarta) e a linha do laço registra a nota SEM `continue` e segue: uma
    regra de pulo aqui mataria justamente o ultimo recurso. Caminho publico:
    `claude-*` sem rota e sem "anthropic" no catalogo e o transparente so da
    cadeia (como o teste de host oficial acima, mas com um corpo que nao
    caberia em janela nenhuma do catalogo)."""
    body = {**BIG_BODY, "model": "claude-opus-4-5"}
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            422,
            json={"type": "error", "error": {"type": "invalid_request_error", "message": "prompt too long"}},
        )
    )
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    pool = UpstreamPool(NO_ANTHROPIC_DECLARED)
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), NO_ANTHROPIC_DECLARED, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 1
    joined = " | ".join(result.trace)
    # a nota fica no rastro E a chamada sai para o host oficial
    assert "context window too small (estimate > inferred window)" in joined
    assert result.status == 422
    assert "prompt too long" in result.body["error"]["message"]


@respx.mock
async def test_an_anthropic_result_reports_the_dict_output_config_effort(
    monkeypatch,
):
    """Linha 803 do caminho bufferizado: quando o payload final tem
    `output_config` como DICT (o caso normal do candidato Anthropic com effort
    configurado), o `effort` enviado ao upstream e capturado e sai em
    `ShuntResult.effort`. E o espelho positivo do teste do `output_config`
    nao-dict acima, e fecha a linha do `.get("effort")`."""
    from app.core.resolver import Resolution

    respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, json=OFFICIAL_ANTHROPIC_OK)
    )
    pool = UpstreamPool(EFFORT_SETTINGS)
    # candidato Anthropic com effort="high": `_override_effort` escreve
    # output_config={"effort": "high"} no payload final
    body = {**BODY, "model": "claude-a", "output_config": {"effort": "low"}}
    try:
        result = await dispatcher._dispatch(
            ShuntRequest("anthropic", body, {}),
            EFFORT_SETTINGS,
            pool,
            Resolution("exact", "claude-a", [ANT_HIGH]),
        )
    finally:
        await pool.aclose()
    assert result.status == 200
    assert result.effort == "high"


@respx.mock
async def test_an_anthropic_result_with_a_non_dict_output_config_reports_none_effort(
    monkeypatch,
):
    """Linha 803 do caminho bufferizado: o effort enviado vem de
    `output_config` SO quando ela e dict. Um cliente pode mandar qualquer tipo
    nesse campo; antes do guard (`isinstance(out_cfg, dict)`) o
    `payload.get("output_config", {}).get("effort")` levantava AttributeError
    no caminho 2xx -- a chamada ja tinha dado certo e virava um 500 nosso."""
    from app.core.resolver import Resolution

    original = dispatcher._payload

    def payload_with_string_config(req, candidate, settings):
        return {**original(req, candidate, settings), "output_config": "alto"}

    monkeypatch.setattr(dispatcher, "_payload", payload_with_string_config)
    respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, json=OFFICIAL_ANTHROPIC_OK)
    )
    pool = UpstreamPool(EFFORT_SETTINGS)
    try:
        result = await dispatcher._dispatch(
            ShuntRequest("anthropic", {**BODY, "model": "claude-a"}, {}),
            EFFORT_SETTINGS,
            pool,
            Resolution("exact", "claude-a", [ANT_HIGH]),
        )
    finally:
        await pool.aclose()
    assert result.status == 200
    # sem crash, e o effort nao inventado: None, nao uma AttributeError
    assert result.effort is None


@respx.mock
async def test_an_exhausted_provider_is_skipped_and_recorded_with_the_next_candidate_serving(
    monkeypatch,
):
    """678-681 do laço bufferizado: provedor no limite de concorrencia pula o
    candidato AGORA (na fila dele que a requisicao nao espera) e a linha de
    rastro diz `busy (1/1 in use)`; o proximo candidato atende."""
    settings = Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key="sk-teste",
                max_concurrency=1,
            ),
            "groq": ProviderConfig(
                base_url="https://api.groq.test/v1", protocol="openai", api_key="sk-groq"
            ),
        },
        models={
            "free": ModelConfig(
                provider="openrouter",
                model="vendor/free",
                context_window=64000,
                max_output_tokens=8192,
            ),
            "fast": ModelConfig(
                provider="groq",
                model="vendor/fast",
                context_window=64000,
                max_output_tokens=8192,
            ),
        },
        routes=[("opus", ["free", "fast"])],
        default_model=None,
    )
    respx.post("https://api.test/v1/chat/completions")
    respx.post("https://api.groq.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_payload("vendor/fast"))
    )
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    pool = UpstreamPool(settings)
    held = pool.try_slot("openrouter")
    assert held is not None
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), settings, pool)
    finally:
        held.release()
        await pool.aclose()
    # o free pulou cheio; o fast respondeu
    assert result.status == 200
    assert result.real_model == "vendor/fast"
    assert any("busy (1/1 in use)" in line for line in result.trace)


# --- prazo: um servidor saturado nao pode consumir a cadeia inteira ----------


@respx.mock
async def test_a_read_timeout_moves_to_the_next_candidate_without_retrying(monkeypatch):
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    route = respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.ReadTimeout(""),
            httpx.Response(200, json=ok_payload("vendor/cheap")),
        ]
    )
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 2
    assert result.real_model == "vendor/cheap"
    # O rastro nomeia o motivo, nao so o status: um ReadTimeout tem str() vazio.
    assert result.trace[0] == "vendor/free: 502 ReadTimeout (attempt 1)"


async def _timeout_of_first_call(monkeypatch, *clock):
    monkeypatch.setattr(dispatcher, "TOTAL_DEADLINE", 5.0)
    monkeypatch.setattr(dispatcher, "time", _Clock(*clock))
    with respx.mock:
        route = respx.post("https://api.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=ok_payload("vendor/free"))
        )
        pool = UpstreamPool(SETTINGS)
        try:
            await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
        finally:
            await pool.aclose()
    return route.calls[0].request.extensions["timeout"]


async def test_the_read_timeout_of_an_attempt_is_bounded_by_the_remaining_deadline(monkeypatch):
    from app.config.config import TIMEOUT_CONNECT, TIMEOUT_POOL, TIMEOUT_WRITE

    # log, limite (0 + 5), tentativa em 2.0 -> sobram 3s de prazo.
    timeout = await _timeout_of_first_call(monkeypatch, 0.0, 0.0, 2.0)
    assert timeout == {
        "connect": TIMEOUT_CONNECT,
        "read": 3.0,
        "write": TIMEOUT_WRITE,
        "pool": TIMEOUT_POOL,
    }


async def test_the_read_timeout_never_exceeds_the_configured_one(monkeypatch):
    from app.config.config import TIMEOUT_READ

    monkeypatch.setattr(dispatcher, "TOTAL_DEADLINE", TIMEOUT_READ * 10)
    monkeypatch.setattr(dispatcher, "time", _Clock(0.0))
    with respx.mock:
        route = respx.post("https://api.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=ok_payload("vendor/free"))
        )
        pool = UpstreamPool(SETTINGS)
        try:
            await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
        finally:
            await pool.aclose()
    assert route.calls[0].request.extensions["timeout"]["read"] == TIMEOUT_READ


async def test_the_read_timeout_keeps_a_floor_when_the_deadline_is_exactly_now(monkeypatch):
    # tentativa exatamente no limite (5.0): o `>` deixa passar, e zero nao
    # pode chegar ao httpx (read=0 estoura na hora).
    timeout = await _timeout_of_first_call(monkeypatch, 0.0, 0.0, 5.0)
    assert timeout["read"] == dispatcher.READ_FLOOR
    assert dispatcher.READ_FLOOR > 0


async def _dispatch_503_then_ok(monkeypatch, wait):
    """Primeira resposta 503 (RETRY), a seguinte 200. Prazo 5s, relogio em 1.0."""
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(dispatcher, "TOTAL_DEADLINE", 5.0)
    monkeypatch.setattr(dispatcher, "time", _Clock(0.0, 0.0, 1.0))
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: wait)
    monkeypatch.setattr(dispatcher.asyncio, "sleep", fake_sleep)
    with respx.mock:
        route = respx.post("https://api.test/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(503, json={"error": {"message": "ocupado"}}),
                httpx.Response(200, json=ok_payload("vendor/free")),
            ]
        )
        pool = UpstreamPool(SETTINGS)
        try:
            result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
        finally:
            await pool.aclose()
    sent = [json.loads(call.request.content)["model"] for call in route.calls]
    return result, sent, slept


async def test_a_backoff_that_would_pass_the_deadline_is_not_slept(monkeypatch):
    # 1.0 + 10.0 > 5.0: dormir so para acordar depois do prazo gasta o tempo
    # que o proximo candidato ainda tinha.
    result, sent, slept = await _dispatch_503_then_ok(monkeypatch, 10.0)
    assert slept == []
    assert sent == ["vendor/free", "vendor/cheap"]
    assert "vendor/free: retry skipped, backoff would pass the deadline" in result.trace


async def test_a_backoff_that_ends_exactly_at_the_deadline_is_still_slept(monkeypatch):
    # 1.0 + 4.0 == 5.0: nao passa do prazo, entao a espera acontece e o
    # mesmo candidato e tentado de novo.
    result, sent, slept = await _dispatch_503_then_ok(monkeypatch, 4.0)
    assert slept == [4.0]
    assert sent == ["vendor/free", "vendor/free"]
    assert not any("retry skipped" in line for line in result.trace)


@respx.mock
async def test_the_trace_separates_a_candidate_cut_mid_retry_from_one_never_tried(monkeypatch):
    # log, limite (0 + 120), 1a tentativa em 1.0, checagem do backoff em 1.0,
    # 2a tentativa ja fora do prazo -- e o proximo candidato tambem.
    monkeypatch.setattr(dispatcher, "time", _Clock(0.0, 0.0, 1.0, 1.0, 10_000.0))
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(503, json={"error": {"message": "ocupado"}})
    )
    pool = UpstreamPool(SETTINGS)
    body = {**BODY, "model": "my-opus"}
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 1
    assert result.trace == [
        "vendor/free: 503 ocupado (attempt 1)",
        "vendor/free: total deadline exceeded",
        "vendor/cheap: not tried, deadline exceeded",
    ]


async def test_the_floor_does_not_raise_a_remaining_deadline_above_it(monkeypatch):
    # O lado do clamp que NAO corta: sobram 0.8s (acima do piso), e e isso que
    # vai para o httpx -- o piso so entra quando o que sobra e menor que ele.
    timeout = await _timeout_of_first_call(monkeypatch, 0.0, 0.0, 4.2)
    assert timeout["read"] == pytest.approx(0.8)


@respx.mock
async def test_the_buffered_backoff_grows_with_the_attempt_number(monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: seen.append(attempt) or 0.0)
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("recusou")
    )
    pool = UpstreamPool(SETTINGS)
    body = {**BODY, "model": "my-opus"}
    try:
        await dispatch(ShuntRequest("anthropic", body, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    # Dois candidatos, cada um com esperas 1, 2, ... ate MAX_ATTEMPTS - 1.
    per_candidate = list(range(1, dispatcher.MAX_ATTEMPTS))
    assert seen == per_candidate * 2


# --------------------------------------------------------------------------
# Effort do modelo sobrescreve o do harness (spec R3)
# --------------------------------------------------------------------------

EFFORT_SETTINGS = Settings(
    providers={
        "oai": ProviderConfig(base_url="https://api.test/v1", protocol="openai", api_key="sk"),
        "ant": ProviderConfig(
            base_url="https://api.anthropic.test", protocol="anthropic", api_key="sk"
        ),
    },
    models={
        "oai-high": ModelConfig(
            provider="oai", model="vendor/oai", context_window=64000,
            max_output_tokens=8192, effort="high",
        ),
        "oai-plain": ModelConfig(
            provider="oai", model="vendor/plain", context_window=64000, max_output_tokens=8192
        ),
        "ant-high": ModelConfig(
            provider="ant", model="claude-a", context_window=200000,
            max_output_tokens=8192, effort="high",
        ),
        "ant-plain": ModelConfig(
            provider="ant", model="claude-b", context_window=200000, max_output_tokens=8192
        ),
    },
    routes=[],
    default_model=None,
)

OAI_HIGH = Candidate(alias="oai-high", provider="oai", model="vendor/oai", protocol="openai")
OAI_PLAIN = Candidate(alias="oai-plain", provider="oai", model="vendor/plain", protocol="openai")
ANT_HIGH = Candidate(alias="ant-high", provider="ant", model="claude-a", protocol="anthropic")
ANT_PLAIN = Candidate(alias="ant-plain", provider="ant", model="claude-b", protocol="anthropic")

CHAT = {"model": "opus", "max_tokens": 64, "messages": [{"role": "user", "content": "oi"}]}


def _anthropic(**extra):
    return ShuntRequest("anthropic", {**BODY, **extra}, {})


def _openai(endpoint="chat", **extra):
    return ShuntRequest("openai", {**CHAT, **extra}, {}, endpoint=endpoint)


def test_effort_override_same_dialect_anthropic_keeps_other_output_config_keys():
    fmt = {"type": "json_schema", "schema": {"type": "object"}}
    payload = dispatcher._payload(
        _anthropic(output_config={"effort": "low", "format": fmt}), ANT_HIGH, EFFORT_SETTINGS
    )
    assert payload["output_config"] == {"effort": "high", "format": fmt}


def test_effort_override_creates_output_config_when_the_harness_sent_none():
    payload = dispatcher._payload(_anthropic(), ANT_HIGH, EFFORT_SETTINGS)
    assert payload["output_config"] == {"effort": "high"}


@pytest.mark.parametrize("bad", ["high", ["effort"], None, 3])
def test_effort_override_replaces_an_output_config_that_is_not_a_dict(bad):
    payload = dispatcher._payload(_anthropic(output_config=bad), ANT_HIGH, EFFORT_SETTINGS)
    assert payload["output_config"] == {"effort": "high"}


def test_effort_override_same_dialect_openai():
    payload = dispatcher._payload(_openai(reasoning_effort="low"), OAI_HIGH, EFFORT_SETTINGS)
    assert payload["reasoning_effort"] == "high"


def test_effort_override_drops_a_reasoning_object_that_only_carried_effort():
    payload = dispatcher._payload(_openai(reasoning={"effort": "low"}), OAI_HIGH, EFFORT_SETTINGS)
    assert payload["reasoning_effort"] == "high"
    assert "reasoning" not in payload


def test_effort_override_keeps_the_other_reasoning_keys():
    payload = dispatcher._payload(
        _openai(reasoning={"effort": "low", "summary": "auto"}), OAI_HIGH, EFFORT_SETTINGS
    )
    assert payload["reasoning_effort"] == "high"
    assert payload["reasoning"] == {"summary": "auto"}


def test_effort_override_leaves_a_reasoning_object_without_effort_alone():
    payload = dispatcher._payload(
        _openai(reasoning={"summary": "auto"}), OAI_HIGH, EFFORT_SETTINGS
    )
    assert payload["reasoning_effort"] == "high"
    assert payload["reasoning"] == {"summary": "auto"}


def test_effort_override_anthropic_caller_openai_candidate():
    payload = dispatcher._payload(
        _anthropic(output_config={"effort": "low"}), OAI_HIGH, EFFORT_SETTINGS
    )
    assert payload["reasoning_effort"] == "high"
    assert "output_config" not in payload


def test_effort_override_openai_caller_anthropic_candidate():
    payload = dispatcher._payload(_openai(reasoning_effort="low"), ANT_HIGH, EFFORT_SETTINGS)
    assert payload["output_config"] == {"effort": "high"}
    assert "reasoning_effort" not in payload


def test_effort_override_never_touches_thinking():
    thinking = {"type": "enabled", "budget_tokens": 2048}
    payload = dispatcher._payload(
        _anthropic(thinking=thinking, output_config={"effort": "low"}), ANT_HIGH, EFFORT_SETTINGS
    )
    assert payload["thinking"] == thinking
    assert payload["output_config"] == {"effort": "high"}


def test_a_model_without_effort_lets_the_harness_effort_through_untouched():
    payload = dispatcher._payload(
        _anthropic(output_config={"effort": "max"}), ANT_PLAIN, EFFORT_SETTINGS
    )
    # Mesmo dialeto: sem normalizacao, `max` segue como veio (spec R5).
    assert payload["output_config"] == {"effort": "max"}
    payload = dispatcher._payload(
        _openai(reasoning={"effort": "minimal"}), OAI_PLAIN, EFFORT_SETTINGS
    )
    assert payload["reasoning"] == {"effort": "minimal"}
    assert "reasoning_effort" not in payload


def test_a_model_without_effort_still_gets_the_translated_harness_effort():
    payload = dispatcher._payload(
        _anthropic(output_config={"effort": "max"}), OAI_PLAIN, EFFORT_SETTINGS
    )
    assert payload["reasoning_effort"] == "xhigh"


def test_a_transparent_candidate_gets_no_override():
    transparent = Candidate(
        alias="ant-high", provider="ant", model="claude-a", protocol="anthropic",
        transparent=True,
    )
    payload = dispatcher._payload(
        _anthropic(output_config={"effort": "low"}), transparent, EFFORT_SETTINGS
    )
    assert payload["output_config"] == {"effort": "low"}
    no_alias = Candidate(
        alias=None, provider="ant", model="claude-x", protocol="anthropic", transparent=True
    )
    assert "output_config" not in dispatcher._payload(_anthropic(), no_alias, EFFORT_SETTINGS)


@pytest.mark.parametrize("endpoint", ["completions", "embeddings"])
def test_completions_and_embeddings_get_no_override(endpoint):
    body = {"model": "opus", "prompt": "oi", "input": "oi"}
    payload = dispatcher._payload(
        ShuntRequest("openai", body, {}, endpoint=endpoint), OAI_HIGH, EFFORT_SETTINGS
    )
    assert "reasoning_effort" not in payload
    payload = dispatcher._payload(
        ShuntRequest("openai", {**body, "reasoning_effort": "low"}, {}, endpoint=endpoint),
        OAI_HIGH,
        EFFORT_SETTINGS,
    )
    assert payload["reasoning_effort"] == "low"


def test_the_override_never_mutates_the_harness_body():
    """A copia de `_payload` e rasa: um override que editasse o dict aninhado
    (`out["output_config"]["effort"] = ...`) mudaria o corpo do harness, e o
    proximo candidato da cadeia receberia o effort do anterior."""
    import copy

    cases = [
        (_anthropic(output_config={"effort": "low", "format": {"type": "json_schema"}}), ANT_HIGH),
        (_openai(reasoning={"effort": "low", "summary": "auto"}), OAI_HIGH),
        (_openai(reasoning={"effort": "low"}), OAI_HIGH),
        (_anthropic(output_config={"effort": "low"}), OAI_HIGH),
        (_openai(reasoning_effort="low"), ANT_HIGH),
    ]
    for req, candidate in cases:
        before = copy.deepcopy(req.body)
        dispatcher._payload(req, candidate, EFFORT_SETTINGS)
        assert req.body == before
