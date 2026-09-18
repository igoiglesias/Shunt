import httpx
import pytest
import respx

from app.config.settings import ModelCaps, ModelConfig, ProviderConfig, Settings
from app.core.dispatcher import ShuntRequest, dispatch, outbound_headers
from app.core.resolver import Candidate
from app.core.upstream import UpstreamPool

SETTINGS = Settings(
    providers={"openrouter": ProviderConfig(base_url="https://api.test/v1",
                                            protocol="openai", api_key_env=None)},
    models={
        "free": ModelConfig(provider="openrouter", model="vendor/free",
                            context_window=64000, max_output_tokens=8192),
        "cheap": ModelConfig(provider="openrouter", model="vendor/cheap",
                             context_window=64000, max_output_tokens=8192),
    },
    routes=[("opus", ["free", "cheap"])],
    default_model=None,
)

BODY = {"model": "claude-opus-4-5", "max_tokens": 64,
        "messages": [{"role": "user", "content": "oi"}]}


def ok_payload(model):
    return {"id": "chatcmpl-1", "model": model,
            "choices": [{"message": {"content": "pronto"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


@respx.mock
async def test_first_candidate_answers_and_response_is_translated():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_payload("vendor/free")))
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
        side_effect=[httpx.Response(400, json={"error": {"message": "contexto"}}),
                     httpx.Response(200, json=ok_payload("vendor/cheap"))])
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 2
    assert result.real_model == "vendor/cheap"


@respx.mock
async def test_transport_error_retries_the_same_candidate():
    route = respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[httpx.ConnectError("recusou"),
                     httpx.Response(200, json=ok_payload("vendor/free"))])
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
        providers={"anthropic": ProviderConfig(base_url="https://api.anthropic.test",
                                               protocol="anthropic", api_key_env=None)},
        models={"fable": ModelConfig(provider="anthropic", model="claude-fable-5-1",
                                     context_window=200000, max_output_tokens=8192)},
        routes=[("gpt-4o", ["fable"])], default_model=None,
    )
    respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "id": "msg_1", "model": "claude-fable-5-1", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "pronto"}],
            "usage": {"input_tokens": 4, "output_tokens": 2}}))
    pool = UpstreamPool(settings)
    try:
        result = await dispatch(ShuntRequest("openai", {
            "model": "gpt-4o", "messages": [{"role": "user", "content": "oi"}]},
            {}), settings, pool)
    finally:
        await pool.aclose()
    assert result.body["object"] == "chat.completion"
    assert result.body["choices"][0]["message"]["content"] == "pronto"
    assert result.body["model"] == "gpt-4o"


@respx.mock
async def test_exhausted_chain_returns_the_last_error_and_names_every_candidate():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "nao deu"}}))
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 400
    assert result.body["type"] == "error"
    assert result.body["error"]["type"] == "invalid_request_error"
    assert "free" in result.body["error"]["message"]
    assert "cheap" in result.body["error"]["message"]


# --- credenciais: o ponto mais sensível do proxy -----------------------------

KEYED = Settings(
    providers={"openrouter": ProviderConfig(base_url="https://api.test/v1",
                                            protocol="openai", api_key_env="SHUNT_TEST_KEY")},
    models={"free": ModelConfig(provider="openrouter", model="vendor/free",
                                context_window=64000, max_output_tokens=8192)},
    routes=[("opus", ["free"])], default_model=None,
)

TRANSPARENT = Settings(
    providers={"anthropic": ProviderConfig(base_url="https://api.anthropic.test",
                                           protocol="anthropic", api_key_env="SHUNT_TEST_KEY")},
    models={}, routes=[], default_model=None,
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
    monkeypatch.setenv("SHUNT_TEST_KEY", "sk-da-config")
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_payload("vendor/free")))
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
    monkeypatch.setenv("SHUNT_TEST_KEY", "sk-da-config")
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "id": "msg_1", "model": "claude-sonnet-9", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "ok"}], "usage": {}}))
    pool = UpstreamPool(TRANSPARENT)
    body = {**BODY, "model": "claude-sonnet-9"}
    try:
        result = await dispatch(
            ShuntRequest("anthropic", body, dict(CLIENT_HEADERS)), TRANSPARENT, pool)
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
    monkeypatch.setenv("SHUNT_TEST_KEY", "sk-da-config")
    candidate = Candidate(alias=None, provider="anthropic", model="m", protocol="anthropic")
    headers = outbound_headers(ShuntRequest("anthropic", BODY, dict(CLIENT_HEADERS)),
                               candidate, TRANSPARENT)
    assert headers["x-api-key"] == "sk-da-config"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in headers
    assert headers["content-type"] == "application/json"


def test_outbound_headers_routed_without_a_configured_key_sends_no_credential():
    candidate = Candidate(alias="free", provider="openrouter", model="vendor/free",
                          protocol="openai")
    headers = outbound_headers(ShuntRequest("anthropic", BODY, dict(CLIENT_HEADERS)),
                               candidate, SETTINGS)
    assert headers == {"content-type": "application/json"}


def test_outbound_headers_transparent_drops_exactly_three_headers():
    candidate = Candidate(alias=None, provider="anthropic", model="m",
                          protocol="anthropic", transparent=True)
    headers = outbound_headers(ShuntRequest("anthropic", BODY, {"Host": "x", "Content-Length": "1",
                                                                "Accept-Encoding": "gzip",
                                                                "X-Api-Key": "sk"}),
                               candidate, TRANSPARENT)
    assert headers == {"X-Api-Key": "sk"}


# --- escolha de caminho, corpo e tradução ------------------------------------

ANTHROPIC_SETTINGS = Settings(
    providers={"anthropic": ProviderConfig(base_url="https://api.anthropic.test",
                                           protocol="anthropic", api_key_env=None)},
    models={"fable": ModelConfig(provider="anthropic", model="claude-fable-5-1",
                                 context_window=200000, max_output_tokens=8192)},
    routes=[("gpt-4o", ["fable"]), ("claude-opus-4-5", ["fable"])], default_model=None,
)


@respx.mock
async def test_endpoint_without_a_path_on_the_provider_protocol_is_skipped():
    route = respx.post("https://api.anthropic.test/v1/messages")
    pool = UpstreamPool(ANTHROPIC_SETTINGS)
    try:
        result = await dispatch(
            ShuntRequest("openai", {"model": "gpt-4o", "input": "oi"}, {},
                         endpoint="embeddings"),
            ANTHROPIC_SETTINGS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 0
    assert result.status == 502
    assert "endpoint not supported" in result.body["error"]["message"]
    assert result.trace == ["fable: endpoint not supported"]


@respx.mock
async def test_the_output_cap_comes_from_the_candidate_not_from_the_client():
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "id": "msg_1", "model": "claude-fable-5-1", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "ok"}], "usage": {}}))
    pool = UpstreamPool(ANTHROPIC_SETTINGS)
    try:
        await dispatch(ShuntRequest("openai", {
            "model": "gpt-4o", "max_tokens": 999999,
            "messages": [{"role": "user", "content": "oi"}]}, {}),
            ANTHROPIC_SETTINGS, pool)
    finally:
        await pool.aclose()
    import json as _json
    sent = _json.loads(route.calls[0].request.content)
    assert sent["max_tokens"] == 8192
    assert sent["model"] == "claude-fable-5-1"


@respx.mock
async def test_same_protocol_passes_the_body_through_rewriting_only_the_model():
    upstream = {"id": "msg_1", "model": "claude-fable-5-1", "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1}}
    route = respx.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, json=upstream))
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
    providers={"openrouter": ProviderConfig(base_url="https://api.test/v1",
                                            protocol="openai", api_key_env=None)},
    models={"free": ModelConfig(provider="openrouter", model="vendor/free",
                                supports=ModelCaps(tools=False),
                                context_window=64000, max_output_tokens=8192)},
    routes=[("opus", ["free"])], default_model=None,
)


@respx.mock
async def test_a_chain_emptied_by_the_capability_filter_returns_400_without_calling_anyone():
    route = respx.post("https://api.test/v1/chat/completions")
    pool = UpstreamPool(NO_TOOLS)
    body = {**BODY, "tools": [{"name": "grep", "input_schema": {"type": "object"}}]}
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), NO_TOOLS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 0
    assert result.status == 400
    assert result.real_model is None
    assert "no candidate can serve this request" in result.body["error"]["message"]
    assert result.trace == ["free: no tool support"]


class _Clock:
    """Relógio monotônico falso: devolve os valores dados, depois repete o último."""

    def __init__(self, *values):
        self._values = list(values)

    def monotonic(self):
        return self._values.pop(0) if len(self._values) > 1 else self._values[0]


@respx.mock
async def test_the_total_deadline_cuts_the_chain_short(monkeypatch):
    import app.core.dispatcher as dispatcher

    monkeypatch.setattr(dispatcher, "time", _Clock(0.0, 1.0, 10_000.0))
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "nao deu"}}))
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 1
    assert "cheap: total deadline exceeded" in result.trace
    assert "free: 400 (attempt 1)" in result.trace


@respx.mock
async def test_429_retries_the_same_candidate_when_retry_after_fits_the_budget(monkeypatch):
    import app.core.dispatcher as dispatcher

    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    route = respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[httpx.Response(429, headers={"retry-after": "1"}, json={}),
                     httpx.Response(200, json=ok_payload("vendor/free"))])
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
        side_effect=[httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"},
                                    json={}),
                     httpx.Response(200, json=ok_payload("vendor/cheap"))])
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == 2
    assert result.real_model == "vendor/cheap"


@respx.mock
async def test_a_non_json_error_body_becomes_the_message_verbatim(monkeypatch):
    import app.core.dispatcher as dispatcher

    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(403, text="cota estourada"))
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 403
    assert "cota estourada" in result.body["error"]["message"]


@respx.mock
async def test_a_candidate_that_only_ever_fails_transport_is_retried_to_the_limit(monkeypatch):
    import app.core.dispatcher as dispatcher

    monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)
    route = respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("recusou"))
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert route.call_count == dispatcher.MAX_ATTEMPTS * 2
    assert result.status == 502
    assert "recusou" in result.body["error"]["message"]
    assert result.real_model is None


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
            400, content=b"{nao e json", headers={"content-type": "application/json"}))
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 400
    assert "{nao e json" in result.body["error"]["message"]


@respx.mock
async def test_a_json_error_body_that_is_not_an_object_falls_back_to_the_raw_text():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json=["explodiu"]))
    pool = UpstreamPool(SETTINGS)
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 400
    assert "explodiu" in result.body["error"]["message"]


MIXED = Settings(
    providers={"openrouter": ProviderConfig(base_url="https://api.test/v1",
                                            protocol="openai", api_key_env=None)},
    models={
        "free": ModelConfig(provider="openrouter", model="vendor/free",
                            supports=ModelCaps(tools=False),
                            context_window=64000, max_output_tokens=8192),
        "cheap": ModelConfig(provider="openrouter", model="vendor/cheap",
                             context_window=64000, max_output_tokens=8192),
        "backup": ModelConfig(provider="openrouter", model="vendor/backup",
                              context_window=64000, max_output_tokens=8192),
    },
    routes=[("opus", ["free", "cheap", "backup"])], default_model=None,
)


@respx.mock
async def test_a_successful_result_still_reports_what_happened_before_it():
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[httpx.Response(400, json={"error": {"message": "nao deu"}}),
                     httpx.Response(200, json=ok_payload("vendor/backup"))])
    pool = UpstreamPool(MIXED)
    body = {**BODY, "tools": [{"name": "grep", "input_schema": {"type": "object"}}]}
    try:
        result = await dispatch(ShuntRequest("anthropic", body, {}), MIXED, pool)
    finally:
        await pool.aclose()
    assert result.real_model == "vendor/backup"
    assert result.trace == ["free: no tool support", "cheap: 400 (attempt 1)"]
