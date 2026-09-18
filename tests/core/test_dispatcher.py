import httpx
import pytest
import respx

from app.config.settings import ModelCaps, ModelConfig, ProviderConfig, Settings
from app.core.dispatcher import ShuntRequest, dispatch, outbound_headers
from app.core.resolver import Candidate
from app.core.upstream import UpstreamPool

SETTINGS = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key_env=None
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
                base_url="https://api.anthropic.test", protocol="anthropic", api_key_env=None
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
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key_env="SHUNT_TEST_KEY"
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
            api_key_env="SHUNT_TEST_KEY",
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
    monkeypatch.setenv("SHUNT_TEST_KEY", "sk-da-config")
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
    monkeypatch.setenv("SHUNT_TEST_KEY", "sk-da-config")
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
    monkeypatch.setenv("SHUNT_TEST_KEY", "sk-da-config")
    candidate = Candidate(alias=None, provider="anthropic", model="m", protocol="anthropic")
    headers = outbound_headers(
        ShuntRequest("anthropic", BODY, dict(CLIENT_HEADERS)), candidate, TRANSPARENT
    )
    assert headers["x-api-key"] == "sk-da-config"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in headers
    assert headers["content-type"] == "application/json"


def test_outbound_headers_routed_without_a_configured_key_sends_no_credential():
    candidate = Candidate(
        alias="free", provider="openrouter", model="vendor/free", protocol="openai"
    )
    headers = outbound_headers(
        ShuntRequest("anthropic", BODY, dict(CLIENT_HEADERS)), candidate, SETTINGS
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
            {"Host": "x", "Content-Length": "1", "Accept-Encoding": "gzip", "X-Api-Key": "sk"},
        ),
        candidate,
        TRANSPARENT,
    )
    assert headers == {"X-Api-Key": "sk"}


# --- escolha de caminho, corpo e tradução ------------------------------------

ANTHROPIC_SETTINGS = Settings(
    providers={
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test", protocol="anthropic", api_key_env=None
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
        result = await dispatch(
            ShuntRequest("openai", {"model": "gpt-4o", "input": "oi"}, {}, endpoint="embeddings"),
            ANTHROPIC_SETTINGS,
            pool,
        )
    finally:
        await pool.aclose()
    assert route.call_count == 0
    assert result.status == 502
    assert "endpoint not supported" in result.body["error"]["message"]
    assert result.trace == ["fable: endpoint not supported"]


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
            base_url="https://api.test/v1", protocol="openai", api_key_env=None
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
    from app.core import dispatcher

    monkeypatch.setattr(dispatcher, "time", _Clock(0.0, 1.0, 10_000.0))
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "nao deu"}})
    )
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
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
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
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
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
    try:
        await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
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
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
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
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 400
    assert "explodiu" in result.body["error"]["message"]


MIXED = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key_env=None
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
    assert result.trace == ["free: no tool support", "cheap: 400 (attempt 1)"]


# --- rodada 1 de correcoes ---------------------------------------------------

VISION = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key_env=None
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
    assert result.trace == ["cego: no vision support"]


ANTHROPIC_PAIR = Settings(
    providers={
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test", protocol="anthropic", api_key_env=None
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
    assert "a: unreadable body (attempt 1)" in result.trace


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
    assert route.call_count == 2
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
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
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
    try:
        result = await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert result.status == 400
    assert '"code"' in result.body["error"]["message"]


MIXED_PROTOCOLS = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key_env=None
        ),
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test", protocol="anthropic", api_key_env=None
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
    assert any("openai_um: request translation failed" in line for line in result.trace)


BLIND_THEN_SEEING = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key_env=None
        ),
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test", protocol="anthropic", api_key_env=None
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
    assert result.trace[0].startswith("probe (cego): request translation failed:")
    assert result.trace[1:] == ["cego: no vision support"]


PATH_SETTINGS = Settings(
    providers={
        "openrouter": ProviderConfig(
            base_url="https://api.test/v1", protocol="openai", api_key_env=None
        ),
        "anthropic": ProviderConfig(
            base_url="https://api.anthropic.test", protocol="anthropic", api_key_env=None
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
            base_url="https://gateway.exemplo/v1", protocol="openai", api_key_env="SHUNT_TEST_KEY"
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
    monkeypatch.setenv("SHUNT_TEST_KEY", "sk-da-config")
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
    monkeypatch.setenv("SHUNT_TEST_KEY", "sk-da-config")
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
    monkeypatch.setenv("SHUNT_TEST_KEY", "sk-da-config")
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
    assert "free: unreadable body (attempt 1)" in result.trace


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
    assert route.call_count == 2


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
