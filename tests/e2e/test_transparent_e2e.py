"""Modo transparente, credenciais e deteccao de dialeto, vistos de fora.

O que estes testes protegem: um modelo sem rota nao pode ser traduzido, a chave
do cliente nao pode vazar para um provedor que usa a chave do servidor, e a
resposta tem de sair no dialeto de quem perguntou.
"""

from tests.e2e.fake_provider import Scripted

ANTHROPIC_ANSWER = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-fable-5-1",
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "direto"}],
    "usage": {"input_tokens": 3, "output_tokens": 2},
}


def test_unrouted_model_goes_untranslated_to_the_original_provider(shunt, provider):
    provider.queue(Scripted(json_body=ANTHROPIC_ANSWER))
    body = shunt.post(
        "/v1/messages",
        json={
            "model": "claude-fable-5-1",
            "max_tokens": 64,
            "system": [{"type": "text", "text": "seja breve"}],
            "messages": [{"role": "user", "content": "oi"}],
        },
        headers={
            "x-api-key": "sk-ant-do-cliente",
            "anthropic-beta": "context-1m-2025-08-07",
        },
    ).json()
    sent = provider.bodies[0]
    assert provider.calls[0].url.path == "/v1/messages"
    assert sent["system"] == [{"type": "text", "text": "seja breve"}], "nao pode traduzir"
    assert body["content"][0]["text"] == "direto"


def test_transparent_mode_forwards_the_client_headers_verbatim(shunt, provider):
    provider.queue(Scripted(json_body=ANTHROPIC_ANSWER))
    shunt.post(
        "/v1/messages",
        json={
            "model": "claude-fable-5-1",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "oi"}],
        },
        headers={
            "x-api-key": "sk-ant-do-cliente",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "context-1m-2025-08-07",
        },
    )
    sent = provider.calls[0].headers
    assert sent["x-api-key"] == "sk-ant-do-cliente"
    assert sent["anthropic-beta"] == "context-1m-2025-08-07"
    assert "host" not in {k.lower() for k in sent} or sent["host"] != "testserver"


def test_routed_mode_uses_the_server_key_and_drops_the_client_one(shunt, provider):
    provider.queue(
        Scripted(
            json_body={
                "id": "c",
                "model": "qwen3-8b",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            }
        )
    )
    shunt.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "oi"}],
        },
        headers={"x-api-key": "sk-ant-do-cliente"},
    )
    sent = str(dict(provider.calls[0].headers))
    assert "sk-ant-do-cliente" not in sent


def test_model_with_no_route_and_no_declared_provider_is_refused(shunt, provider):
    response = shunt.post(
        "/v1/messages",
        json={
            "model": "modelo-sem-dono",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "oi"}],
        },
    )
    assert response.status_code == 400
    assert "provider" in response.json()["error"]["message"]
    assert provider.calls == []


def test_openai_harness_reaching_an_anthropic_provider_gets_openai_back(shunt, provider, settings):
    settings.routes = [("gpt-4o", ["fable"])]
    settings.models["fable"] = settings.models["free"].model_copy(
        update={"provider": "anthropic", "model": "claude-fable-5-1"}
    )
    provider.queue(Scripted(json_body=ANTHROPIC_ANSWER))
    body = shunt.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "oi"}]},
    ).json()
    assert provider.calls[0].url.path == "/v1/messages"
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "direto"
    assert body["model"] == "gpt-4o"


def test_models_endpoint_answers_in_the_caller_dialect(shunt):
    anthropic = shunt.get("/v1/models", headers={"anthropic-version": "2023-06-01"}).json()
    openai = shunt.get("/v1/models", headers={"authorization": "Bearer sk"}).json()
    assert anthropic["data"][0]["type"] == "model"
    assert anthropic["has_more"] is False
    assert openai["object"] == "list"
    assert "type" not in openai["data"][0]


def test_count_tokens_answers_locally_when_the_target_is_not_anthropic(shunt, provider):
    body = shunt.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-opus-4-5",
            "messages": [{"role": "user", "content": "uma frase de teste"}],
        },
    ).json()
    assert body["input_tokens"] > 0
    assert provider.calls == []


def test_count_tokens_falls_back_to_the_estimate_when_the_upstream_call_dies(
    shunt, provider, settings
):
    """Transporte que morre e a mesma situacao de um upstream que nao responde 200.

    A rota ja cai na estimativa local quando o destino devolve qualquer status
    diferente de 200; uma conexao recusada nao pode ser o unico caso que virou
    um 500 de `text/plain` no colo do cliente.
    """
    settings.routes = [("opus", ["fable"])]
    settings.models["fable"] = settings.models["free"].model_copy(
        update={"provider": "anthropic", "model": "claude-fable-5-1"}
    )
    provider.queue(Scripted(raise_transport=True))
    response = shunt.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-opus-4-5",
            "messages": [{"role": "user", "content": "uma frase de teste"}],
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["input_tokens"] > 0
