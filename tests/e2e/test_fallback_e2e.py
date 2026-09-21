"""Falha e recuperacao vistas de fora: o que o cliente recebe quando o de cima cai.

Cada teste aqui mede a decisao pelo efeito na rede -- quantas chamadas sairam e
qual candidato respondeu -- e nao pelo estado interno do dispatcher.
"""

from tests.e2e.fake_provider import Scripted, sse_chunks

OK = {
    "id": "chatcmpl-ok",
    "model": "vendor/free",
    "choices": [{"message": {"content": "pronto"}, "finish_reason": "stop"}],
}

ASK = {
    "model": "claude-opus-4-5",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "oi"}],
}


def test_400_skips_to_the_next_candidate_without_retrying(shunt, provider):
    provider.queue(
        Scripted(status=400, json_body={"error": {"message": "contexto"}}),
        Scripted(json_body=OK),
    )
    response = shunt.post("/v1/messages", json=ASK)
    assert response.status_code == 200
    assert len(provider.calls) == 2
    assert response.headers["x-shunt-model"] == "vendor/free"


def test_429_with_short_retry_after_retries_the_same_candidate(shunt, provider):
    provider.queue(
        Scripted(
            status=429,
            json_body={"error": {"message": "devagar"}},
            headers={"retry-after": "1"},
        ),
        Scripted(json_body={**OK, "model": "qwen3-8b"}),
    )
    response = shunt.post("/v1/messages", json=ASK)
    assert response.status_code == 200
    assert response.headers["x-shunt-model"] == "qwen3-8b", "devia ter repetido no mesmo"


def test_429_with_long_retry_after_skips_to_the_next_candidate(shunt, provider):
    provider.queue(
        Scripted(
            status=429, json_body={"error": {"message": "fila"}}, headers={"retry-after": "120"}
        ),
        Scripted(json_body=OK),
    )
    assert shunt.post("/v1/messages", json=ASK).headers["x-shunt-model"] == "vendor/free"


def test_transport_error_retries_then_moves_on(shunt, provider):
    provider.queue(
        Scripted(raise_transport=True),
        Scripted(raise_transport=True),
        Scripted(raise_transport=True),
        Scripted(json_body=OK),
    )
    response = shunt.post("/v1/messages", json=ASK)
    assert response.status_code == 200
    assert response.headers["x-shunt-model"] == "vendor/free"


def test_exhausted_chain_answers_an_anthropic_error_naming_every_candidate(shunt, provider):
    # Tres candidatos desde que o degrau de baixo entrou na cadeia resolvida:
    # `qwen`, `free` e o modelo pedido em modo transparente. Um transporte que
    # cai e repetido `MAX_ATTEMPTS` vezes por candidato.
    provider.queue(
        Scripted(status=400, json_body={"error": {"message": "nao deu"}}),
        Scripted(status=400, json_body={"error": {"message": "nao deu"}}),
        Scripted(status=400, json_body={"error": {"message": "nao deu"}}),
    )
    body = shunt.post("/v1/messages", json=ASK).json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "qwen" in body["error"]["message"]
    assert "free" in body["error"]["message"]
    assert "claude-opus-4-5" in body["error"]["message"]


def test_error_inside_a_200_stream_falls_back_before_the_client_sees_anything(shunt, provider):
    provider.queue(
        Scripted(sse=[b'data: {"error": {"message": "sem credito"}}\n\n']),
        Scripted(
            sse=sse_chunks(
                {"choices": [{"delta": {"content": "segundo"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            )
        ),
    )
    with shunt.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as response:
        body = "".join(response.iter_text())
    assert "sem credito" not in body
    assert "segundo" in body
    assert body.count("event: message_start") == 1, "nao pode reabrir a mensagem"


def test_error_after_the_first_event_is_reported_in_the_anthropic_shape(shunt, provider):
    provider.queue(
        Scripted(
            sse=[
                b'data: {"choices": [{"delta": {"content": "comecou"}}]}\n\n',
                b'data: {"error": {"message": "caiu no meio"}}\n\n',
            ]
        )
    )
    with shunt.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as response:
        body = "".join(response.iter_text())
    assert "event: error" in body
    assert "caiu no meio" in body
    assert len(provider.calls) == 1, "depois do primeiro evento nao ha fallback"


def test_no_capable_candidate_answers_400_saying_why_each_was_dropped(shunt, provider, settings):
    """O degrau de baixo transparente so existe quando da para deduzir o
    provedor do nome pedido. Sem isso -- e sem `default_model` -- a cadeia
    esvaziada pelo filtro continua virando 400 com o motivo de cada descarte."""
    settings.routes = [("opus", ["mudo"])]
    body = shunt.post(
        "/v1/messages",
        json={
            **ASK,
            "model": "acme-super-opus",
            "tools": [{"name": "read", "input_schema": {"type": "object"}}],
        },
    ).json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "no tool support" in body["error"]["message"]
    assert provider.calls == [], "candidato incapaz nao pode ser chamado"


def test_transport_failure_names_the_exception_class_in_the_error(shunt, provider):
    """Medido contra llama.cpp real: um `ReadTimeout` do httpx tem `str()` vazio,

    e a mensagem de erro chegava ao cliente como `" - tried: qwen-local:  "`,
    sem dizer nada. O nome da classe e o unico pedaco que sempre existe.
    """
    # Tres candidatos (`qwen`, `free` e o degrau de baixo transparente), cada um
    # repetido `MAX_ATTEMPTS` = 3 vezes num transporte que cai.
    provider.queue(*[Scripted(raise_transport=True) for _ in range(9)])
    body = shunt.post("/v1/messages", json=ASK).json()
    assert "ConnectError" in body["error"]["message"]


def test_transport_failure_in_a_stream_names_the_exception_class(shunt, provider):
    provider.queue(*[Scripted(raise_transport=True) for _ in range(6)])
    with shunt.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as response:
        body = "".join(response.iter_text())
    assert "event: error" in body
    assert "ConnectError" in body


def test_candidate_whose_credential_is_unset_is_skipped_not_called_unauthenticated(
    shunt, provider, settings, monkeypatch
):
    """Declarar `api_key_env` e nao ter a variavel e erro de instalacao.

    Chamar o provedor assim mando um pedido sem credencial, que volta 401 e
    gasta uma tentativa para dizer o que a configuracao ja sabia. Um provedor
    que declara `api_key_env=None` e outro caso: ele nao quer credencial, e
    continua sendo chamado.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings.routes = [("opus", ["free", "qwen"])]
    provider.queue(Scripted(json_body={**OK, "model": "qwen3-8b"}))
    response = shunt.post("/v1/messages", json=ASK)
    assert response.status_code == 200
    assert response.headers["x-shunt-model"] == "qwen3-8b"
    assert len(provider.calls) == 1, "o candidato sem credencial nao pode ser chamado"
    assert provider.calls[0].url.host == "fake.local", "a chamada foi para o provedor errado"


def test_the_error_names_the_environment_variable_that_is_missing(shunt, settings, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings.routes = [("opus", ["free"])]
    body = shunt.post("/v1/messages", json=ASK).json()
    assert "OPENROUTER_API_KEY" in body["error"]["message"]


def test_a_stream_also_skips_the_candidate_whose_credential_is_unset(
    shunt, provider, settings, monkeypatch
):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings.routes = [("opus", ["free", "qwen"])]
    provider.queue(
        Scripted(
            sse=sse_chunks(
                {"choices": [{"delta": {"content": "do local"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            )
        )
    )
    with shunt.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as response:
        body = "".join(response.iter_text())
    assert len(provider.calls) == 1, "o candidato sem credencial nao pode ser chamado"
    assert provider.calls[0].url.host == "fake.local", "a chamada foi para o provedor errado"
    assert "do local" in body
    assert "OPENROUTER_API_KEY" not in body


def test_a_transparent_candidate_is_called_even_with_no_server_credential(
    shunt, provider, monkeypatch
):
    """Transparente usa a chave do CLIENTE, entao a do servidor pode faltar.

    Sem esta distincao, a regra de credencial ausente mataria justamente o modo
    em que o proxy nao deve ter credencial nenhuma.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider.queue(
        Scripted(
            json_body={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-fable-5-1",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "direto"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
    )
    body = shunt.post(
        "/v1/messages",
        json={
            "model": "claude-fable-5-1",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "oi"}],
        },
        headers={"x-api-key": "sk-ant-do-cliente"},
    ).json()
    assert len(provider.calls) == 1
    assert body["content"][0]["text"] == "direto"
    assert provider.calls[0].headers["x-api-key"] == "sk-ant-do-cliente"
