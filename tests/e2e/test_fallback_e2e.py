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
    provider.queue(
        Scripted(status=400, json_body={"error": {"message": "nao deu"}}),
        Scripted(status=400, json_body={"error": {"message": "nao deu"}}),
    )
    body = shunt.post("/v1/messages", json=ASK).json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "qwen" in body["error"]["message"]
    assert "free" in body["error"]["message"]


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
    settings.routes = [("opus", ["mudo"])]
    body = shunt.post(
        "/v1/messages",
        json={**ASK, "tools": [{"name": "read", "input_schema": {"type": "object"}}]},
    ).json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "no tool support" in body["error"]["message"]
    assert provider.calls == [], "candidato incapaz nao pode ser chamado"


def test_transport_failure_names_the_exception_class_in_the_error(shunt, provider):
    """Medido contra llama.cpp real: um `ReadTimeout` do httpx tem `str()` vazio,

    e a mensagem de erro chegava ao cliente como `" - tried: qwen-local:  "`,
    sem dizer nada. O nome da classe e o unico pedaco que sempre existe.
    """
    provider.queue(*[Scripted(raise_transport=True) for _ in range(6)])
    body = shunt.post("/v1/messages", json=ASK).json()
    assert "ConnectError" in body["error"]["message"]


def test_transport_failure_in_a_stream_names_the_exception_class(shunt, provider):
    provider.queue(*[Scripted(raise_transport=True) for _ in range(6)])
    with shunt.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as response:
        body = "".join(response.iter_text())
    assert "event: error" in body
    assert "ConnectError" in body
