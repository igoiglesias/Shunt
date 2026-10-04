"""Falha e recuperacao vistas de fora: o que o cliente recebe quando o de cima cai.

Cada teste aqui mede a decisao pelo efeito na rede -- quantas chamadas sairam e
qual candidato respondeu -- e nao pelo estado interno do dispatcher.
"""

import logging
import time
from threading import Thread

import pytest
from fastapi.testclient import TestClient

from app.config.settings import ModelCaps, ModelConfig
from app.core import dispatcher
from app.core.upstream import UpstreamPool
from app.main import app as shunt_app
from tests.e2e.fake_provider import Scripted, ScriptedTransport, sse_chunks

OK = {
    "id": "chatcmpl-ok",
    "model": "vendor/free",
    "choices": [{"message": {"content": "pronto"}, "finish_reason": "stop"}],
}

OPENAI_COMPLETION = {
    "id": "cmpl_1",
    "object": "text_completion",
    "model": "qwen3-8b",
    "choices": [{"text": " completado", "finish_reason": "stop", "index": 0}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


@pytest.fixture
def trace_lines():
    """As linhas de log do Shunt, com o `attempts` de cada requisicao.

    O logger `shunt` tem `propagate = False` (`observability.py:272`, para nao
    duplicar a linha no handler do uvicorn), e o `caplog` do pytest so ve o que
    chega ao logger raiz -- entao ele nao captura nada. Anexar um handler
    proprio ao logger e o jeito de ler o rastro que o operador le.
    """
    logger = logging.getLogger("shunt")
    lines: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    capture = _Capture()
    logger.addHandler(capture)
    try:
        yield lines
    finally:
        logger.removeHandler(capture)


class _Clock:
    """Relogio monotonic falso: devolve os valores dados, depois repete o ultimo.

    O unico jeito deterministico de o deadline cortar a cadeia: sem ele, a
    unica alternativa e dormir o atraso de verdade (`dispatcher.py:704` le
    `time.monotonic()`). Copiado do teste unitario (`tests/core/test_dispatcher.py:719`).
    """

    def __init__(self, *values: float) -> None:
        self._values = list(values)

    def monotonic(self) -> float:
        return self._values.pop(0) if len(self._values) > 1 else self._values[0]


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


def test_a_400_arriving_in_the_response_headers_of_a_stream_falls_to_the_next_candidate(
    shunt, provider
):
    """`dispatcher.py:1271`: o `classify` roda ANTES de qualquer byte ir para o fio.

    Um 400 nao e RETRY (`attempt.py:44`), entao o loop de tentativas para na
    primeira e a cadeia segue. O cliente nunca ve o erro do provedor de cima.
    """
    provider.queue(
        Scripted(status=400, json_body={"error": {"message": "contexto"}}),
        Scripted(sse=sse_chunks({"choices": [{"delta": {"content": "do segundo"}}]})),
    )
    with shunt.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert len(provider.calls) == 2, "o primeiro candidato e abandonado, nao reenviado"
    assert provider.calls[0].url.host == "fake.local", "a primeira chamada foi para o qwen"
    assert provider.calls[1].url.host == "fake.openrouter", "a segunda foi para o free"
    assert "do segundo" in body
    assert "contexto" not in body, "nenhum byte do erro do provedor pode vazar"


def test_a_500_arriving_in_the_response_headers_of_a_stream_is_retried_then_abandoned(
    shunt, provider
):
    """Ao contrario do 400, o 500 E retry (`attempt.py:42`): esgota as tentativas
    no mesmo candidato antes de cair para o proximo (`dispatcher.py:1279`)."""
    provider.queue(
        Scripted(status=500, json_body={"error": {"message": "servidor fora"}}),
        Scripted(status=500, json_body={"error": {"message": "servidor fora"}}),
        Scripted(status=500, json_body={"error": {"message": "servidor fora"}}),
        Scripted(sse=sse_chunks({"choices": [{"delta": {"content": "do segundo"}}]})),
    )
    with shunt.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert len(provider.calls) == 4, "3 tentativas no qwen e a queda para o free"
    assert all(call.url.host == "fake.local" for call in provider.calls[:3])
    assert provider.calls[3].url.host == "fake.openrouter"
    assert "do segundo" in body
    assert "servidor fora" not in body


def test_a_429_with_a_long_retry_after_in_a_stream_skips_to_the_next_candidate(
    shunt, provider
):
    """429 so e retry quando o `retry-after` cabe no budget (`attempt.py:36-38`;

    `RETRY_AFTER_BUDGET` = 5s em `config.py:18`). Acima disso e SKIP: reenviar
    faria o cliente esperar a fila do provedor, e o proximo candidato atende.
    """
    provider.queue(
        Scripted(
            status=429,
            json_body={"error": {"message": "fila cheia"}},
            headers={"retry-after": "120"},
        ),
        Scripted(sse=sse_chunks({"choices": [{"delta": {"content": "do segundo"}}]})),
    )
    with shunt.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert len(provider.calls) == 2, "o 429 acima do budget nao repete no mesmo host"
    assert provider.calls[0].url.host == "fake.local"
    assert provider.calls[1].url.host == "fake.openrouter"
    assert "do segundo" in body
    assert "fila cheia" not in body


def test_the_503_to_200_recovery_does_not_sleep_the_real_backoff(shunt, provider, monkeypatch):
    """Sentinela: o e2e roda a cadeia 503 -> 200 sem dormir o backoff real.

    Espiona `dispatcher.asyncio.sleep` e soma o que teria sido esperado. Sem a
    fixture `no_real_backoff` (autouse em `tests/e2e/conftest.py`), a soma e
    >= 1.0 porque o dispatcher dorme o backoff de verdade a cada retry.
    """
    waits: list[float] = []
    real_sleep = dispatcher.asyncio.sleep

    async def spy_sleep(seconds):
        waits.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(dispatcher.asyncio, "sleep", spy_sleep)
    provider.queue(
        Scripted(status=503, json_body={"error": {"message": "fora do ar"}}),
        Scripted(json_body=OK),
    )
    response = shunt.post("/v1/messages", json=ASK)
    assert response.status_code == 200
    assert sum(waits) == 0


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


def test_a_connection_drop_after_the_first_event_is_reported_in_the_anthropic_shape(
    shunt, provider
):
    """`dispatcher.py:1327-1347`: depois do 200 e do primeiro evento, a conexao
    caiu. Nao ha mais fallback -- o `message_start` ja foi para o fio e um
    segundo provedor nao pode substitui-lo. O cliente recebe o erro na forma
    Anthropic (`_stream_error`), no MESMO stream, sem reabrir a mensagem.
    """
    valid = sse_chunks(
        {"choices": [{"delta": {"content": "comecou"}}]},
        {"choices": [{"delta": {"content": " nao acabou"}}]},
    )
    # O corte precisa cair entre dois eventos: o SSEDecoder so conclui um
    # evento na linha em branco, e cortar no meio deixaria o ultimo evento
    # no decoder -- `state.started` ficaria False e haveria fallback.
    served = len(valid[0])
    provider.queue(Scripted(sse=valid, raise_after=served))
    with shunt.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as response:
        body = "".join(response.iter_text())
    assert "comecou" in body, "os bytes validos sao entregues antes da queda"
    assert "event: error" in body
    assert "ReadError" in body, "o nome da classe e a parte que sempre existe"
    assert len(provider.calls) == 1, "depois do primeiro evento nao ha fallback"


def test_no_valid_event_within_the_first_event_deadline_falls_back(shunt, provider, monkeypatch):
    """`dispatcher.py:1309`: o provedor aceitou a conexao e calou. Sem um evento
    valido no prazo, o candidato e abandonado e o proximo atende.

    O prazo e lido do modulo `dispatcher` (`config.py:20`, default 20s); aqui ele
    e rebaixado para a leitura nao esperar o silencio de verdade. O atraso e
    aplicado no CONSUMO da stream (`Scripted.first_byte_delay`), porque o
    `ASGITransport` do httpx bufferiza a resposta inteira e um atraso na
    producao sumiria dentro do `send`.
    """
    monkeypatch.setattr(dispatcher, "FIRST_EVENT_DEADLINE", 0.05)
    provider.queue(
        Scripted(
            sse=sse_chunks({"choices": [{"delta": {"content": "tarde demais"}}]}),
            first_byte_delay=0.2,
        ),
        Scripted(sse=sse_chunks({"choices": [{"delta": {"content": "do segundo"}}]})),
    )
    with shunt.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert len(provider.calls) == 2, "o candidato calado e abandonado"
    assert "tarde demais" not in body, "o stream atrasado nao pode chegar ao cliente"
    assert "do segundo" in body


def test_a_connection_drop_before_the_first_event_falls_back(shunt, provider):
    """O mesmo corte ANTES do primeiro evento valido e uma queda gratuita: nada
    chegou ao cliente, e o proximo candidato ainda esta disponivel."""
    provider.queue(
        Scripted(sse=[b": ping\n\n"], raise_after=1),
        Scripted(sse=sse_chunks({"choices": [{"delta": {"content": "do segundo"}}]})),
    )
    with shunt.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as response:
        body = "".join(response.iter_text())
    assert len(provider.calls) == 2, "antes do primeiro evento o candidato e abandonado"
    assert "do segundo" in body


def test_a_request_with_an_image_drops_every_blind_candidate(shunt, provider, settings):
    """`capabilities.py:93`: so sobra o degrau de baixo, e ele tambem nao existe
    quando o nome pedido nao deixa deduzir o provedor -- a cadeia esvaziada
    vira 400 com o motivo de cada descarte."""
    settings.routes = [("opus", ["mudo"])]
    body = shunt.post(
        "/v1/messages",
        json={
            **ASK,
            "model": "acme-super-opus",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "o que e isso?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgoAAAANSUhEUg==",
                            },
                        },
                    ],
                }
            ],
        },
    ).json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "no vision support" in body["error"]["message"]
    assert provider.calls == [], "candidato sem vision nao pode ser chamado"


def test_a_request_with_an_image_falls_to_the_candidate_that_can_see(shunt, provider, settings):
    """O filtro derruba o cego e a cadeia segue para quem enxerga: o pedido
    chega ao segundo candidato, sem retry cego no primeiro."""
    settings.models["vidente"] = ModelConfig(
        provider="openrouter",
        model="vendor/vidente",
        supports=ModelCaps(tools=True, streaming=True, vision=True),
        context_window=64000,
        max_output_tokens=8192,
    )
    settings.routes = [("opus", ["mudo", "vidente"])]
    provider.queue(Scripted(json_body={**OK, "model": "vendor/vidente"}))
    response = shunt.post(
        "/v1/messages",
        json={
            **ASK,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "o que e isso?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgoAAAANSUhEUg==",
                            },
                        },
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200
    assert response.headers["x-shunt-model"] == "vendor/vidente"
    assert len(provider.calls) == 1, "o cego foi derrubado, o vidente atendeu"


def test_a_streaming_request_drops_the_candidate_that_cannot_stream(shunt, provider, settings):
    """`capabilities.py:95`: `streaming=False` na ficha do modelo derruba o
    candidato de um pedido `stream: true`."""
    settings.models["surdo"] = ModelConfig(
        provider="openrouter",
        model="vendor/surdo",
        supports=ModelCaps(tools=True, streaming=False, vision=True),
        context_window=64000,
        max_output_tokens=8192,
    )
    settings.routes = [("opus", ["surdo", "free"])]
    provider.queue(
        Scripted(sse=sse_chunks({"choices": [{"delta": {"content": "do segundo"}}]}))
    )
    with shunt.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert len(provider.calls) == 1, "o surdo foi derrubado antes de qualquer chamada"
    assert provider.calls[0].url.host == "fake.openrouter", "a chamada foi para o free"
    assert "do segundo" in body


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
    shunt, provider, settings
):
    """`api_key=None` num provedor roteado e erro de instalacao.

    Chamar o provedor assim mando um pedido sem credencial, que volta 401 e
    gasta uma tentativa para dizer o que a configuracao ja sabia. Um candidato
    transparente e outro caso: ele usa a chave do cliente, e continua sendo
    chamado sem credencial do servidor.
    """
    settings.providers["openrouter"].api_key = None
    settings.routes = [("opus", ["free", "qwen"])]
    provider.queue(Scripted(json_body={**OK, "model": "qwen3-8b"}))
    response = shunt.post("/v1/messages", json=ASK)
    assert response.status_code == 200
    assert response.headers["x-shunt-model"] == "qwen3-8b"
    assert len(provider.calls) == 1, "o candidato sem credencial nao pode ser chamado"
    assert provider.calls[0].url.host == "fake.local", "a chamada foi para o provedor errado"


def test_the_error_names_the_provider_without_a_credential(shunt, settings):
    settings.providers["openrouter"].api_key = None
    settings.routes = [("opus", ["free"])]
    body = shunt.post("/v1/messages", json=ASK).json()
    assert "provider openrouter sem chave configurada" in body["error"]["message"]


def test_a_stream_also_skips_the_candidate_whose_credential_is_unset(
    shunt, provider, settings
):
    settings.providers["openrouter"].api_key = None
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
    assert "sem chave configurada" not in body


def test_a_transparent_candidate_is_called_even_with_no_server_credential(
    shunt, provider, settings
):
    """Transparente usa a chave do CLIENTE, entao a do servidor pode faltar.

    Sem esta distincao, a regra de credencial ausente mataria justamente o modo
    em que o proxy nao deve ter credencial nenhuma.
    """
    settings.providers["anthropic"].api_key = None
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


# ---------------------------------------------------------------------------
# Lote 2: motivos de pular um candidato, vistos pelo efeito na rede
# ---------------------------------------------------------------------------


def test_a_full_provider_is_skipped_and_the_next_candidate_serves(
    shunt, provider, settings, trace_lines
):
    """B7 -- `dispatcher.py:626-631`: slot ocupado e pulado na hora.

    E concorrencia de verdade: um pedido sobe ao provedor de UM slot por uma
    thread separada, e so a liberacao do primeiro devolve o lugar. O
    `pool` injetado em `app.state` e recriado aqui porque os gates nascem do
    snapshot de `settings` no construtor do `UpstreamPool`.

    O trace nao e exposto na rede, mas e o que o operador le: a linha
    `busy (1/1 in use)` vai no `attempts` do log da requisicao.
    """
    settings.providers["local"].max_concurrency = 1
    settings.routes = [("opus", ["slow", "free"]), ("slow", ["slow"])]
    settings.models["slow"] = ModelConfig(
        provider="local",
        model="qwen3-8b",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=32768,
        max_output_tokens=4096,
    )
    pool = UpstreamPool(settings, transport=ScriptedTransport(provider))
    shunt_app.state.settings = settings
    shunt_app.state.pool = pool
    with TestClient(shunt_app) as client:
        # O pedido que ocupa o slot e um STREAM lento: so `chunk_delay` (na
        # producao dos eventos, que o `StreamingResponse` entrega ao consumo)
        # mantem a thread segurando o slot o tempo necessario. Um bufferizado
        # terminaria antes do segundo chegar.
        provider.queue(
            Scripted(
                sse=sse_chunks({"choices": [{"delta": {"content": "devagar"}}]}),
                chunk_delay=0.6,
            ),
            Scripted(json_body=OK),
        )
        thread = Thread(
            target=client.post,
            # O pedido que ocupa o slot fala o dialeto do provedor (openai ->
            # openai): assim ele e puro passthrough, sem traducao, e o SSE
            # chega intacto. Um pedido Anthropic aqui caia em
            # `unreadable body` -- o stream OpenAI nao se traduz.
            kwargs={
                "url": "/v1/chat/completions",
                "json": {
                    "model": "slow",
                    "messages": [{"role": "user", "content": "segurando o slot"}],
                    "stream": True,
                },
            },
        )
        thread.start()
        # O primeiro pedido precisa estar com o slot na mao antes do segundo
        # tentar toma-lo: sem isso os dois chegam vazios e ninguem e pulado.
        time.sleep(0.15)
        response = client.post("/v1/messages", json=ASK)
        thread.join(5)
    assert response.status_code == 200, response.text
    assert response.headers["x-shunt-model"] == "vendor/free", (
        "o provedor cheio foi pulado e o segundo candidato atendeu"
    )
    assert len(provider.calls) == 2
    assert provider.calls[0].url.host == "fake.local", "o primeiro chegou ao provedor de 1 slot"
    assert provider.calls[1].url.host == "fake.openrouter", "o segundo caiu no free"
    assert any("busy (1/1 in use)" in line for line in trace_lines), trace_lines


def test_the_total_deadline_cuts_the_candidate_short(shunt, provider, settings, monkeypatch):
    """B8 -- `dispatcher.py:704-711`: o prazo total corta a cadeia.

    O relogio e substituido pelo mesmo `_Clock` dos testes unitarios
    (`tests/core/test_dispatcher.py:719`): sem ele, a unica forma de o prazo
    estourar e dormir de verdade. Quatro instantes: o da duracao do log
    (`dispatch`, `dispatcher.py:544`), o limite (0 + TOTAL_DEADLINE), o da
    tentativa 1 do `instavel` -- dentro do prazo -- e o da tentativa 2, ja
    fora. O 503 e RETRY (`attempt.py:42`), entao sem o corte o candidato
    seria repetido `MAX_ATTEMPTS` vezes antes de a cadeia seguir.
    """
    monkeypatch.setattr(dispatcher, "TOTAL_DEADLINE", 5.0)
    # Sequencia de leituras do relogio: a duracao do log e o limite (0+5), a
    # tentativa 1 do `instavel` em 1.0 (dentro do prazo), a checagem do backoff
    # em 10_000 (fora -- o retry e pulado) e a entrada do `free` em 1.0 (de
    # volta ao prazo). Sem esse ultimo valor baixo o relogio repetiria 10_000
    # e o `free` tambem receberia `not tried`.
    monkeypatch.setattr(dispatcher, "time", _Clock(0.0, 0.0, 1.0, 10_000.0, 1.0))
    settings.routes = [("opus", ["instavel", "free"])]
    settings.models["instavel"] = ModelConfig(
        provider="openrouter",
        model="vendor/instavel",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=64000,
        max_output_tokens=8192,
    )
    provider.queue(
        Scripted(status=503, json_body={"error": {"message": "fora do ar"}}),
        Scripted(json_body=OK),
    )
    response = shunt.post("/v1/messages", json=ASK)
    assert response.status_code == 200, response.text
    assert response.headers["x-shunt-model"] == "vendor/free", response.text
    assert len(provider.calls) == 2, "uma chamada no instavel, uma no free"
    assert provider.calls[0].url.host == "fake.openrouter"
    assert provider.calls[1].url.host == "fake.openrouter"


def test_the_deadline_names_the_candidate_that_was_never_tried(
    shunt, provider, settings, monkeypatch, trace_lines
):
    """B8, a outra face: o rastro separa quem cortou no meio de quem nem chegou.

    `dispatcher.py:708-710`: `not tried, deadline exceeded` numa tentativa que
    comecou DEPOIS do prazo. O relogio salta para 10_000 logo apos a primeira
    falha do `instavel`, e o `free` -- alcancado fora do prazo -- recebe a
    linha que diz que ele NEM foi tentado, nao que estourou prazo. Sao duas
    falhas diferentes e o operador precisa distinguir.
    """
    monkeypatch.setattr(dispatcher, "TOTAL_DEADLINE", 5.0)
    monkeypatch.setattr(dispatcher, "time", _Clock(0.0, 0.0, 1.0, 10_000.0))
    settings.routes = [("opus", ["instavel", "free"])]
    settings.models["instavel"] = ModelConfig(
        provider="openrouter",
        model="vendor/instavel",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=64000,
        max_output_tokens=8192,
    )
    provider.queue(
        Scripted(status=503, json_body={"error": {"message": "fora do ar"}}),
        Scripted(json_body=OK),
    )
    response = shunt.post("/v1/messages", json=ASK)
    # O prazo ja foi todo: a cadeia inteira e cortada e o erro sobe. A linha
    # que importa e a que distingue as duas falhas.
    assert response.status_code == 503, response.text
    assert any("vendor/free: not tried, deadline exceeded" in line for line in trace_lines), trace_lines
    assert any("retry skipped, backoff would pass the deadline" in line for line in trace_lines), (
        trace_lines
    )


def test_a_2xx_with_an_unreadable_body_falls_to_the_next_candidate(
    shunt, provider, settings
):
    """B9 -- `dispatcher.py:734-747`: 200 com corpo que nao e JSON.

    Um gateway interposto devolvendo uma pagina HTML com status 200. O
    `response.json()` do httpx levanta, o candidato e declarado 502 com a
    linha `unreadable body` e a cadeia segue para o proximo.
    """
    settings.routes = [("opus", ["surdo", "free"])]
    settings.models["surdo"] = ModelConfig(
        provider="openrouter",
        model="vendor/surdo",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=64000,
        max_output_tokens=8192,
    )
    provider.queue(
        Scripted(
            status=200,
            raw_body=b"<html>gateway em vez do provedor</html>",
            headers={"content-type": "text/html"},
        ),
        Scripted(json_body=OK),
    )
    response = shunt.post("/v1/messages", json=ASK)
    assert response.status_code == 200, response.text
    assert response.headers["x-shunt-model"] == "vendor/free"
    assert len(provider.calls) == 2, "o corpo ilegivel derrubou so este candidato"
    assert provider.calls[0].url.host == "fake.openrouter"


def test_the_unreadable_body_is_named_in_the_error_when_the_chain_is_exhausted(
    shunt, provider, settings, trace_lines
):
    """B9 com a cadeia inteira respondendo HTML: o 502 final carrega a linha."""
    settings.routes = [("opus", ["surdo", "outro"])]
    settings.models["surdo"] = ModelConfig(
        provider="openrouter",
        model="vendor/surdo",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=64000,
        max_output_tokens=8192,
    )
    settings.models["outro"] = ModelConfig(
        provider="local",
        model="qwen3-8b",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=32768,
        max_output_tokens=4096,
    )
    provider.queue(
        Scripted(
            status=200,
            raw_body=b"<html>gateway</html>",
            headers={"content-type": "text/html"},
        ),
        Scripted(
            status=200,
            raw_body=b"<html>outro gateway</html>",
            headers={"content-type": "text/html"},
        ),
        Scripted(
            status=200,
            raw_body=b"<html>o degrau de baixo tambem</html>",
            headers={"content-type": "text/html"},
        ),
    )
    body = shunt.post("/v1/messages", json=ASK).json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"
    assert "unreadable body" in body["error"]["message"]
    lines = trace_lines
    assert any("unreadable body (attempt 1)" in line for line in lines), lines


def test_a_chain_that_does_not_fit_falls_to_the_default_model_anyway(
    shunt, provider, settings
):
    """B10 -- `dispatcher.py:518-543`: a escada por context window.

    Tudo na cadeia cai por `SIZE_DROP` e sobra vazio: o degrau de baixo e o
    `default_model`, que atende MESMO NAO CABENDO -- e a escolha do operador
    para "quando nada serve". O rastro carrega a linha `taken anyway`.
    """
    settings.default_model = "largao"
    settings.routes = [("opus", ["curto"])]
    settings.models["curto"] = ModelConfig(
        provider="openrouter",
        model="vendor/curto",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=50,
        max_output_tokens=16,
    )
    settings.models["largao"] = ModelConfig(
        provider="local",
        model="qwen3-8b",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=100,
        max_output_tokens=16,
    )
    provider.queue(Scripted(json_body={**OK, "model": "qwen3-8b"}))
    big = {**ASK, "messages": [{"role": "user", "content": "x" * 8000}]}
    response = shunt.post("/v1/messages", json=big)
    assert response.status_code == 200, response.text
    assert response.headers["x-shunt-model"] == "qwen3-8b", (
        "o default_model atendeu mesmo sem caber"
    )
    assert len(provider.calls) == 1, "o modelo apertado foi descartado antes de chamar"
    assert provider.calls[0].url.host == "fake.local", "a chamada foi para o degrau de baixo"


def test_the_taken_anyway_line_is_named_after_the_default_model(shunt, settings, trace_lines):
    """B10, a linha da escada: ela se apresenta pelo MODELO, nunca pelo alias."""
    settings.default_model = "largao"
    settings.routes = [("opus", ["curto"])]
    settings.models["curto"] = ModelConfig(
        provider="openrouter",
        model="vendor/curto",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=50,
        max_output_tokens=16,
    )
    settings.models["largao"] = ModelConfig(
        provider="local",
        model="qwen3-8b",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=100,
        max_output_tokens=16,
    )
    big = {**ASK, "messages": [{"role": "user", "content": "x" * 8000}]}
    shunt.post("/v1/messages", json=big)
    lines = trace_lines
    assert any(
        "qwen3-8b: taken anyway, nothing in the chain fits" in line for line in lines
    ), lines


def test_a_provider_that_cannot_serve_the_endpoint_is_skipped(shunt, provider, settings):
    """B11 -- `dispatcher.py:589-591`: o par (protocolo, endpoint) nao existe.

    A Anthropic nao tem `/completions` nem `/embeddings` (`PATHS`,
    `dispatcher.py:71-82`): um candidato anthropic num pedido de completion e
    pulado com `endpoint not supported`, e o proximo atende.
    """
    settings.routes = [("text-davinci", ["antigo", "free"])]
    settings.models["antigo"] = ModelConfig(
        provider="anthropic",
        model="claude-antigo",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=200000,
        max_output_tokens=8192,
    )
    settings.models["free"] = settings.models["free"].model_copy(
        update={"provider": "local", "model": "qwen3-8b"}
    )
    provider.queue(Scripted(json_body=OPENAI_COMPLETION))
    response = shunt.post(
        "/v1/completions",
        json={"model": "text-davinci", "prompt": "diga oi", "max_tokens": 16},
    )
    assert response.status_code == 200, response.text
    assert response.json()["object"] == "text_completion"
    assert len(provider.calls) == 1, "o candidato sem path foi pulado"
    assert provider.calls[0].url.host == "fake.local", "a chamada foi para o free"
    assert provider.calls[0].url.path == "/v1/completions"


def test_the_unsupported_endpoint_is_named_when_nothing_serves_it(
    shunt, provider, settings, trace_lines
):
    """B11 com a cadeia sem path: o erro nomeia o motivo de cada pulo.

    O status e 502 e nao 400 de proposito: o pulo por `PATHS` nao conta como
    skip de elegibilidade (`_transparent_skip_reason`, T4/T5) -- so ele nao
    dispara a guarda do 400 (`dispatcher.py:646-650`). O que importa e que o
    motivo chega ao operador e que ninguem foi chamado.
    """
    settings.routes = [("text-davinci", ["antigo"])]
    settings.models["antigo"] = ModelConfig(
        provider="anthropic",
        model="claude-antigo",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=200000,
        max_output_tokens=8192,
    )
    body = shunt.post(
        "/v1/completions",
        json={"model": "text-davinci", "prompt": "diga oi", "max_tokens": 16},
    ).json()
    assert body["error"]["type"] == "server_error"
    assert "claude-antigo: endpoint not supported" in body["error"]["message"]
    assert provider.calls == [], "ninguem foi chamado"
    assert any("claude-antigo: endpoint not supported" in line for line in trace_lines), trace_lines


def test_a_stream_whose_only_candidate_cannot_stream_answers_400_saying_why(
    shunt, provider, settings
):
    """B12 -- `dispatcher.py:1082-1084`: a cadeia vazia do stream tambem e 400.

    O espelho de stream do `test_no_capable_candidate_answers_400_saying_why_each_was_dropped`
    (`test_fallback_e2e.py:372`). A unica forma da cadeia ficar VAZIA e o
    degrau de baixo nao existir: o nome pedido nao casa nenhuma rota e nao
    deixa deduzir o provedor (`resolver.py:59-67`, `UnknownProviderError`), e
    o unico candidato da rota cai por capability. Sem essa guarda o stream
    devolveria o 502 generico de "nada respondeu" -- mentira, nada foi
    chamado.
    """
    settings.routes = [("acme", ["surdo"])]
    settings.models["surdo"] = ModelConfig(
        provider="openrouter",
        model="vendor/surdo",
        supports=ModelCaps(tools=True, streaming=False, vision=False),
        context_window=64000,
        max_output_tokens=8192,
    )
    with shunt.stream(
        "POST", "/v1/messages", json={**ASK, "model": "acme-opus", "stream": True}
    ) as response:
        assert response.status_code == 200  # o SSE de erro vem como 200 + event: error
        body = "".join(response.iter_text())
    assert provider.calls == [], "candidato incapaz nao pode ser chamado"
    assert "event: error" in body
    assert "no streaming support" in body
    assert "vendor/surdo" in body


def test_a_stream_whose_only_candidate_lacks_tools_answers_400_saying_why(
    shunt, provider, settings
):
    """B12 com outro motivo de descarte: `tools=False` num pedido com tools.

    O `mudo` ja existe no catalogo (`conftest.py:64`) sem tools -- so a rota
    muda, apontando para ele como unico candidato de um nome sem degrau.
    """
    settings.routes = [("acme", ["mudo"])]
    with shunt.stream(
        "POST",
        "/v1/messages",
        json={
            **ASK,
            "model": "acme-opus",
            "stream": True,
            "tools": [{"name": "read", "input_schema": {"type": "object"}}],
        },
    ) as response:
        body = "".join(response.iter_text())
    assert provider.calls == [], "candidato incapaz nao pode ser chamado"
    assert "event: error" in body
    assert "no tool support" in body
    assert "vendor/mudo" in body


def test_a_candidate_that_cannot_be_translated_is_skipped(shunt, provider, settings):
    """B13 -- `dispatcher.py:616-622`: a traducao do pedido falhou.

    Mais de 4 `stop_sequences` faz `anthropic_request_to_openai` levantar
    (`to_openai.py:229-233`), e isso e defeito DESTE candidato -- o proximo,
    que fala o protocolo do cliente, leva o corpo sem traducao nenhuma.
    """
    settings.routes = [("opus", ["traduz", "antropico"])]
    settings.models["traduz"] = ModelConfig(
        provider="openrouter",
        model="vendor/traduz",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=200000,
        max_output_tokens=8192,
    )
    settings.models["antropico"] = ModelConfig(
        provider="anthropic",
        model="claude-direto",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=200000,
        max_output_tokens=8192,
    )
    provider.queue(
        Scripted(
            json_body={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-direto",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "sem traducao"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
    )
    response = shunt.post(
        "/v1/messages",
        json={**ASK, "stop_sequences": ["a", "b", "c", "d", "e"]},
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-shunt-model"] == "claude-direto", (
        "o candidato intraduzivel foi pulado e o proximo atendeu"
    )
    assert len(provider.calls) == 1, "so o candidato que fala o dialeto do cliente foi chamado"
    assert provider.calls[0].url.host == "fake.anthropic"
    assert response.json()["content"][0]["text"] == "sem traducao"


def test_the_failed_translation_is_named_in_the_error(shunt, provider, settings, trace_lines):
    """B13, a linha da traducao: ela nomeia o candidato e o motivo."""
    settings.routes = [("opus", ["traduz", "antropico"])]
    settings.models["traduz"] = ModelConfig(
        provider="openrouter",
        model="vendor/traduz",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=200000,
        max_output_tokens=8192,
    )
    settings.models["antropico"] = ModelConfig(
        provider="anthropic",
        model="claude-direto",
        supports=ModelCaps(tools=True, streaming=True, vision=False),
        context_window=200000,
        max_output_tokens=8192,
    )
    provider.queue(Scripted(status=500, json_body={"error": {"message": "caiu"}}))
    shunt.post("/v1/messages", json={**ASK, "stop_sequences": ["a", "b", "c", "d", "e"]})
    lines = trace_lines
    assert any(
        "vendor/traduz: request translation failed" in line for line in lines
    ), lines
