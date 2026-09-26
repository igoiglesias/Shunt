"""Uma linha estruturada por requisicao, e nenhuma credencial dentro dela.

O log e a unica janela para o que o proxy decidiu: qual regra casou, qual
candidato respondeu, quantas tentativas custou. Sem ele, diagnosticar um
fallback silencioso exige reproduzir a requisicao.
"""

import json
import logging
from datetime import UTC, datetime, timedelta

import httpx
import respx

from app.core import observability
from app.core.dispatcher import ShuntRequest, dispatch, dispatch_stream
from app.core.observability import RequestLog, configure_logging, log_request, redact
from app.core.upstream import UpstreamPool
from tests.core.test_dispatcher import SETTINGS

BODY = {"model": "claude-opus-4-5", "max_tokens": 16, "messages": [{"role": "user", "content": "oi"}]}
SSE_HEADERS = {"content-type": "text/event-stream"}


def lines(caplog):
    return [json.loads(r.message) for r in caplog.records if r.name == "shunt"]


# --------------------------------------------------------------------------
# redact
# --------------------------------------------------------------------------


def test_credentials_are_redacted():
    out = redact(
        {
            "x-api-key": "sk-ant-segredo",
            "authorization": "Bearer sk-segredo",
            "content-type": "application/json",
        }
    )
    assert out["x-api-key"] == "[redacted]"
    assert out["authorization"] == "[redacted]"
    assert out["content-type"] == "application/json"


def test_every_credential_header_is_covered_whatever_its_casing():
    """O cabecalho chega na caixa que o cliente digitou. Comparar sem baixar a
    caixa deixa `X-Api-Key` passar inteiro para o log."""
    out = redact(
        {
            "Authorization": "Bearer a",
            "X-Api-Key": "b",
            "Api-Key": "c",
            "Proxy-Authorization": "d",
            "X-Shunt-Token": "e",
        }
    )
    assert set(out.values()) == {"[redacted]"}


def test_redaction_keeps_the_header_name_so_the_line_still_says_what_arrived():
    out = redact({"x-api-key": "sk-segredo"})
    assert "x-api-key" in out
    assert "sk-segredo" not in json.dumps(out)


# --------------------------------------------------------------------------
# log_request
# --------------------------------------------------------------------------


def test_log_line_carries_the_rule_that_matched(caplog):
    entry = RequestLog(
        request_id="r1",
        requested_model="claude-opus-4-5",
        rule="family",
        matched="opus",
        candidate="qwen",
        attempts=["qwen: 400", "free: 200"],
        input_tokens=10,
        output_tokens=3,
        ttft_ms=120,
        duration_ms=900,
        translated=True,
    )
    with caplog.at_level(logging.INFO, logger="shunt"):
        log_request(entry)
    payload = json.loads(caplog.records[-1].message)
    assert payload["rule"] == "family"
    assert payload["matched"] == "opus"
    assert payload["attempts"] == ["qwen: 400", "free: 200"]


def test_the_line_is_one_json_object_on_one_line(caplog):
    """Uma linha por requisicao e o contrato: `grep` e `jq` leem assim, e um
    JSON indentado quebraria os dois."""
    with caplog.at_level(logging.INFO, logger="shunt"):
        log_request(RequestLog(request_id="r", requested_model="m", rule="exact", matched=None,
                               candidate=None))
    message = caplog.records[-1].message
    assert "\n" not in message
    assert json.loads(message)["request_id"] == "r"


def test_non_ascii_is_not_escaped(caplog):
    with caplog.at_level(logging.INFO, logger="shunt"):
        log_request(RequestLog(request_id="r", requested_model="modelo-são-paulo", rule="exact",
                               matched=None, candidate=None))
    assert "são" in caplog.records[-1].message


# --------------------------------------------------------------------------
# Ligado no dispatcher
# --------------------------------------------------------------------------


@respx.mock
async def test_a_served_request_logs_the_candidate_and_the_tokens(caplog):
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4},
            },
        )
    )
    pool = UpstreamPool(SETTINGS)
    try:
        with caplog.at_level(logging.INFO, logger="shunt"):
            await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    entry = lines(caplog)[-1]
    assert entry["requested_model"] == "claude-opus-4-5"
    assert entry["rule"] == "family"
    assert entry["candidate"] is not None
    assert entry["input_tokens"] == 11
    assert entry["output_tokens"] == 4
    assert entry["translated"] is True
    assert entry["duration_ms"] >= 0


@respx.mock
async def test_a_failed_chain_still_logs_with_no_candidate(caplog):
    """A requisicao que ninguem serviu e justamente a que se quer no log."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(402, json={"error": {"message": "sem credito"}})
    )
    pool = UpstreamPool(SETTINGS)
    try:
        with caplog.at_level(logging.INFO, logger="shunt"):
            await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    entry = lines(caplog)[-1]
    assert entry["candidate"] is None
    assert entry["attempts"]


async def test_a_request_no_candidate_can_serve_logs_too(caplog):
    """O 400 de cadeia vazia sai antes de qualquer HTTP, e e um caminho de
    saida proprio: sem log aqui, a falha mais facil de diagnosticar e a unica
    invisivel. Aqui a entrada e maior que o `context_window` dos dois
    candidatos, entao o filtro de capacidade esvazia a cadeia."""
    body = {**BODY, "messages": [{"role": "user", "content": "x" * 400_000}]}
    pool = UpstreamPool(SETTINGS)
    try:
        with caplog.at_level(logging.INFO, logger="shunt"):
            await dispatch(ShuntRequest("anthropic", body, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert lines(caplog)[-1]["candidate"] is None


@respx.mock
async def test_each_request_gets_its_own_id(caplog):
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    pool = UpstreamPool(SETTINGS)
    try:
        with caplog.at_level(logging.INFO, logger="shunt"):
            await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
            await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    first, second = lines(caplog)[-2:]
    assert first["request_id"] != second["request_id"]


@respx.mock
async def test_a_stream_logs_the_time_to_the_first_event(caplog):
    """`ttft_ms` so existe no caminho de streaming, e e a metrica que diz se um
    modelo local esta lento antes de o usuario desistir."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            text='data: {"choices": [{"delta": {"content": "oi"}}]}\n\ndata: [DONE]\n\n',
        )
    )
    pool = UpstreamPool(SETTINGS)
    try:
        with caplog.at_level(logging.INFO, logger="shunt"):
            async for _ in dispatch_stream(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool):
                pass
    finally:
        await pool.aclose()
    entry = lines(caplog)[-1]
    assert entry["ttft_ms"] is not None
    assert entry["ttft_ms"] >= 0
    assert entry["candidate"] is not None


@respx.mock
async def test_a_buffered_request_has_no_ttft(caplog):
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    pool = UpstreamPool(SETTINGS)
    try:
        with caplog.at_level(logging.INFO, logger="shunt"):
            await dispatch(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    assert lines(caplog)[-1]["ttft_ms"] is None


@respx.mock
async def test_the_client_credential_never_reaches_the_log(caplog):
    """O teste que importa: o corpo e os cabecalhos passam pelo dispatcher, e
    nenhum caminho pode deixar a chave do cliente cair na linha."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    headers = {"x-api-key": "sk-do-cliente-secreta", "authorization": "Bearer tambem-secreto"}
    pool = UpstreamPool(SETTINGS)
    try:
        with caplog.at_level(logging.INFO, logger="shunt"):
            await dispatch(ShuntRequest("anthropic", BODY, headers), SETTINGS, pool)
    finally:
        await pool.aclose()
    dumped = json.dumps(lines(caplog)[-1])
    assert "sk-do-cliente-secreta" not in dumped
    assert "tambem-secreto" not in dumped


@respx.mock
async def test_tokens_are_read_in_whichever_dialect_the_answer_came_back_in(caplog):
    """O corpo que chega ao log e o corpo JA TRADUZIDO, e ele fala o dialeto de
    quem perguntou: `input_tokens` para um cliente Anthropic,
    `prompt_tokens` para um cliente OpenAI. Ler so uma das duas grafias zera a
    contagem de metade das requisicoes."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 13, "completion_tokens": 6},
            },
        )
    )
    openai_body = {"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "oi"}]}
    pool = UpstreamPool(SETTINGS)
    try:
        with caplog.at_level(logging.INFO, logger="shunt"):
            await dispatch(ShuntRequest("openai", openai_body, {}), SETTINGS, pool)
    finally:
        await pool.aclose()
    entry = lines(caplog)[-1]
    assert entry["input_tokens"] == 13
    assert entry["output_tokens"] == 6


# --------------------------------------------------------------------------
# O logger precisa de um handler para existir fora do pytest
# --------------------------------------------------------------------------


def test_configure_logging_gives_the_logger_somewhere_to_write():
    """MEDIDO antes desta funcao existir: sob `uvicorn app.main:app`, nenhuma
    linha aparecia. O `caplog` do pytest anexa um handler por conta propria, e
    por isso os testes passavam enquanto o recurso estava morto no servidor.
    """
    logger = logging.getLogger("shunt")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    configure_logging()

    assert logger.handlers
    assert logger.level == logging.INFO


def test_configuring_twice_does_not_double_every_line():
    """O `lifespan` roda uma vez por processo, mas um `--reload` reimporta o
    modulo. Dois handlers no mesmo logger escrevem a mesma linha duas vezes, e
    um contador que leia esse log passa a contar o dobro."""
    logger = logging.getLogger("shunt")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    configure_logging()
    configure_logging()

    assert len(logger.handlers) == 1


def test_the_level_comes_from_the_environment(monkeypatch):
    """Um proxy local numa sessao de depuracao quer DEBUG; um rodando o dia
    inteiro quer WARNING. Sem a variavel, INFO, que e onde a linha sai."""
    monkeypatch.setenv("SHUNT_LOG_LEVEL", "WARNING")
    logger = logging.getLogger("shunt")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    configure_logging()

    assert logger.level == logging.WARNING


def test_an_unreadable_level_falls_back_to_info_instead_of_crashing(monkeypatch):
    """Erro de digitacao numa variavel de ambiente nao pode derrubar o boot do
    proxy: o log e observabilidade, nao o servico."""
    monkeypatch.setenv("SHUNT_LOG_LEVEL", "VERBOSO")
    logger = logging.getLogger("shunt")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    configure_logging()

    assert logger.level == logging.INFO


@respx.mock
async def test_a_stream_logs_the_tokens_it_consumed(caplog):
    """MEDIDO contra o llama.cpp: a linha de uma requisicao em streaming saia
    com `input_tokens: 0` e `output_tokens: 0`, e streaming e quase todo o
    trafego de um harness. O `usage` chega num chunk de `choices` vazio, que
    so o tradutor ve."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            text=(
                'data: {"choices": [{"delta": {"content": "oi"}}]}\n\n'
                'data: {"choices": [], "usage": {"prompt_tokens": 21, "completion_tokens": 8}}\n\n'
                "data: [DONE]\n\n"
            ),
        )
    )
    pool = UpstreamPool(SETTINGS)
    try:
        with caplog.at_level(logging.INFO, logger="shunt"):
            async for _ in dispatch_stream(ShuntRequest("anthropic", BODY, {}), SETTINGS, pool):
                pass
    finally:
        await pool.aclose()
    entry = lines(caplog)[-1]
    assert entry["input_tokens"] == 21
    assert entry["output_tokens"] == 8


def test_the_line_does_not_also_reach_the_root_logger():
    """A linha e do proxy, nao do uvicorn. Deixa-la propagar entrega a mesma
    requisicao tambem ao handler raiz, e o operador le cada uma duas vezes."""
    logger = logging.getLogger("shunt")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    configure_logging()

    seen = []

    class Counter(logging.Handler):
        def emit(self, record):
            seen.append(record)

    root = logging.getLogger()
    counter = Counter()
    root.addHandler(counter)
    try:
        log_request(RequestLog(request_id="r", requested_model="m", rule="exact", matched=None,
                               candidate=None))
    finally:
        root.removeHandler(counter)

    assert seen == []


# --------------------------------------------------------------------------
# as_event: started_at e o INICIO da requisicao
# --------------------------------------------------------------------------

_FROZEN_END = datetime(2026, 9, 23, 12, 0, 0, 500000, tzinfo=UTC)


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        # Honra o `tz`: um `now()` sem fuso devolve instante ingenuo, como o real.
        return _FROZEN_END.astimezone(tz) if tz else _FROZEN_END.replace(tzinfo=None)


def test_started_at_is_the_end_minus_the_duration(monkeypatch):
    """`as_event` roda quando a requisicao ja TERMINOU.

    Gravar `now()` cru poria o fim na coluna que todo consumidor le como
    inicio: a janela `since`, a fita ao vivo e o dossie. O inicio e o fim
    menos a duracao medida.
    """
    monkeypatch.setattr(observability, "datetime", _FrozenDatetime)
    event = observability.as_event(
        RequestLog(request_id="r", requested_model="m", rule="exact", matched=None,
                   candidate=None, duration_ms=1500)
    )
    assert event["started_at"] == _FROZEN_END - timedelta(milliseconds=1500)
    assert event["started_at"] == datetime(2026, 9, 23, 11, 59, 59, tzinfo=UTC)


def test_a_request_without_measured_duration_started_when_it_ended(monkeypatch):
    """Duracao padrao 0 -- a listagem de modelos -- nao desloca o instante."""
    monkeypatch.setattr(observability, "datetime", _FrozenDatetime)
    event = observability.as_event(
        RequestLog(request_id="r", requested_model="m", rule="exact", matched=None,
                   candidate=None)
    )
    assert event["started_at"] == _FROZEN_END
    assert event["started_at"].tzinfo is UTC
