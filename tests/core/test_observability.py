"""Uma linha estruturada por requisicao, e nenhuma credencial dentro dela.

O log e a unica janela para o que o proxy decidiu: qual regra casou, qual
candidato respondeu, quantas tentativas custou. Sem ele, diagnosticar um
fallback silencioso exige reproduzir a requisicao.
"""

import json
import logging
from datetime import UTC, datetime, timedelta

import httpx
import pytest
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
    # O `last_resort` de `828591e` tenta o transparente no host oficial quando
    # a rota esgota; sem mock dele a chamada saida para a internet.
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(402, json={"error": {"message": "sem credito"}})
    )
    respx.post("https://api.anthropic.com/v1/messages").mock(
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


# --------------------------------------------------------------------------
# as_event: effort vai para o evento, nunca some no caminho
# --------------------------------------------------------------------------


def test_as_event_copies_the_effort_from_the_entry():
    """O dict do evento copia `entry.effort` (observability.py:257).

    Antes da correcao a chave nao existia: o dispatcher media o effort
    aplicado, mas a coluna do banco ficava sempre NULL mesmo com o effort
    tendo ido para o upstream. Sem esta linha o teste falha com KeyError."""
    entry = RequestLog(
        request_id="r", requested_model="m", rule="exact", matched=None,
        candidate=None, effort="high",
    )
    assert observability.as_event(entry)["effort"] == "high"


def test_as_event_keeps_a_missing_effort_as_none():
    """Sem effort aplicado (`entry.effort` None) o evento grava NULL, nunca uma
    chave ausente: a coluna nova precisa do valor para separar silencio de
    sobrescrita no painel."""
    entry = RequestLog(
        request_id="r", requested_model="m", rule="exact", matched=None,
        candidate=None,
    )
    event = observability.as_event(entry)
    assert "effort" in event
    assert event["effort"] is None


# --------------------------------------------------------------------------
# kind: o painel precisa separar requisicao de modelo do trafego repassado
# --------------------------------------------------------------------------


class _SpyRecorder:
    def __init__(self):
        self.stored = []
        self.records = []

    def record(self, event):
        self.records.append(event)

    def store(self, event):
        self.stored.append(event)


def test_log_request_stores_the_event_with_kind_model():
    """Toda requisicao de modelo vai para o banco com `kind` preenchido.

    NULL tambem significaria modelo, mas as linhas antigas so existem porque a
    coluna e nova: daqui em diante a escrita deixa claro o que gravou.
    """
    spy = _SpyRecorder()
    original = observability.recorder()
    observability.set_recorder(spy)
    try:
        log_request(RequestLog(request_id="r1", requested_model="m", rule="exact",
                               matched=None, candidate=None))
    finally:
        observability.set_recorder(original)
    assert len(spy.records) == 1
    assert spy.records[0]["kind"] == "model"
    assert spy.stored == []


def test_a_request_that_measured_nothing_stores_null_not_zero():
    """Quem nao passou tokens grava NULL (silencio), nunca 0 (medicao).

    O default da coluna e a semantica da Task 2: os produtores passam None
    quando o provedor nao disse nada, e uma rota que esquecer de setar cai no
    DEFAULT do dataclass. Se ele for 0, a linha vira "provedor mediu zero" e
    entra na taxa de geracao -- o oposto do que o painel quer dizer. O 0 so
    existe de verdade na linha de relay, que escreve explicito.
    """
    spy = _SpyRecorder()
    original = observability.recorder()
    observability.set_recorder(spy)
    try:
        log_request(RequestLog(request_id="sem-medida", requested_model="m", rule="exact",
                               matched=None, candidate=None, duration_ms=250))
    finally:
        observability.set_recorder(original)

    event = spy.records[0]
    assert event["input_tokens"] is None
    assert event["output_tokens"] is None


def test_log_relay_logs_a_line_and_stores_without_the_panel_bus(caplog):
    """`log_relay`: linha JSON sem credencial, e `store` em vez de `record`.

    O relay nao entra no barramento do painel (SSE/`recent`): o trafego de
    passagem nao e uma requisicao de modelo, e misturar os dois faria cada
    repasse aparecer na tela como se o proxy tivesse chamado um modelo.
    """
    spy = _SpyRecorder()
    original = observability.recorder()
    observability.set_recorder(spy)
    with caplog.at_level(logging.INFO, logger="shunt"):
        try:
            observability.log_relay(observability.RelayLog(
                path="/t/secreto-do-token/api/oauth/usage",
                target="anthropic",
                status=200,
                error_type=None,
                duration_ms=42,
            ))
        finally:
            observability.set_recorder(original)

    line = json.dumps(lines(caplog)[-1])
    assert "secreto-do-token" not in line, "o valor do token nao pode sair no log"
    assert "token=" not in line, "a query crua nao pode sair no log"

    assert len(spy.stored) == 1, "o relay grava pelo `store`"
    assert spy.records == [], "o relay nao publica no barramento do painel"
    event = spy.stored[0]
    assert event["kind"] == "relay"
    assert event["route"] == "/t/***/api/oauth/usage"
    assert event["dialect"] == "anthropic"
    assert event["status"] == 200
    assert event["rule"] == "none"
    assert event["requested_model"] == ""
    assert event["input_tokens"] == 0
    assert event["candidate_model"] is None


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (504, "upstream_timeout"),
        (502, "upstream_unreachable"),
    ],
)
def test_log_relay_infers_the_error_type_from_the_status(caplog, status, expected):
    """`log_relay` com `error_type=None`: 504 vira `upstream_timeout`, 502
    vira `upstream_unreachable` -- a linha do log E o evento gravado usam o
    nome inferido. Sem essa inferencia, um painel filtrando por `error_type`
    nunca veria o repasse que o upstream nao respondeu."""
    spy = _SpyRecorder()
    original = observability.recorder()
    with caplog.at_level(logging.INFO, logger="shunt"):
        try:
            observability.set_recorder(spy)
            observability.log_relay(observability.RelayLog(
                path="/t/secreto/api/v1/messages",
                target="anthropic",
                status=status,
                error_type=None,
                duration_ms=5,
            ))
        finally:
            observability.set_recorder(original)

    linha = lines(caplog)[-1]
    assert linha["status"] == status
    assert linha["error_type"] == expected
    event = spy.stored[0]
    assert event["error_type"] == expected


def test_log_relay_keeps_the_error_type_the_caller_already_named(caplog):
    """So 504/502 inferem: um `error_type` ja nomeado pelo chamador e a unica
    verdade -- inferir por cima apagaria o motivo real. E um outro status com
    `error_type=None` nao inventa nada: fica nulo."""
    spy = _SpyRecorder()
    original = observability.recorder()
    with caplog.at_level(logging.INFO, logger="shunt"):
        try:
            observability.set_recorder(spy)
            observability.log_relay(observability.RelayLog(
                path="/t/secreto/api/v1/messages",
                target="openai",
                status=502,
                error_type="dns_failure",
                duration_ms=5,
            ))
            observability.log_relay(observability.RelayLog(
                path="/t/secreto/api/v1/messages",
                target="openai",
                status=500,
                error_type=None,
                duration_ms=5,
            ))
        finally:
            observability.set_recorder(original)

    linhas = [json.loads(r.message) for r in caplog.records if r.name == "shunt"]
    assert linhas[-2]["error_type"] == "dns_failure"
    assert linhas[-1]["error_type"] is None
    assert spy.stored[0]["error_type"] == "dns_failure"
    assert spy.stored[1]["error_type"] is None


# --------------------------------------------------------------------------
# M2: a expressao do dispatcher distingue silencio de medicao
# --------------------------------------------------------------------------


@respx.mock
async def test_a_silent_usage_logs_null_and_never_zero(caplog):
    """Provedor que nao enviou usage gera input_tokens/output_tokens NULOS.

    O `dispatch` monta esses campos com `usage.get(...) if ... is not None else
    usage.get(...)`. A tentacao anterior era `usage.get("input_tokens") or
    usage.get("prompt_tokens") or 0`: com ela, um provedor em silencio total
    (nenhuma das duas chaves) caia no `0` final, e a linha passava a dizer que
    o provedor mediu zero tokens -- o que a entrada da taxa de geracao e
    denuncia como "o modelo e infinitamente lento". Nulo e silencio.
    """
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-sem-usage",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )
    )
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-sem-usage",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
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
    assert entry["candidate"] is not None
    assert entry["input_tokens"] is None
    assert entry["output_tokens"] is None


@respx.mock
async def test_um_usage_em_chaves_openai_sem_traducao_vai_para_a_linha(caplog):
    """O ramo `else` da expressao do `dispatch` (dispatcher.py) nunca foi lido.

    Com o mesmo protocolo nos dois lados nao ha traducao de resposta: o corpo
    que chega ao log fala `prompt_tokens`/`completion_tokens`, e so o `else`
    (`.get("prompt_tokens")` / `.get("completion_tokens")`) faz esses numeros
    aparecerem na linha. O teste anterior cobria o ramo direto porque pedia um
    provedor OpenAI a um caller Anthropic -- e o corpo chega TRADUZIDO de
    volta para `input_tokens`/`output_tokens`.
    """
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-cru",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 31, "completion_tokens": 17},
            },
        )
    )
    openai_body = {"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "oi"}]}
    pool = UpstreamPool(SETTINGS)
    try:
        with caplog.at_level(logging.INFO, logger="shunt"):
            await dispatch(
                ShuntRequest("openai", openai_body, {}, endpoint="chat"), SETTINGS, pool
            )
    finally:
        await pool.aclose()

    entry = lines(caplog)[-1]
    assert entry["candidate"] is not None
    assert entry["input_tokens"] == 31
    assert entry["output_tokens"] == 17
