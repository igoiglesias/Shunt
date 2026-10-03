"""Uma linha estruturada por requisicao, e nenhuma credencial dentro dela.

Este proxy decide coisas que nao aparecem na resposta: qual regra de rota
casou, quantos candidatos foram descartados antes de um responder, quantas
tentativas cada um custou. Sem essa linha, diagnosticar um fallback silencioso
exige reproduzir a requisicao, e uma requisicao com modelo local nem sempre se
reproduz igual.

Uma linha, um objeto JSON, sem indentacao: e assim que `grep` e `jq` leem, e um
JSON quebrado em varias linhas quebraria os dois.

O que NAO entra: corpo da requisicao, corpo da resposta e valor de qualquer
cabecalho de credencial. O log de um proxy e o lugar mais facil de vazar a
chave do cliente, porque ele ve as duas pontas.
"""

import json
import logging
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta

from app.core.official_hosts import Protocol
from app.core.token_auth import mask_path
from app.stats.recorder import Recorder

logger = logging.getLogger("shunt")

# Comparados sempre em caixa baixa: o cabecalho chega na caixa que o cliente
# digitou, e `X-Api-Key` passaria inteiro por uma comparacao literal.
SECRET_HEADERS = frozenset({"authorization", "x-api-key", "api-key", "proxy-authorization", "x-shunt-token"})

REDACTED = "[redacted]"


@dataclass
class RequestLog:
    """O que se quer saber de uma requisicao depois que ela terminou.

    `matched` e o padrao de rota que casou (None quando nenhuma casou);
    `candidate` e o modelo que de fato respondeu (None quando nenhum
    respondeu); `ttft_ms` so existe em streaming, onde ha um primeiro evento
    a cronometrar.
    """

    request_id: str
    requested_model: str
    rule: str
    matched: str | None
    candidate: str | None
    attempts: list[str] = field(default_factory=list)
    # O default e SILENCIO (None), e nunca 0. Zero seria uma MEDICAO ("provedor
    # disse que gastou 0"), e `queries._usage_reported` le o NULL para separar
    # as duas: uma rota que esquecer de setar cairia no default e gravaria uma
    # medicao que nao aconteceu, entrando na taxa de geracao. Quem tem 0 de
    # verdade e a linha de relay, que escreve explicito (`log_relay` abaixo);
    # os produtores passam None pelo mesmo motivo. Decisao: nenhum caller
    # depende do default 0 -- os produtores reais (dispatcher.py:453, 1446 e
    # v1.py:372) sempre passaram o valor ou None, e os testes que criam
    # RequestLog sem tokens nao somam tokens.
    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_ms: int | None = None
    duration_ms: int = 0
    translated: bool = False
    # O que o painel precisa e a linha de log nao precisava. Tudo com valor
    # padrao: uma rota que nao sabe responder um destes campos -- a listagem de
    # modelos nao tem candidato, uma recusa nao tem provedor -- omite e segue.
    route: str = ""
    dialect: str = ""
    stream: bool = False
    status: int = 0
    error_type: str | None = None
    provider: str | None = None
    tools_offered: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    thinking_blocks: int = 0
    # De qual projeto veio, e em que sessao do harness. Extraidos do corpo na
    # ingestao, porque o corpo GUARDADO e cortado e nem sempre existe.
    project: str | None = None
    session_id: str | None = None
    # O cache que o provedor reportou, quando reportou.
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    # O texto da conversa, quando `SHUNT_STORE_BODIES` esta ligado. Fica FORA
    # do que vai para o log: uma conversa inteira no stdout do proxy seria
    # outra coisa, e o log e lido por quem so quer a linha estruturada.
    body: dict | None = None


@dataclass
class RelayLog:
    """O que se quer saber de um repasse de rota desconhecida.

    `path` e o path que o CLIENTE pediu (o prefixo `/t/<token>/` ja reescrito
    pela middleware, quando veio por la); `target` e o protocolo do host
    oficial que recebeu a requisicao. `error_type` e o que o chamador ja sabe:
    as unicas falhas que o repasse responde ANTES de o status do upstream
    existir sao timeout (504) e upstream fora de alcance (502), e quem as
    causa as nomeia na hora. Ausente, `log_relay` deriva do status o mesmo
    nome -- 504 vira `upstream_timeout`, 502 vira `upstream_unreachable`.
    """

    path: str
    target: Protocol
    status: int
    error_type: str | None = None
    duration_ms: int = 0


def redact(headers: dict[str, str]) -> dict[str, str]:
    """Os mesmos cabecalhos, com o VALOR das credenciais trocado.

    O nome fica: saber que um `x-api-key` chegou e metade do diagnostico de
    uma autenticacao recusada, e o nome sozinho nao autentica ninguem.
    """
    return {k: (REDACTED if k.lower() in SECRET_HEADERS else v) for k, v in headers.items()}


_recorder = Recorder(None)


def set_recorder(recorder: Recorder) -> None:
    """Liga o armazem. Chamado pelo `lifespan`, uma vez.

    O padrao e um gravador sem engine, que e um no-op: quem importa este modulo
    fora de um processo com banco -- um teste de unidade, um script -- nao
    precisa saber que existe persistencia.
    """
    global _recorder
    _recorder = recorder


def recorder() -> Recorder:
    return _recorder


def log_request(entry: RequestLog) -> None:
    """A linha de log, e o mesmo evento empilhado para o painel.

    A ordem importa: o log sai primeiro. `record()` nao levanta e nao espera,
    mas se um dia levantar, a linha de log ja foi.
    """
    printable = {k: v for k, v in asdict(entry).items() if k != "body"}
    logger.info(json.dumps(printable, ensure_ascii=False))
    _recorder.record(as_event(entry))


def log_relay(entry: RelayLog) -> None:
    """A linha de log do repasse, e o evento gravado pelo `store`.

    `store` e nao `record`: o trafego repassado nao entra no barramento do
    painel (SSE/`recent`), que mostra requisicoes de modelo. A linha de log e
    a unica janela para o que o repasse decidiu, e nela NAO sobe a query nem
    os headers: um dos lugares onde `?token=` podia ter sido deixado em claro
    era justamente a query, e `mask_path` so esconde o prefixo `/t/<token>/`.
    """
    status = entry.status
    error_type = entry.error_type
    if error_type is None:
        if status == 504:
            error_type = "upstream_timeout"
        elif status == 502:
            error_type = "upstream_unreachable"
    route = mask_path(entry.path)
    logger.info(
        json.dumps(
            {
                "kind": "relay",
                "route": route,
                "dialect": entry.target,
                "status": status,
                "error_type": error_type,
                "duration_ms": entry.duration_ms,
            },
            ensure_ascii=False,
        )
    )
    _recorder.store(
        {
            "kind": "relay",
            "request_id": uuid.uuid4().hex,
            "started_at": datetime.now(UTC) - timedelta(milliseconds=entry.duration_ms),
            "route": route,
            "dialect": entry.target,
            "stream": False,
            "requested_model": "",
            "rule": "none",
            "matched": None,
            "provider": None,
            "candidate_model": None,
            "status": status,
            "error_type": error_type,
            # NOT NULL no banco: o repasse nao tem uso de tokens.
            "input_tokens": 0,
            "output_tokens": 0,
            "ttft_ms": None,
            "duration_ms": entry.duration_ms,
            "attempts": [],
            "fell_back": False,
            "tools_offered": [],
            "tools_called": [],
            "thinking_blocks": 0,
            "project": None,
            "session_id": None,
            "cached_input_tokens": None,
            "cache_write_tokens": None,
        }
    )


def as_event(entry: RequestLog) -> dict:
    """O registro do jeito que a tabela guarda.

    `fell_back` e derivado em vez de guardado: houve fallback quando alguem
    respondeu DEPOIS de alguma tentativa ter entrado no rastro.

    `started_at` e o INICIO: esta funcao roda quando a requisicao ja terminou,
    entao o instante e o agora menos a duracao medida. Gravar `now()` cru poria
    o fim numa coluna que a janela `since`, a fita e o dossie leem como inicio.
    """
    return {
        "request_id": entry.request_id,
        # `kind` marca a linha como requisicao de modelo. As linhas antigas
        # ficaram NULL na migracao, e NULL tambem significa modelo -- mas a
        # escrita nova deixa o tipo explicito.
        "kind": "model",
        "started_at": datetime.now(UTC) - timedelta(milliseconds=entry.duration_ms),
        "route": entry.route,
        "dialect": entry.dialect,
        "stream": entry.stream,
        "requested_model": entry.requested_model,
        "rule": entry.rule,
        "matched": entry.matched,
        "provider": entry.provider,
        "candidate_model": entry.candidate,
        "status": entry.status,
        "error_type": entry.error_type,
        "input_tokens": entry.input_tokens,
        "output_tokens": entry.output_tokens,
        "ttft_ms": entry.ttft_ms,
        "duration_ms": entry.duration_ms,
        "attempts": list(entry.attempts),
        "fell_back": bool(entry.attempts) and entry.candidate is not None,
        "tools_offered": list(entry.tools_offered),
        "tools_called": list(entry.tools_called),
        "thinking_blocks": entry.thinking_blocks,
        "project": entry.project,
        "session_id": entry.session_id,
        "cached_input_tokens": entry.cached_input_tokens,
        "cache_write_tokens": entry.cache_write_tokens,
        **({"body": entry.body} if entry.body else {}),
    }


def configure_logging() -> None:
    """Da ao logger `shunt` um lugar para escrever, e o nivel pedido.

    Sem isto a linha estruturada nao existe fora do pytest, que anexa um
    handler proprio por conta do `caplog` -- medido: sob `uvicorn
    app.main:app` nenhuma linha aparecia enquanto os testes passavam.

    Chamado pelo `lifespan`, DEPOIS de o uvicorn montar a configuracao de log
    dele. E idempotente porque um `--reload` reexecuta o ciclo de vida, e dois
    handlers no mesmo logger escrevem a mesma linha duas vezes -- um contador
    que leia esse log passaria a contar o dobro.

    A linha ja e um JSON completo, entao o handler nao acrescenta prefixo
    nenhum: qualquer texto antes do `{` quebraria `jq`.
    """
    logger.setLevel(_level_from_env())
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    # A linha e do proxy, nao do uvicorn: propagar a entregaria tambem ao
    # handler raiz, e o operador leria cada requisicao duas vezes.
    logger.propagate = False


def _level_from_env() -> int:
    """`SHUNT_LOG_LEVEL`, ou INFO.

    Um nome que o `logging` nao conhece cai em INFO em vez de derrubar o boot:
    log e observabilidade, nao o servico.
    """
    raw = os.environ.get("SHUNT_LOG_LEVEL", "").strip().upper()
    level = logging.getLevelNamesMapping().get(raw)
    return level if isinstance(level, int) else logging.INFO
