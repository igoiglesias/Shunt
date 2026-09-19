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
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from app.stats.recorder import Recorder

logger = logging.getLogger("shunt")

# Comparados sempre em caixa baixa: o cabecalho chega na caixa que o cliente
# digitou, e `X-Api-Key` passaria inteiro por uma comparacao literal.
SECRET_HEADERS = frozenset({"authorization", "x-api-key", "api-key", "proxy-authorization"})

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
    input_tokens: int = 0
    output_tokens: int = 0
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


def as_event(entry: RequestLog) -> dict:
    """O registro do jeito que a tabela guarda.

    `fell_back` e derivado em vez de guardado: houve fallback quando alguem
    respondeu DEPOIS de alguma tentativa ter entrado no rastro.
    """
    return {
        "request_id": entry.request_id,
        "started_at": datetime.now(UTC),
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
