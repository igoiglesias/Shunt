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


def redact(headers: dict[str, str]) -> dict[str, str]:
    """Os mesmos cabecalhos, com o VALOR das credenciais trocado.

    O nome fica: saber que um `x-api-key` chegou e metade do diagnostico de
    uma autenticacao recusada, e o nome sozinho nao autentica ninguem.
    """
    return {k: (REDACTED if k.lower() in SECRET_HEADERS else v) for k, v in headers.items()}


def log_request(entry: RequestLog) -> None:
    logger.info(json.dumps(asdict(entry), ensure_ascii=False))


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
