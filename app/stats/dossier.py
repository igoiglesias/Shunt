"""O dossie do periodo: o que se manda para o analista, e nada alem disso.

Nao e o dump do banco. Um modelo lendo 4.000 linhas cruas gasta a janela com
repeticao e devolve conselho generico; o que muda a decisao de quem opera o
proxy sao os agregados, a cadeia, a razao entre ferramenta oferecida e chamada,
e um punhado de requisicoes extremas com o rastro de cada uma.

Tres regras mandam aqui:

- **Filtro identico ao da tela.** O periodo analisado e o periodo que a pessoa
  esta olhando, entao o dossie usa as MESMAS clausulas da busca de auditoria.
  Uma segunda implementacao do filtro daria analise de um recorte diferente do
  que esta na tela, e ninguem perceberia.
- **Total exato, leitura limitada.** O numero de requisicoes vem de um `COUNT`
  sobre o periodo inteiro; os agregados que precisam de Python leem no maximo
  `row_limit` linhas e o dossie DIZ que amostrou. Numero que mente sobre a
  propria origem e pior que numero ausente.
- **Conversa e escolha, com teto.** O texto so entra quando a gravacao esta
  ligada, ja redigido pela captura, e cortado: a conversa inteira de um harness
  custa mais tokens que a analise que ela sustentaria.
"""

from collections import Counter
from datetime import datetime

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from app.config.config import ROW_LIMIT
from app.stats import queries
from app.stats.models import RequestBody, RequestEvent

# Teto de linhas lidas para os agregados em Python (`ROW_LIMIT`, em
# `app/config/config.py`). Cinco mil requisicoes ja descrevem o comportamento
# de um periodo; ler mais muda o custo, nao a conclusao.

# Quantas requisicoes extremas de cada tipo entram no dossie.
TOP = 5
# Quantas conversas viram amostra, e quanto texto cabe em cada ponta.
CONVERSATIONS = 3
SAMPLE_CHARS = 4000

# Os filtros que a tela de auditoria manda e que a busca entende. Fica explicito
# porque o dossie ecoa o que foi pedido, e ecoar um kwarg inventado esconderia
# um filtro que nao foi aplicado.
FILTER_FIELDS = (
    "since",
    "until",
    "text",
    "route",
    "dialect",
    "provider",
    "candidate_model",
    "requested_model",
    "error_type",
    "project",
    "status_min",
    "status_max",
    "stream",
    "fell_back",
    "has_tools",
    "min_duration_ms",
    "min_tokens",
)


def _moment(value) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else value


def _declared(filters: dict) -> dict:
    """So o que foi realmente pedido, ja em forma de JSON."""
    declared = {}
    for field in FILTER_FIELDS:
        # `since` e `until` ja saem em `period`; repeti-los aqui daria duas
        # respostas para "qual foi a janela" dentro do mesmo documento.
        if field in ("since", "until"):
            continue
        value = filters.get(field)
        if value is None or value == "":
            continue
        declared[field] = _moment(value) if isinstance(value, datetime) else value
    return declared


def _latency(values: list[int]) -> dict:
    return {
        "p50": queries._percentile(values, 0.50),
        "p95": queries._percentile(values, 0.95),
        "p99": queries._percentile(values, 0.99),
    }


def _cache(rows: list[RequestEvent]) -> dict:
    """O que os provedores disseram sobre cache -- e sobre quantas requisicoes.

    Vai no dossie porque sem isso o analista conclui pelo que NAO ve: um modelo
    lendo este periodo sem esta secao recomendou ligar um cache que ja estava
    ligado, e propos um `cache_control` que o provedor desta rota nem entende.
    """
    informadas = [row for row in rows if row.cached_input_tokens is not None]
    entrada = sum(row.input_tokens or 0 for row in informadas)
    lido = sum(row.cached_input_tokens or 0 for row in informadas)
    return {
        "reported_requests": len(informadas),
        "silent_requests": len(rows) - len(informadas),
        "cached_input_tokens": lido,
        "input_tokens_of_reported": entrada,
        "hit_rate": round(lido / entrada, 4) if entrada else None,
    }


def _volume(rows: list[RequestEvent], total: int, row_limit: int) -> dict:
    durations = [row.duration_ms for row in rows if row.duration_ms is not None]
    return {
        "requests": total,
        "rows_read": len(rows),
        # Amostrado quando o `COUNT` do periodo passou do que foi lido: os
        # agregados abaixo descrevem a amostra, e quem le a analise tem de saber.
        "sampled": total > len(rows),
        "errors": sum(1 for row in rows if row.status >= 400),
        "streamed": sum(1 for row in rows if row.stream),
        "fell_back": sum(1 for row in rows if row.fell_back),
        "input_tokens": sum(row.input_tokens or 0 for row in rows),
        "output_tokens": sum(row.output_tokens or 0 for row in rows),
        "duration_ms": _latency(durations),
        "cache": _cache(rows),
    }


def _by_model(rows: list[RequestEvent], limit: int) -> list[dict]:
    grouped: dict[tuple[str | None, str | None], dict] = {}
    durations: dict[tuple[str | None, str | None], list[int]] = {}
    for row in rows:
        key = (row.candidate_model, row.provider)
        entry = grouped.setdefault(
            key,
            {
                "candidate_model": row.candidate_model,
                "provider": row.provider,
                "requests": 0,
                "errors": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
        )
        entry["requests"] += 1
        entry["errors"] += 1 if row.status >= 400 else 0
        entry["input_tokens"] += row.input_tokens or 0
        entry["output_tokens"] += row.output_tokens or 0
        if row.duration_ms is not None:
            durations.setdefault(key, []).append(row.duration_ms)
    for key, entry in grouped.items():
        observed = durations.get(key, [])
        entry["p50_ms"] = queries._percentile(observed, 0.50)
        entry["p95_ms"] = queries._percentile(observed, 0.95)
    ordered = sorted(grouped.values(), key=lambda entry: entry["requests"], reverse=True)
    return ordered[:limit]


def _by_requested_model(rows: list[RequestEvent], limit: int) -> list[dict]:
    counted: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    for row in rows:
        # Modelo pedido vazio existe -- `/v1/models` grava assim -- e uma linha
        # sem nome no dossie e uma linha que o analista nao consegue citar.
        name = (row.requested_model or "").strip()
        if not name:
            continue
        counted[name] += 1
        tokens[name] += (row.input_tokens or 0) + (row.output_tokens or 0)
    return [
        {"requested_model": name, "requests": count, "total_tokens": tokens[name]}
        for name, count in counted.most_common(limit)
    ]


def _by_project(rows: list[RequestEvent], limit: int) -> list[dict]:
    """De quais projetos veio o periodo, com o "sem projeto" a mostra.

    Escondido, ele viraria uma conta que nao fecha com o volume, e o analista
    concluiria sobre a parte achando que via o todo.
    """
    counted: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    for row in rows:
        name = (row.project or "").strip() or "sem projeto"
        counted[name] += 1
        tokens[name] += (row.input_tokens or 0) + (row.output_tokens or 0)
    return [
        {"project": name, "requests": count, "total_tokens": tokens[name]}
        for name, count in counted.most_common(limit)
    ]


def _chain(rows: list[RequestEvent], limit: int) -> dict:
    skipped: Counter[tuple[str, str]] = Counter()
    first_try = 0
    for row in rows:
        if not row.attempts:
            if row.candidate_model is not None:
                first_try += 1
            continue
        for attempt in row.attempts:
            # Mesmo separador da consulta do painel: o nome do modelo carrega
            # dois-pontos, e partir no primeiro trunca o nome e desclassifica
            # o motivo.
            name, _, motive = str(attempt).partition(queries.SKIP_DELIMITER)
            reason = queries._skip_reason(motive)
            if reason is None:
                continue
            skipped[(name.strip(), reason)] += 1
    total = len(rows)
    return {
        "requests": total,
        "first_candidate_answered": first_try,
        "first_candidate_rate": round(first_try / total, 4) if total else 0.0,
        "skips": [
            {"candidate": name, "reason": reason, "count": count}
            for (name, reason), count in skipped.most_common(limit)
        ],
    }


def _tools(rows: list[RequestEvent], limit: int) -> list[dict]:
    offered: Counter[str] = Counter()
    called: Counter[str] = Counter()
    for row in rows:
        offered.update(str(name) for name in row.tools_offered or [])
        called.update(str(name) for name in row.tools_called or [])
    # Ordena pelo que foi OFERECIDO: a ferramenta que paga prompt em toda
    # requisicao e nunca e chamada e exatamente a que se quer ver no dossie, e
    # ordenar por chamada a empurraria para fora do teto.
    names = [name for name, _ in offered.most_common(limit)]
    names += [name for name, _ in called.most_common(limit) if name not in names]
    return [
        {
            "tool": name,
            "offered": offered[name],
            "called": called[name],
            "call_rate": round(called[name] / offered[name], 4) if offered[name] else None,
        }
        for name in names[:limit]
    ]


def _extreme(row: RequestEvent) -> dict:
    return {
        "request_id": row.request_id,
        "started_at": queries._utc(row.started_at),
        "route": row.route,
        "requested_model": row.requested_model,
        "provider": row.provider,
        "candidate_model": row.candidate_model,
        "status": row.status,
        "error_type": row.error_type,
        "stream": row.stream,
        "input_tokens": row.input_tokens,
        "output_tokens": row.output_tokens,
        "total_tokens": (row.input_tokens or 0) + (row.output_tokens or 0),
        "duration_ms": row.duration_ms,
        "ttft_ms": row.ttft_ms,
        "fell_back": row.fell_back,
        "attempts": list(row.attempts or []),
        "tools_called": list(row.tools_called or []),
    }


def _errors(rows: list[RequestEvent], limit: int) -> list[dict]:
    grouped: Counter[tuple[int, str | None]] = Counter()
    for row in rows:
        if row.status >= 400:
            grouped[(row.status, row.error_type)] += 1
    return [
        {"status": status, "error_type": error_type, "requests": count}
        for (status, error_type), count in grouped.most_common(limit)
    ]


def _conversations(
    session: Session, rows: list[RequestEvent], how_many: int, sample_chars: int
) -> list[dict]:
    """As conversas das requisicoes mais caras que tiverem texto gravado.

    Mais caras, e nao as primeiras: quem pede a analise quer saber onde o token
    esta indo, e a conversa de uma requisicao barata nao responde isso.
    """
    ordered = sorted(
        rows,
        key=lambda row: (row.input_tokens or 0) + (row.output_tokens or 0),
        reverse=True,
    )
    sample: list[dict] = []
    for row in ordered:
        if len(sample) == how_many:
            break
        body = session.get(RequestBody, row.request_id)
        if body is None:
            continue
        prompt = body.prompt or ""
        answer = body.answer or ""
        sample.append(
            {
                "request_id": row.request_id,
                "candidate_model": row.candidate_model,
                "total_tokens": (row.input_tokens or 0) + (row.output_tokens or 0),
                "prompt": prompt[:sample_chars],
                "answer": answer[:sample_chars],
                # Cortado aqui ou ja cortado na gravacao: as duas coisas mudam o
                # que o analista pode concluir do texto.
                "truncated": bool(body.truncated)
                or len(prompt) > sample_chars
                or len(answer) > sample_chars,
            }
        )
    return sample


def build(
    engine: Engine,
    *,
    filters: dict | None = None,
    row_limit: int = ROW_LIMIT,
    top: int = TOP,
    include_conversations: bool = True,
    conversations: int = CONVERSATIONS,
    sample_chars: int = SAMPLE_CHARS,
) -> dict:
    """O periodo inteiro num documento, pronto para virar prompt."""
    filters = dict(filters or {})
    # Filtro desconhecido e recusado em vez de ignorado: um nome errado passaria
    # despercebido e a analise descreveria um recorte MAIOR do que o da tela,
    # com cara de ter obedecido o filtro.
    unknown = sorted(set(filters) - set(FILTER_FIELDS))
    if unknown:
        raise ValueError(f"filtro desconhecido no dossie: {', '.join(unknown)}")
    declared = _declared(filters)
    # Por NOME, e nao por posicao: a lista de filtros cresce, e um argumento a
    # mais no meio faria o valor de um filtro chegar como outro sem erro nenhum.
    # `kind` e FIXO em "model" e nao entra em `FILTER_FIELDS`: o dossie analisa
    # o trafego de modelo, e a linha de relay e rastro de rede. A pessoa pede o
    # dossie do periodo que esta vendo na tela, e a tela mostra modelo.
    clauses = queries._search_clauses(
        **{field: filters.get(field) for field in FILTER_FIELDS},
        kind="model",
    )

    with Session(engine) as session:
        total = int(
            session.scalar(select(func.count()).select_from(RequestEvent).where(*clauses)) or 0
        )
        rows = list(
            session.scalars(
                select(RequestEvent)
                .where(*clauses)
                .order_by(RequestEvent.started_at.desc(), RequestEvent.id.desc())
                .limit(row_limit)
            )
        )
        sample = (
            _conversations(session, rows, conversations, sample_chars)
            if include_conversations
            else []
        )

    slowest = sorted(rows, key=lambda row: row.duration_ms or 0, reverse=True)[:top]
    costliest = sorted(
        rows,
        key=lambda row: (row.input_tokens or 0) + (row.output_tokens or 0),
        reverse=True,
    )[:top]
    return {
        "period": {
            "since": _moment(filters.get("since")),
            "until": _moment(filters.get("until")),
            "filters": declared,
        },
        "volume": _volume(rows, total, row_limit),
        "by_model": _by_model(rows, top * 2),
        "by_requested_model": _by_requested_model(rows, top * 2),
        "by_project": _by_project(rows, top * 2),
        "chain": _chain(rows, top * 2),
        "tools": _tools(rows, top * 2),
        "errors": _errors(rows, top * 2),
        "slowest": [_extreme(row) for row in slowest],
        "costliest": [_extreme(row) for row in costliest],
        "conversations": sample,
    }
