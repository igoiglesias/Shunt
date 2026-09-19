"""As perguntas que o painel faz, cada uma com janela e teto.

Tres regras valem para todas elas:

- **Janela obrigatoria.** Nenhuma consulta varre a tabela inteira; toda uma
  recebe `hours` e filtra por `started_at`. E por isso que o indice de tempo
  existe.
- **Teto obrigatorio.** Agrupamento devolve no maximo `limit` linhas. Um painel
  que tenta desenhar 4.000 modelos nao fica mais informativo, fica ilegivel.
- **Nada de objeto do ORM para fora.** Cada funcao devolve `dict` e `list`
  prontos para virar JSON. O painel nao conhece SQLAlchemy.

As ferramentas ficam em JSON dentro da linha, entao a contagem delas acontece
em Python: sao poucas por requisicao, e um `json_each` amarraria a consulta ao
dialeto do SQLite.
"""

from collections import Counter
from datetime import UTC, datetime, timedelta

from sqlalchemy import Engine, case, func, select
from sqlalchemy.orm import Session

from app.stats.models import RequestEvent

DEFAULT_HOURS = 24
DEFAULT_LIMIT = 10


def _utc(moment: datetime | None) -> str | None:
    """Instante sempre com fuso explicito, em ISO.

    O SQLite devolve o `datetime` ingenuo mesmo em coluna com `timezone=True`.
    Quem grava e o `Recorder`, sempre em UTC, entao o ingenuo que volta e UTC --
    e dizer isso e o que impede o navegador de ler como hora local.
    """
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.isoformat()


def _since(hours: float) -> datetime:
    return datetime.now(UTC) - timedelta(hours=hours)


def _percentile(values: list[int], fraction: float) -> int | None:
    """Percentil por indice, sem numpy e sem interpolacao.

    Interpolar entre dois pontos daria um numero que nenhuma requisicao viveu.
    Para ler latencia, o valor observado e mais honesto.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def totals(engine: Engine, hours: float = DEFAULT_HOURS) -> dict:
    """A faixa de cima: volume, tokens, erro e latencia."""
    since = _since(hours)
    with Session(engine) as session:
        row = session.execute(
            select(
                func.count(RequestEvent.id),
                func.coalesce(func.sum(RequestEvent.input_tokens), 0),
                func.coalesce(func.sum(RequestEvent.output_tokens), 0),
                func.coalesce(func.sum(case((RequestEvent.status >= 400, 1), else_=0)), 0),
                func.coalesce(func.sum(case((RequestEvent.fell_back, 1), else_=0)), 0),
                func.coalesce(func.sum(case((RequestEvent.stream, 1), else_=0)), 0),
            ).where(RequestEvent.started_at >= since)
        ).one()
        durations = list(
            session.scalars(
                select(RequestEvent.duration_ms).where(RequestEvent.started_at >= since)
            )
        )
        ttfts = [
            value
            for value in session.scalars(
                select(RequestEvent.ttft_ms).where(
                    RequestEvent.started_at >= since, RequestEvent.ttft_ms.is_not(None)
                )
            )
            if value is not None
        ]
    requests, input_tokens, output_tokens, errors, fallbacks, streams = row
    return {
        "requests": requests,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "errors": errors,
        "error_rate": round(errors / requests, 4) if requests else 0.0,
        "fallbacks": fallbacks,
        "streams": streams,
        "p50_duration_ms": _percentile(durations, 0.50),
        "p95_duration_ms": _percentile(durations, 0.95),
        "p50_ttft_ms": _percentile(ttfts, 0.50),
        "p95_ttft_ms": _percentile(ttfts, 0.95),
    }


def per_hour(engine: Engine, hours: float = DEFAULT_HOURS) -> list[dict]:
    """A serie temporal: uma linha por hora, com requisicoes e tokens."""
    since = _since(hours)
    bucket = func.strftime("%Y-%m-%dT%H:00", RequestEvent.started_at)
    with Session(engine) as session:
        rows = session.execute(
            select(
                bucket,
                func.count(RequestEvent.id),
                func.coalesce(func.sum(RequestEvent.input_tokens), 0),
                func.coalesce(func.sum(RequestEvent.output_tokens), 0),
                func.coalesce(func.sum(case((RequestEvent.status >= 400, 1), else_=0)), 0),
            )
            .where(RequestEvent.started_at >= since)
            .group_by(bucket)
            .order_by(bucket)
        ).all()
    return [
        {
            "hour": hour,
            "requests": requests,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "errors": errors,
        }
        for hour, requests, input_tokens, output_tokens, errors in rows
    ]


def _grouped(engine: Engine, column, hours: float, limit: int, label: str) -> list[dict]:
    since = _since(hours)
    with Session(engine) as session:
        rows = session.execute(
            select(
                column,
                func.count(RequestEvent.id),
                func.coalesce(func.sum(RequestEvent.input_tokens + RequestEvent.output_tokens), 0),
                func.coalesce(func.sum(case((RequestEvent.status >= 400, 1), else_=0)), 0),
                func.coalesce(func.avg(RequestEvent.duration_ms), 0),
            )
            .where(RequestEvent.started_at >= since, column.is_not(None), column != "")
            .group_by(column)
            .order_by(func.count(RequestEvent.id).desc())
            .limit(limit)
        ).all()
    return [
        {
            label: value,
            "requests": requests,
            "tokens": tokens,
            "errors": errors,
            "avg_duration_ms": int(avg),
        }
        for value, requests, tokens, errors, avg in rows
    ]


def by_model(engine: Engine, hours: float = DEFAULT_HOURS, limit: int = DEFAULT_LIMIT) -> list[dict]:
    return _grouped(engine, RequestEvent.candidate_model, hours, limit, "model")


def by_provider(
    engine: Engine, hours: float = DEFAULT_HOURS, limit: int = DEFAULT_LIMIT
) -> list[dict]:
    return _grouped(engine, RequestEvent.provider, hours, limit, "provider")


def by_route(engine: Engine, hours: float = DEFAULT_HOURS, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """Inclui `count_tokens` e a listagem de modelos, que nao passam pelo dispatcher."""
    return _grouped(engine, RequestEvent.route, hours, limit, "route")


def by_requested_model(
    engine: Engine, hours: float = DEFAULT_HOURS, limit: int = DEFAULT_LIMIT
) -> list[dict]:
    """O que o cliente PEDIU, que e outra pergunta: `haiku` pedido, Groq servido."""
    return _grouped(engine, RequestEvent.requested_model, hours, limit, "requested_model")


def errors_by_type(
    engine: Engine, hours: float = DEFAULT_HOURS, limit: int = DEFAULT_LIMIT
) -> list[dict]:
    since = _since(hours)
    with Session(engine) as session:
        rows = session.execute(
            select(RequestEvent.status, RequestEvent.error_type, func.count(RequestEvent.id))
            .where(RequestEvent.started_at >= since, RequestEvent.status >= 400)
            .group_by(RequestEvent.status, RequestEvent.error_type)
            .order_by(func.count(RequestEvent.id).desc())
            .limit(limit)
        ).all()
    return [
        {"status": status, "error_type": error_type, "requests": count}
        for status, error_type, count in rows
    ]


def chain_health(engine: Engine, hours: float = DEFAULT_HOURS) -> dict:
    """Quantas vezes o primeiro candidato bastou, e quem foi pulado quando nao."""
    since = _since(hours)
    with Session(engine) as session:
        rows = list(
            session.execute(
                select(RequestEvent.attempts, RequestEvent.candidate_model).where(
                    RequestEvent.started_at >= since
                )
            ).all()
        )
    skipped: Counter[str] = Counter()
    first_try = 0
    for attempts, candidate in rows:
        if not attempts:
            if candidate is not None:
                first_try += 1
            continue
        for attempt in attempts:
            skipped[str(attempt).split(":", 1)[0]] += 1
    total = len(rows)
    return {
        "requests": total,
        "first_candidate_answered": first_try,
        "first_candidate_rate": round(first_try / total, 4) if total else 0.0,
        "skips": [{"candidate": name, "count": count} for name, count in skipped.most_common(10)],
    }


def tool_usage(
    engine: Engine, hours: float = DEFAULT_HOURS, limit: int = DEFAULT_LIMIT
) -> list[dict]:
    """Oferecida e chamada lado a lado: a razao diz quais ferramentas pagam o prompt."""
    since = _since(hours)
    with Session(engine) as session:
        rows = list(
            session.execute(
                select(RequestEvent.tools_offered, RequestEvent.tools_called).where(
                    RequestEvent.started_at >= since
                )
            ).all()
        )
    offered: Counter[str] = Counter()
    called: Counter[str] = Counter()
    for tools_offered, tools_called in rows:
        offered.update(str(name) for name in tools_offered or [])
        called.update(str(name) for name in tools_called or [])
    names = [name for name, _ in called.most_common(limit)]
    names += [name for name, _ in offered.most_common(limit) if name not in names]
    return [
        {
            "tool": name,
            "offered": offered[name],
            "called": called[name],
            "call_rate": round(called[name] / offered[name], 4) if offered[name] else None,
        }
        for name in names[:limit]
    ]


def recent(engine: Engine, limit: int = 20) -> list[dict]:
    """As ultimas requisicoes, que e o que se olha quando algo acabou de quebrar."""
    with Session(engine) as session:
        rows = list(
            session.scalars(select(RequestEvent).order_by(RequestEvent.id.desc()).limit(limit))
        )
    return [
        {
            "request_id": row.request_id,
            "started_at": _utc(row.started_at),
            "route": row.route,
            "requested_model": row.requested_model,
            "provider": row.provider,
            "candidate_model": row.candidate_model,
            "status": row.status,
            "stream": row.stream,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "duration_ms": row.duration_ms,
            "ttft_ms": row.ttft_ms,
            "fell_back": row.fell_back,
            "tools_called": list(row.tools_called or []),
        }
        for row in rows
    ]


def snapshot(engine: Engine, hours: float = DEFAULT_HOURS) -> dict:
    """Tudo que o painel desenha, num documento so."""
    return {
        "window_hours": hours,
        "totals": totals(engine, hours),
        "per_hour": per_hour(engine, hours),
        "by_model": by_model(engine, hours),
        "by_provider": by_provider(engine, hours),
        "by_route": by_route(engine, hours),
        "by_requested_model": by_requested_model(engine, hours),
        "errors": errors_by_type(engine, hours),
        "chain": chain_health(engine, hours),
        "tools": tool_usage(engine, hours),
        "recent": recent(engine),
    }
