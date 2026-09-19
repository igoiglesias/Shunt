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

import json
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections import Counter
from datetime import UTC, datetime, timedelta

from sqlalchemy import Engine, String, and_, case, cast, delete, func, or_, select
from sqlalchemy.orm import Session

from app.stats.models import RequestBody, RequestEvent

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


SEARCH_LIMIT = 50
MAX_SEARCH_LIMIT = 500


def encode_cursor(row: RequestEvent) -> str:
    """A marca da ultima linha entregue, opaca para quem chama.

    Carrega os tres campos que as duas ordenacoes usam, para que trocar de
    ordem no meio da paginacao nao devolva lixo -- devolve outra pagina.
    """
    mark = {
        "id": row.id,
        "started_at": _utc(row.started_at),
        "duration_ms": row.duration_ms,
    }
    return urlsafe_b64encode(json.dumps(mark).encode()).decode().rstrip("=")


def decode_cursor(cursor: str | None) -> dict | None:
    """O cursor de volta, ou None quando ele nao serve.

    Cursor quebrado -- copiado pela metade de uma URL, de outra versao -- e
    "comece do inicio", e nao um erro na cara de quem esta investigando.
    """
    if not cursor:
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        mark = json.loads(urlsafe_b64decode(padded.encode()).decode())
    except (ValueError, TypeError):
        return None
    if not isinstance(mark, dict) or "id" not in mark:
        return None
    return mark


def _parse_moment(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(UTC)
    return datetime.fromisoformat(raw)


def _as_event(row: RequestEvent) -> dict:
    """Uma linha inteira, pronta para virar JSON. A tela de auditoria le tudo."""
    return {
        "id": row.id,
        "request_id": row.request_id,
        "started_at": _utc(row.started_at),
        "route": row.route,
        "dialect": row.dialect,
        "stream": row.stream,
        "requested_model": row.requested_model,
        "rule": row.rule,
        "matched": row.matched,
        "provider": row.provider,
        "candidate_model": row.candidate_model,
        "status": row.status,
        "error_type": row.error_type,
        "input_tokens": row.input_tokens,
        "output_tokens": row.output_tokens,
        "ttft_ms": row.ttft_ms,
        "duration_ms": row.duration_ms,
        "attempts": list(row.attempts or []),
        "fell_back": row.fell_back,
        "tools_offered": list(row.tools_offered or []),
        "tools_called": list(row.tools_called or []),
        "thinking_blocks": row.thinking_blocks,
    }


def _text_clause(text: str):
    """Trecho livre onde um humano procura: id, modelos, ferramentas, cadeia.

    As listas ficam em JSON dentro da linha, e um `LIKE` sobre o texto do JSON
    acha o nome da ferramenta e o motivo do pulo sem prender a consulta a uma
    funcao de JSON que so o SQLite tem.
    """
    needle = f"%{text.strip()}%"
    columns = (
        RequestEvent.request_id,
        RequestEvent.requested_model,
        RequestEvent.candidate_model,
        RequestEvent.provider,
        RequestEvent.route,
        RequestEvent.error_type,
        RequestEvent.matched,
        cast(RequestEvent.tools_offered, String),
        cast(RequestEvent.tools_called, String),
        cast(RequestEvent.attempts, String),
    )
    return or_(*[column.like(needle) for column in columns])


def _search_clauses(
    since: datetime | None,
    until: datetime | None,
    text: str | None,
    route: str | None,
    dialect: str | None,
    provider: str | None,
    candidate_model: str | None,
    requested_model: str | None,
    error_type: str | None,
    status_min: int | None,
    status_max: int | None,
    stream: bool | None,
    fell_back: bool | None,
    has_tools: bool | None,
    min_duration_ms: int | None,
    min_tokens: int | None,
) -> list:
    clauses = []
    if since is not None:
        clauses.append(RequestEvent.started_at >= since)
    if until is not None:
        clauses.append(RequestEvent.started_at <= until)
    if text:
        clauses.append(_text_clause(text))
    for column, value in (
        (RequestEvent.route, route),
        (RequestEvent.dialect, dialect),
        (RequestEvent.provider, provider),
        (RequestEvent.candidate_model, candidate_model),
        (RequestEvent.requested_model, requested_model),
        (RequestEvent.error_type, error_type),
    ):
        if value:
            clauses.append(column == value)
    if status_min is not None:
        clauses.append(RequestEvent.status >= status_min)
    if status_max is not None:
        clauses.append(RequestEvent.status <= status_max)
    if stream is not None:
        clauses.append(RequestEvent.stream.is_(stream))
    if fell_back is not None:
        clauses.append(RequestEvent.fell_back.is_(fell_back))
    if has_tools is not None:
        # Lista vazia em JSON e "[]": quem chamou ferramenta tem mais que isso.
        empty = cast(RequestEvent.tools_called, String).in_(("[]", "null"))
        clauses.append(~empty if has_tools else empty)
    if min_duration_ms is not None:
        clauses.append(RequestEvent.duration_ms >= min_duration_ms)
    if min_tokens is not None:
        clauses.append(RequestEvent.input_tokens + RequestEvent.output_tokens >= min_tokens)
    return clauses


def search_events(
    engine: Engine,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    text: str | None = None,
    route: str | None = None,
    dialect: str | None = None,
    provider: str | None = None,
    candidate_model: str | None = None,
    requested_model: str | None = None,
    error_type: str | None = None,
    status_min: int | None = None,
    status_max: int | None = None,
    stream: bool | None = None,
    fell_back: bool | None = None,
    has_tools: bool | None = None,
    min_duration_ms: int | None = None,
    min_tokens: int | None = None,
    order_by: str = "time",
    limit: int = SEARCH_LIMIT,
    cursor: str | None = None,
) -> dict:
    """Uma pagina de requisicoes, o total da busca inteira, e o proximo cursor.

    O total conta a busca, e nao a pagina: um `LIMIT` sozinho mente sobre o
    tamanho do resultado assim que a tabela cresce, e a tela precisa dizer
    "1 a 50 de 3.412".

    O cursor e o `id` da ultima linha entregue, e nao um deslocamento: linha
    nova chegando durante a paginacao empurraria um `OFFSET` e faria a proxima
    pagina repetir o que ja foi visto.
    """
    limit = max(1, min(limit, MAX_SEARCH_LIMIT))
    by_duration = order_by == "duration"
    clauses = _search_clauses(
        since, until, text, route, dialect, provider, candidate_model, requested_model,
        error_type, status_min, status_max, stream, fell_back, has_tools,
        min_duration_ms, min_tokens,
    )
    # Ordenar por `id` NAO e ordenar por tempo: o id cresce com a INSERCAO, e o
    # gravador entrega em lote, entao duas requisicoes da mesma rajada podem
    # entrar fora da ordem em que aconteceram. O `id` entra so como desempate,
    # que e o que torna o cursor estavel.
    ordering = (
        (RequestEvent.duration_ms.desc(), RequestEvent.id.desc())
        if by_duration
        else (RequestEvent.started_at.desc(), RequestEvent.id.desc())
    )
    with Session(engine) as session:
        total = session.scalar(select(func.count()).select_from(RequestEvent).where(*clauses)) or 0
        paged = list(clauses)
        mark = decode_cursor(cursor)
        if mark is not None:
            key = RequestEvent.duration_ms if by_duration else RequestEvent.started_at
            value = (
                mark.get("duration_ms")
                if by_duration
                else _parse_moment(mark.get("started_at"))
            )
            # Cursor de uma versao anterior pode nao trazer o campo desta
            # ordenacao; faltando, a busca comeca do inicio em vez de estourar.
            if value is not None:
                paged.append(
                    or_(key < value, and_(key == value, RequestEvent.id < mark["id"]))
                )
        rows = list(
            session.scalars(
                select(RequestEvent).where(*paged).order_by(*ordering).limit(limit + 1)
            )
        )
    has_more = len(rows) > limit
    rows = rows[:limit]
    return {
        "events": [_as_event(row) for row in rows],
        "total": int(total),
        "next_cursor": encode_cursor(rows[-1]) if has_more and rows else None,
    }


def event_detail(engine: Engine, request_id: str) -> dict | None:
    """Uma requisicao inteira pelo id, ou None. A tela abre o detalhe por aqui."""
    with Session(engine) as session:
        row = session.scalars(
            select(RequestEvent).where(RequestEvent.request_id == request_id).limit(1)
        ).first()
        return _as_event(row) if row is not None else None


def body_of(engine: Engine, request_id: str) -> dict | None:
    """O texto da conversa de UMA requisicao, ou None quando nao foi gravada."""
    with Session(engine) as session:
        row = session.get(RequestBody, request_id)
        if row is None:
            return None
        return {
            "request_id": row.request_id,
            "prompt": row.prompt or "",
            "answer": row.answer or "",
            "prompt_bytes": row.prompt_bytes,
            "answer_bytes": row.answer_bytes,
            "truncated": row.truncated,
        }


def facets(engine: Engine, hours: float = 24 * 30, limit: int = 60) -> dict:
    """Os valores que existem de verdade, para a tela montar os seletores.

    Oferecer um provedor que nunca respondeu nada e um filtro que so devolve
    lista vazia. Cada lista vem com a contagem, porque saber que um modelo
    aparece tres vezes muda a decisao de clicar nele.
    """
    since = _since(hours)

    def distinct(column) -> list[dict]:
        with Session(engine) as session:
            rows = session.execute(
                select(column, func.count(RequestEvent.id))
                .where(RequestEvent.started_at >= since, column.is_not(None), column != "")
                .group_by(column)
                .order_by(func.count(RequestEvent.id).desc())
                .limit(limit)
            ).all()
        return [{"value": value, "count": count} for value, count in rows]

    return {
        "providers": distinct(RequestEvent.provider),
        "candidate_models": distinct(RequestEvent.candidate_model),
        "requested_models": distinct(RequestEvent.requested_model),
        "routes": distinct(RequestEvent.route),
        "error_types": distinct(RequestEvent.error_type),
    }


def delete_events(engine: Engine, older_than_hours: float | None = None) -> int:
    """Apaga o historico e devolve quantas linhas sairam.

    Sem `older_than_hours` limpa tudo; com ele guarda o recorte recente, que e
    o caso de quem quer zerar o passado sem perder o que esta acontecendo
    agora. Apagar estatistica nao afeta uma requisicao em voo: o gravador so
    insere, e a fila dele nao e consultada aqui.
    """
    statement = delete(RequestEvent)
    bodies_statement = delete(RequestBody)
    if older_than_hours is not None:
        cut = _since(older_than_hours)
        statement = statement.where(RequestEvent.started_at < cut)
        # O texto vai junto: conversa sem o evento dela nao serve a ninguem, e
        # ficaria invisivel para sempre na tela de auditoria.
        bodies_statement = bodies_statement.where(
            RequestBody.request_id.in_(
                select(RequestEvent.request_id).where(RequestEvent.started_at < cut)
            )
        )
    with Session(engine) as session:
        session.execute(bodies_statement)
        result = session.execute(statement)
        # `rowcount` existe no resultado de um DELETE, mas nao na assinatura
        # generica de `execute`; o `getattr` e para o verificador de tipos, e
        # nao para o tempo de execucao.
        removed = getattr(result, "rowcount", 0) or 0
        session.commit()
    return int(removed)


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
