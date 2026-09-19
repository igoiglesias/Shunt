"""As rotas da tela de auditoria: buscar, abrir uma requisicao, exportar.

A pergunta aqui e outra: o painel responde "como esta indo", e isto responde "o
que aconteceu naquela requisicao". Por isso a busca aceita filtro combinado,
devolve o total da busca inteira e pagina por cursor, e o detalhe traz a cadeia
de tentativas na ordem em que aconteceu.

O que NAO existe aqui, e nao e esquecimento: prompt e resposta. Eles nunca sao
gravados -- e a promessa do README -- entao nao ha o que mostrar.
"""

import asyncio
import csv
import io
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.stats import queries

router = APIRouter()

# Colunas da exportacao, na ordem em que uma planilha faz sentido.
CSV_COLUMNS = (
    "started_at",
    "request_id",
    "route",
    "dialect",
    "stream",
    "requested_model",
    "provider",
    "candidate_model",
    "status",
    "error_type",
    "input_tokens",
    "output_tokens",
    "ttft_ms",
    "duration_ms",
    "fell_back",
    "attempts",
    "tools_offered",
    "tools_called",
)
EXPORT_LIMIT = 5000


def _engine(request: Request):
    recorder = getattr(request.app.state, "recorder", None)
    return recorder.engine if recorder is not None else None


def _text(params, name: str) -> str | None:
    value = params.get(name)
    return value.strip() or None if isinstance(value, str) else None


def _flag(params, name: str) -> bool | None:
    """Tri-estado: ausente e "tanto faz", e nao "falso".

    Sem isso um filtro desmarcado na tela viraria "so as que nao fizeram
    streaming", que e o oposto do que o usuario quis dizer.
    """
    raw = params.get(name)
    if raw is None or raw == "":
        return None
    return raw.lower() in {"1", "true", "sim", "on", "yes"}


def _number(params, name: str) -> int | None:
    raw = params.get(name)
    try:
        return int(raw) if raw not in (None, "") else None
    except ValueError:
        return None


def _moment(params, name: str) -> datetime | None:
    """Instante ISO, sempre com fuso.

    Sem offset, assume UTC, que e como o gravador escreve. O "Z" do final ja e
    entendido pelo `fromisoformat` desde o Python 3.11.
    """
    raw = params.get(name)
    if not raw:
        return None
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def filters_from(params) -> dict:
    """Os parametros da URL virando argumentos da busca.

    Um filtro que nao parseia e um filtro ausente: quem esta investigando
    prefere ver o resultado mais largo a ver um 400 por causa de uma data
    digitada pela metade.
    """
    return {
        "since": _moment(params, "since"),
        "until": _moment(params, "until"),
        "text": _text(params, "text"),
        "route": _text(params, "route"),
        "dialect": _text(params, "dialect"),
        "provider": _text(params, "provider"),
        "candidate_model": _text(params, "candidate_model"),
        "requested_model": _text(params, "requested_model"),
        "error_type": _text(params, "error_type"),
        "status_min": _number(params, "status_min"),
        "status_max": _number(params, "status_max"),
        "stream": _flag(params, "stream"),
        "fell_back": _flag(params, "fell_back"),
        "has_tools": _flag(params, "has_tools"),
        "min_duration_ms": _number(params, "min_duration_ms"),
        "min_tokens": _number(params, "min_tokens"),
        "order_by": "duration" if params.get("order_by") == "duration" else "time",
    }


def _no_database() -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={"error": "sem banco configurado: nao ha historico para auditar"},
    )


@router.get("/api/requests")
async def search_requests(request: Request):
    engine = _engine(request)
    if engine is None:
        return _no_database()
    params = request.query_params
    filters = filters_from(params)
    limit = _number(params, "limit") or queries.SEARCH_LIMIT
    cursor = _text(params, "cursor")
    # Em thread, como toda consulta: o dialeto do libsql e sincrono e o laco de
    # eventos esta servindo requisicoes de modelo ao mesmo tempo.
    page = await asyncio.to_thread(
        queries.search_events, engine, limit=limit, cursor=cursor, **filters
    )
    return JSONResponse(page)


@router.get("/api/requests/export")
async def export_requests(request: Request):
    """A mesma busca em CSV, para levar a investigacao para outro lugar."""
    engine = _engine(request)
    if engine is None:
        return _no_database()
    filters = filters_from(request.query_params)
    page = await asyncio.to_thread(
        queries.search_events, engine, limit=EXPORT_LIMIT, **filters
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for event in page["events"]:
        writer.writerow(
            {
                **event,
                # Listas viram texto separado por espaco: uma planilha nao le
                # JSON, e o nome da ferramenta e o que se procura ali.
                "attempts": " | ".join(event["attempts"]),
                "tools_offered": " ".join(event["tools_offered"]),
                "tools_called": " ".join(event["tools_called"]),
            }
        )
    buffer.seek(0)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    cut = page["total"] > len(page["events"])
    headers = {
        "content-disposition": f'attachment; filename="shunt-requests-{stamp}.csv"',
        # Quem exporta precisa saber que levou uma parte. Um cabecalho porque o
        # CSV em si nao tem onde dizer isso sem sujar a planilha, e a tela le
        # daqui para avisar antes do download.
        "x-shunt-exported": str(len(page["events"])),
        "x-shunt-total": str(page["total"]),
        "x-shunt-truncated": "1" if cut else "0",
    }
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


@router.get("/api/requests/facets")
async def facets(request: Request):
    """Os valores que existem, para os seletores da tela."""
    engine = _engine(request)
    if engine is None:
        return _no_database()
    return JSONResponse(await asyncio.to_thread(queries.facets, engine))


@router.get("/api/requests/{request_id}/body")
async def request_body(request: Request, request_id: str):
    """O texto da conversa daquela requisicao.

    404 quando nao ha: pode ser que a gravacao de conversa estivesse desligada
    quando ela passou, e a tela diz isso em vez de mostrar vazio.
    """
    engine = _engine(request)
    if engine is None:
        return _no_database()
    body = await asyncio.to_thread(queries.body_of, engine, request_id)
    if body is None:
        return JSONResponse(
            status_code=404,
            content={"error": "conversa nao gravada para esta requisicao"},
        )
    return JSONResponse(body)


@router.get("/api/requests/{request_id}")
async def one_request(request: Request, request_id: str):
    engine = _engine(request)
    if engine is None:
        return _no_database()
    event = await asyncio.to_thread(queries.event_detail, engine, request_id)
    if event is None:
        return JSONResponse(status_code=404, content={"error": "requisicao nao encontrada"})
    return JSONResponse(event)
