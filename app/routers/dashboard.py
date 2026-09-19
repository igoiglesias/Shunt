"""A API do painel: um resumo cacheado e um stream que nao toca o banco.

As duas rotas aqui existem para o navegador, e nenhuma delas pode custar
latencia a API:

- `/api/stats` cacheia o resumo por alguns segundos. Recarregar a pagina em
  laco nao multiplica consulta.
- `/api/stats/stream` escuta o barramento em memoria do gravador. Navegador
  aberto nao vira consulta repetida -- ele recebe o que ja passou pela fila.

Sem banco configurado as duas respondem assim mesmo: o resumo vem vazio com
`enabled: false`, e o stream continua entregando o ao vivo. Um painel que
quebra sem Turso seria pior do que um painel vazio.
"""

import asyncio
import json
import math
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.stats import queries

router = APIRouter()

# Janela padrao e tetos. A janela vem do cliente, mas presa entre limites: uma
# consulta de 10 anos varreria a tabela inteira, que e o que os indices existem
# para evitar.
DEFAULT_HOURS = 24.0
MAX_HOURS = 24.0 * 30
# Um minuto e o menor recorte util: abaixo disso a janela nao contem nem uma
# conversa inteira de um harness.
MIN_HOURS = 1 / 60
CACHE_SECONDS = 5.0
# Batida de keepalive do SSE. Proxy no meio do caminho derruba conexao ociosa,
# e um painel que morre sozinho depois de cinco minutos parados nao serve.
KEEPALIVE_SECONDS = 15.0

_cache: dict[float, tuple[float, dict]] = {}


def _window(raw: str | None) -> float:
    """A janela pedida, presa entre um minuto e trinta dias.

    Fracionaria porque o painel oferece cinco minutos, que e a pergunta de quem
    esta olhando enquanto trabalha: o que acabou de acontecer.
    """
    try:
        hours = float(raw) if raw is not None else DEFAULT_HOURS
    except ValueError:
        return DEFAULT_HOURS
    if math.isnan(hours):  # "nan" parseia como float e nao e uma janela
        return DEFAULT_HOURS
    return max(MIN_HOURS, min(hours, MAX_HOURS))


def _health(request: Request) -> dict:
    """O rodape do painel: o que o operador precisa para saber se confiar nele."""
    recorder = getattr(request.app.state, "recorder", None)
    if recorder is None:
        return {
            "enabled": False,
            "configured": False,
            "queued": 0,
            "dropped": 0,
            "failures": 0,
            "commits": 0,
            "reconnects": 0,
        }
    return {
        "enabled": recorder.enabled,
        # `configured` sem `enabled` e um banco declarado que nao abriu: o
        # worker tenta de novo sozinho, e dizer isso evita que o operador leia
        # "sem banco" e va procurar configuracao que ja esta certa.
        "configured": recorder.configured,
        "queued": recorder.queued,
        "dropped": recorder.dropped,
        "failures": recorder.failures,
        "commits": recorder.commits,
        "reconnects": recorder.reconnects,
    }


def _engine(request: Request):
    recorder = getattr(request.app.state, "recorder", None)
    return recorder.engine if recorder is not None else None


def empty_snapshot(hours: float) -> dict:
    """O mesmo documento, com tudo zerado. O painel nao precisa de dois desenhos."""
    return {
        "window_hours": hours,
        "totals": {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "errors": 0,
            "error_rate": 0.0,
            "fallbacks": 0,
            "streams": 0,
            "p50_duration_ms": None,
            "p95_duration_ms": None,
            "p50_ttft_ms": None,
            "p95_ttft_ms": None,
        },
        "series": {"bucket_minutes": 60, "points": []},
        "by_model": [],
        "by_provider": [],
        "by_route": [],
        "by_requested_model": [],
        "pairs": [],
        "errors": [],
        "chain": {
            "requests": 0,
            "first_candidate_answered": 0,
            "first_candidate_rate": 0.0,
            "skips": [],
        },
        "tools": [],
        "recent": [],
    }


@router.get("/api/stats")
async def stats(request: Request) -> dict:
    hours = _window(request.query_params.get("window"))
    engine = _engine(request)
    if engine is None:
        return {**empty_snapshot(hours), "health": _health(request)}
    cached = _cache.get(hours)
    now = time.monotonic()
    if cached is not None and now - cached[0] < CACHE_SECONDS:
        return {**cached[1], "health": _health(request)}
    # A consulta roda em thread: o dialeto do libsql e sincrono, e segurar o
    # laco de eventos aqui atrasaria toda requisicao de modelo em andamento.
    snapshot = await asyncio.to_thread(queries.snapshot, engine, hours)
    _cache[hours] = (now, snapshot)
    return {**snapshot, "health": _health(request)}


def _iso(value: Any) -> str:
    """Datas sempre em ISO.

    O resumo vem do banco ja em `isoformat()`; o `str()` de um `datetime`
    devolve a mesma data com espaco no lugar do "T". A fita do painel ordena
    por esse texto, entao dois formatos na mesma lista a embaralham.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


@router.post("/api/stats/clear")
async def clear_stats(request: Request) -> JSONResponse:
    """Apaga o historico do painel.

    Exige `{"confirm": true}` no corpo. Nao e ceremonia: a rota apaga sem
    volta, e um POST disparado por engano -- uma aba antiga, um `curl` de
    historico -- nao pode levar o registro embora. `older_than_hours` guarda o
    recorte recente para quem so quer zerar o passado.
    """
    engine = _engine(request)
    if engine is None:
        return JSONResponse(
            status_code=409,
            content={"error": "sem banco configurado: nao ha historico para apagar"},
        )
    try:
        body = await request.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict) or body.get("confirm") is not True:
        return JSONResponse(
            status_code=400,
            content={"error": 'envie {"confirm": true} para apagar o historico'},
        )
    raw = body.get("older_than_hours")
    older = _window(str(raw)) if raw is not None else None
    deleted = await asyncio.to_thread(queries.delete_events, engine, older)
    _cache.clear()
    return JSONResponse({"deleted": deleted, "older_than_hours": older})


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=_iso)}\n\n".encode()


@router.get("/api/stats/stream")
async def stats_stream(request: Request) -> StreamingResponse:
    recorder = getattr(request.app.state, "recorder", None)
    if recorder is None:
        return StreamingResponse(iter([b": sem gravador\n\n"]), media_type="text/event-stream")

    queue = recorder.subscribe()

    async def events():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
                except TimeoutError:
                    yield b": keepalive\n\n"
                    continue
                yield _sse(event)
        finally:
            # O navegador que fecha a aba nao pode deixar uma fila crescendo
            # no processo para sempre.
            recorder.unsubscribe(queue)

    return StreamingResponse(events(), media_type="text/event-stream")
