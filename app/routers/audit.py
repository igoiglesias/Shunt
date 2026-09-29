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
from typing import Annotated

from fastapi import APIRouter, Path, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config.config import EXPORT_LIMIT, MAX_SEARCH_LIMIT, SEARCH_LIMIT
from app.routers._openapi_docs import ADMIN_SESSION as _SESSION
from app.routers._openapi_docs import NO_DATABASE as _NO_DATABASE
from app.routers._openapi_docs import UNAUTHORIZED as _UNAUTHORIZED
from app.stats import analysis, queries

router = APIRouter()

# Colunas da exportacao, na ordem em que uma planilha faz sentido.
CSV_COLUMNS = (
    "started_at",
    "request_id",
    "kind",
    "route",
    "dialect",
    "stream",
    "requested_model",
    "provider",
    "candidate_model",
    "project",
    "status",
    "error_type",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "ttft_ms",
    "duration_ms",
    "fell_back",
    "attempts",
    "tools_offered",
    "tools_called",
)


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


# Os dois unicos tipos de trafego que uma linha pode ser (ver
# `RequestEvent.kind`, em `app/stats/models.py`). O /docs publica a lista, e a
# busca recusa outro nome com `ValueError` (`_search_clauses`): um valor que nao
# e nenhum dos dois e um filtro que veio de outra tela, e nao um recorte pedido.
KINDS = ("model", "relay")


def _kind(params) -> str | None:
    """O tipo de trafego, ou None quando o valor nao e um dos dois nominais.

    Diferente de `_text`: um nome que nao existe nao e "filtro vazio", e o
    contrato de "quem nao parseia e ignorado" faz a busca devolver tudo -- o
    leitor prefere ver o resultado mais largo a ser trancado por um nome errado.
    """
    value = params.get("kind")
    return value if value in KINDS else None


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
        "project": _text(params, "project"),
        "status_min": _number(params, "status_min"),
        "status_max": _number(params, "status_max"),
        "stream": _flag(params, "stream"),
        "fell_back": _flag(params, "fell_back"),
        "has_tools": _flag(params, "has_tools"),
        "min_duration_ms": _number(params, "min_duration_ms"),
        "min_tokens": _number(params, "min_tokens"),
        "kind": _kind(params),
        "order_by": "duration" if params.get("order_by") == "duration" else "time",
    }


# --- Contrato publicado no /docs ----------------------------------------------
#
# Os handlers leem `request.query_params` na mao, e isso e o contrato: filtro
# que nao parseia vira filtro ausente, e nao 422. Trocar por `Query(...)` mudaria
# essa resposta; por isso o esquema vai em `openapi_extra`, que so mexe no
# documento. A lista abaixo espelha `filters_from` -- o teste em
# `tests/test_openapi_api.py` compara as duas.

_REQUEST_ID = Annotated[str, Path(description="The `request_id` of a recorded request.")]
_ANALYSIS_ID = Annotated[int, Path(description="The `id` of a stored analysis.")]


def _query(name: str, description: str, schema: dict) -> dict:
    return {
        "name": name,
        "in": "query",
        "required": False,
        "description": description,
        "schema": schema,
    }


_EXACT = "Exact match on the {}. Empty means no filter."
_FLAG = (
    "Tri-state: `1`, `true`, `yes`, `on` or `sim` keeps only matches; any other value "
    "keeps only non-matches; absent or empty means no filter. {}"
)
_IGNORED = " An unparsable value is ignored, not refused."

_ANALYSIS_FILTER_PARAMS = [
    _query(
        "since",
        "Start of the period, inclusive, ISO 8601. Without an offset it is read as UTC." + _IGNORED,
        {"type": "string", "format": "date-time"},
    ),
    _query(
        "until",
        "End of the period, inclusive, ISO 8601. Without an offset it is read as UTC." + _IGNORED,
        {"type": "string", "format": "date-time"},
    ),
    _query(
        "text",
        "Substring searched in the request id, requested and candidate model, provider, "
        "route, error type, matched rule, project, tools offered and called, and the "
        "attempt chain.",
        {"type": "string"},
    ),
    _query("route", _EXACT.format("proxy route, such as `/v1/messages`"), {"type": "string"}),
    _query("dialect", _EXACT.format("client protocol"), {"type": "string"}),
    _query("provider", _EXACT.format("provider that answered"), {"type": "string"}),
    _query("candidate_model", _EXACT.format("model that answered"), {"type": "string"}),
    _query("requested_model", _EXACT.format("model the client asked for"), {"type": "string"}),
    _query("error_type", _EXACT.format("error type"), {"type": "string"}),
    _query("project", _EXACT.format("project"), {"type": "string"}),
    _query("status_min", "Lowest HTTP status, inclusive." + _IGNORED, {"type": "integer"}),
    _query("status_max", "Highest HTTP status, inclusive." + _IGNORED, {"type": "integer"}),
    _query("stream", _FLAG.format("Matches streamed requests."), {"type": "boolean"}),
    _query("fell_back", _FLAG.format("Matches requests that fell back."), {"type": "boolean"}),
    _query("has_tools", _FLAG.format("Matches requests that called a tool."), {"type": "boolean"}),
    _query(
        "min_duration_ms", "Minimum total duration in milliseconds." + _IGNORED, {"type": "integer"}
    ),
    _query("min_tokens", "Minimum input plus output tokens." + _IGNORED, {"type": "integer"}),
]
_ORDER_PARAM = _query(
    "order_by",
    "`duration` lists the slowest first; anything else lists the newest first.",
    {"type": "string", "enum": ["time", "duration"], "default": "time"},
)
# So a busca documenta: e um filtro que a listagem de fato aplica. O dossie nao
# (`_analysis_filters` o retira), e por isso ele fica fora de
# `_ANALYSIS_FILTER_PARAMS` -- publicar ali seria prometer um recorte que a
# analise sempre recusa em favor do trafego de modelo.
_KIND_PARAM = _query(
    "kind",
    "Kind of traffic: `model` for a request routed through the provider chain, "
    "`relay` for a route the proxy does not implement, passed through verbatim to "
    "the official host. Any other value is ignored, not refused.",
    {"type": "string", "enum": list(KINDS)},
)
_FILTER_PARAMS = [*_ANALYSIS_FILTER_PARAMS, _KIND_PARAM, _ORDER_PARAM]
# O corpo que `analysis.analyse` monta quando a chamada ao modelo falha.
_ANALYSIS_FAILURE = {
    "type": "object",
    "properties": {
        "status": {"type": "integer", "description": "The model's HTTP status, or 502."},
        "error": {"type": "string", "description": "The error message from the model."},
        "text": {"type": "string", "description": "Always empty on failure."},
        "cached": {"type": "boolean", "description": "Always false on failure."},
        "dossier": {"type": "object", "description": "The period summary that was sent."},
        "trace": {
            "type": "array",
            "items": {},
            "description": "The attempts made through the proxy.",
        },
    },
}
# Os outros dois corpos que 401 e 409 podem ter nesta rota: o do
# `HTTPException` de `require_admin` e o de `_no_database`.
_SESSION_ERROR = {
    "type": "object",
    "required": ["detail"],
    "properties": {"detail": {"type": "string"}},
}
_NO_DATABASE_ERROR = {
    "type": "object",
    "required": ["error"],
    "properties": {"error": {"type": "string"}},
}
_PAGE_PARAMS = [
    _query(
        "limit",
        # Sem `minimum`/`maximum`: o servidor prende o valor na faixa, nao recusa.
        "Page size. 0, absent or unparsable uses the default (SHUNT_SEARCH_LIMIT); "
        f"other values are clamped between 1 and {MAX_SEARCH_LIMIT} "
        "(SHUNT_MAX_SEARCH_LIMIT), not refused.",
        {"type": "integer", "default": SEARCH_LIMIT},
    ),
    _query(
        "cursor",
        "Opaque `next_cursor` from the previous page. An invalid cursor restarts from "
        "the first page.",
        {"type": "string"},
    ),
]


def _json(description: str) -> dict:
    return {"description": description, "content": {"application/json": {}}}


def _no_database() -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={"error": "sem banco configurado: nao ha historico para auditar"},
    )


@router.get(
    "/api/requests",
    tags=["Requests"],
    summary="Search recorded requests",
    description=(
        "One page of recorded requests matching every filter, newest first unless "
        "`order_by=duration`. `total` counts the whole search, not the page; pass "
        "`next_cursor` back as `cursor` for the next page. Prompts and responses are "
        "never part of a record."
    ),
    responses={
        "200": _json("A page of events, the search total and the next cursor."),
        **_UNAUTHORIZED,
        **_NO_DATABASE,
    },
    openapi_extra={"security": _SESSION, "parameters": [*_FILTER_PARAMS, *_PAGE_PARAMS]},
)
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


@router.get(
    "/api/requests/export",
    tags=["Requests"],
    summary="Export a search as CSV",
    description=(
        "The same filters as the search, as a CSV attachment of at most "
        "SHUNT_EXPORT_LIMIT rows. `x-shunt-exported`, `x-shunt-total` and "
        "`x-shunt-truncated` say whether the file holds the whole search."
    ),
    response_class=StreamingResponse,
    responses={
        "200": {
            "description": "CSV file, one row per request.",
            "content": {"text/csv": {"schema": {"type": "string"}}},
            "headers": {
                "x-shunt-exported": {
                    "description": "Rows in the file.",
                    "schema": {"type": "integer"},
                },
                "x-shunt-total": {
                    "description": "Rows in the whole search.",
                    "schema": {"type": "integer"},
                },
                "x-shunt-truncated": {
                    "description": "`1` when the file holds fewer rows than the search.",
                    "schema": {"type": "string", "enum": ["0", "1"]},
                },
            },
        },
        **_UNAUTHORIZED,
        **_NO_DATABASE,
    },
    openapi_extra={"security": _SESSION, "parameters": _FILTER_PARAMS},
)
async def export_requests(request: Request):
    """A mesma busca em CSV, para levar a investigacao para outro lugar."""
    engine = _engine(request)
    if engine is None:
        return _no_database()
    filters = filters_from(request.query_params)
    page = await asyncio.to_thread(queries.search_events, engine, limit=EXPORT_LIMIT, **filters)
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


@router.get(
    "/api/requests/facets",
    tags=["Requests"],
    summary="Filter values that exist",
    description=(
        "Distinct projects, providers, models and routes seen in the last 30 days, "
        "each with its request count, to build the search filters."
    ),
    responses={
        "200": _json("Lists of `{value, count}` per field."),
        **_UNAUTHORIZED,
        **_NO_DATABASE,
    },
    openapi_extra={"security": _SESSION},
)
async def facets(request: Request):
    """Os valores que existem, para os seletores da tela."""
    engine = _engine(request)
    if engine is None:
        return _no_database()
    return JSONResponse(await asyncio.to_thread(queries.facets, engine))


@router.get(
    "/api/requests/{request_id}/body",
    tags=["Requests"],
    summary="Conversation text of one request",
    description=(
        "The recorded conversation of a request. Returns 404 when none was recorded, "
        "for instance because conversation capture was off when it ran."
    ),
    responses={
        "200": _json("The recorded conversation."),
        "404": {"description": "No conversation was recorded for this request."},
        **_UNAUTHORIZED,
        **_NO_DATABASE,
    },
    openapi_extra={"security": _SESSION},
)
async def request_body(request: Request, request_id: _REQUEST_ID):
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


@router.get(
    "/api/requests/{request_id}",
    tags=["Requests"],
    summary="One recorded request",
    description="A single request with its attempt chain, in the order the attempts ran.",
    responses={
        "200": _json("The request record."),
        "404": {"description": "No request with this id."},
        **_UNAUTHORIZED,
        **_NO_DATABASE,
    },
    openapi_extra={"security": _SESSION},
)
async def one_request(request: Request, request_id: _REQUEST_ID):
    engine = _engine(request)
    if engine is None:
        return _no_database()
    event = await asyncio.to_thread(queries.event_detail, engine, request_id)
    if event is None:
        return JSONResponse(status_code=404, content={"error": "requisicao nao encontrada"})
    return JSONResponse(event)


# --- Analise do periodo -------------------------------------------------------
#
# O botao "Analisar este periodo" herda os filtros da tela: o periodo analisado
# e o periodo que a pessoa esta olhando. Por isso os filtros vem da query, os
# MESMOS parametros da busca, e o corpo carrega so o que e da analise -- o
# modelo e o pedido explicito de refazer.


def _analysis_filters(params) -> dict:
    filters = filters_from(params)
    # `order_by` ordena uma listagem; um dossie nao tem ordem de leitura, e
    # passa-lo adiante seria um filtro que a busca entende e a analise nao.
    filters.pop("order_by", None)
    # `kind` e parecido, mas ao contrario: o dossie SEMPRE le trafego de modelo
    # (`kind="model"` fixo em `app/stats/dossier.py`), e a chave nao entra em
    # `FILTER_FIELDS`, que recusaria filtro desconhecido com `ValueError` -- ou
    # seja, repassa-la seria transformar um filtro da tela num 500 na analise.
    filters.pop("kind", None)
    return filters


async def _body_of(request: Request) -> dict:
    """O corpo JSON, ou vazio. O botao manda POST sem corpo, e isso e valido."""
    try:
        payload = await request.json()
    except ValueError:
        # Corpo vazio levanta `JSONDecodeError`, que e um `ValueError`: POST sem
        # corpo e o caminho normal do botao, e nao um erro para mostrar na tela.
        return {}
    return payload if isinstance(payload, dict) else {}


@router.post(
    "/api/analysis",
    tags=["Analysis"],
    summary="Ask a model to read a period",
    description=(
        "Builds a summary of the requests matching the query filters and sends it to a "
        "model through the proxy itself. The body is optional: `model` defaults to "
        "`default_model`, and a previous analysis of the same filters and model is "
        "returned from storage unless `refresh` is true. When the model call fails, "
        "its HTTP status is passed through (502 when it answered without text)."
    ),
    responses={
        "200": _json("The analysis text, whether it came from storage, and usage."),
        # 401 e 409 colidem: a mesma chave vale para o erro do proprio Shunt e
        # para o status do provedor que `analyse` repassa, cada um com seu corpo.
        "401": {
            "description": "No valid admin session cookie (`detail`), or the model's "
            "provider refused the key and its 401 is passed through (failure body).",
            "content": {
                "application/json": {"schema": {"anyOf": [_SESSION_ERROR, _ANALYSIS_FAILURE]}}
            },
        },
        "409": {
            "description": "No database is configured (`error` only); no model was "
            "given and there is no `default_model` (failure body without `dossier` "
            "and `trace`); or the provider answered 409, passed through.",
            "content": {
                "application/json": {"schema": {"anyOf": [_NO_DATABASE_ERROR, _ANALYSIS_FAILURE]}}
            },
        },
        "502": {
            "description": "The model answered without text.",
            "content": {"application/json": {"schema": _ANALYSIS_FAILURE}},
        },
        # `analyse` devolve o status do modelo quando ele falha: qualquer 4xx/5xx
        # que o provedor mandar, com o mesmo corpo.
        "default": {
            "description": "The model call failed; its HTTP status is passed through.",
            "content": {"application/json": {"schema": _ANALYSIS_FAILURE}},
        },
    },
    openapi_extra={
        "security": _SESSION,
        "parameters": _ANALYSIS_FILTER_PARAMS,
        "requestBody": {
            "required": False,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "model": {
                                "type": "string",
                                "description": "Model alias to use. Defaults to `default_model`.",
                            },
                            "refresh": {
                                "type": "boolean",
                                "default": False,
                                "description": "Run again instead of returning a stored analysis.",
                            },
                        },
                    }
                }
            },
        },
    },
)
async def analyse_period(request: Request):
    """Manda o periodo para um modelo e devolve a leitura dele."""
    engine = _engine(request)
    if engine is None:
        return _no_database()
    payload = await _body_of(request)
    state = request.app.state
    result = await analysis.analyse(
        engine,
        state.settings,
        getattr(state, "pool", None),
        filters=_analysis_filters(request.query_params),
        model=payload.get("model"),
        refresh=bool(payload.get("refresh")),
    )
    return JSONResponse(status_code=result["status"], content=result)


@router.get(
    "/api/analysis",
    tags=["Analysis"],
    summary="Stored analyses and available models",
    description=(
        "Recent stored analyses, plus the configured model aliases and "
        "`default_model`, so a client can choose a model before spending on one."
    ),
    responses={
        "200": _json("`analyses`, `models` and `default_model`."),
        **_UNAUTHORIZED,
        **_NO_DATABASE,
    },
    openapi_extra={"security": _SESSION},
)
async def list_analyses(request: Request):
    """As analises ja pagas, e quem pode ler o periodo.

    O catalogo vem junto porque a tela precisa oferecer a escolha ANTES de
    gastar: `default_model` e opcional, e uma instalacao sem ele so descobriria
    que nao ha para quem mandar depois de clicar. Vem daqui, e nao de
    `/v1/models`, porque aquela rota e do proxy e grava uma linha no historico a
    cada visita -- a tela ficaria poluindo o proprio periodo que analisa.
    """
    engine = _engine(request)
    if engine is None:
        return _no_database()
    items = await asyncio.to_thread(analysis.recent, engine)
    settings = request.app.state.settings
    return JSONResponse(
        {
            "analyses": items,
            "models": sorted(settings.models),
            "default_model": settings.default_model,
        }
    )


@router.get(
    "/api/analysis/{analysis_id}",
    tags=["Analysis"],
    summary="One stored analysis",
    description="A stored analysis by id, with the summary it was built from.",
    responses={
        "200": _json("The stored analysis."),
        "404": {"description": "No analysis with this id."},
        **_UNAUTHORIZED,
        **_NO_DATABASE,
    },
    openapi_extra={"security": _SESSION},
)
async def one_analysis(request: Request, analysis_id: _ANALYSIS_ID):
    engine = _engine(request)
    if engine is None:
        return _no_database()
    item = await asyncio.to_thread(analysis.one, engine, analysis_id)
    if item is None:
        return JSONResponse(status_code=404, content={"error": "analise nao encontrada"})
    return JSONResponse(item)
