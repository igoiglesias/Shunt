"""A superficie inteira que o proxy expoe: as duas familias de rota.

Tres decisoes moram aqui e em nenhum outro lugar:

- **O `endpoint` e do CLIENTE, o path e do PROVEDOR.** Cada rota nomeia seu
  endpoint logico (`messages`, `chat`, `completions`, `embeddings`) e o
  dispatcher escolhe o path pelo par (protocolo do provedor, endpoint). Um
  candidato cujo protocolo nao tem path para aquele endpoint e pulado --
  a API Anthropic nao tem equivalente de `/completions` nem de `/embeddings`.
- **O envelope de erro segue o protocolo de quem perguntou** (`error_body`),
  e nao mais sempre a forma Anthropic, que era o certo por acidente enquanto
  `/v1/messages` era a unica rota.
- **A resolucao acontece antes de decidir o tipo da resposta.** Se ela ficasse
  para dentro de `dispatch_stream`, que e um gerador assincrono, o
  `UnknownProviderError` so estouraria na primeira iteracao -- depois do 200 e
  do cabecalho `text/event-stream` ja terem ido para a rede, onde nao existe
  mais status a informar.

Streaming so existe em `/v1/messages` e `/v1/chat/completions`. `/v1/completions`
e `/v1/embeddings` respondem um unico documento JSON neste proxy.
"""

import time
import uuid
from collections.abc import Mapping
from json import JSONDecodeError

import anyio
import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError
from starlette.types import Receive, Scope, Send

from app.config.settings import Settings
from app.core.dispatcher import (
    ROUTES,
    ShuntRequest,
    _provider_config,
    dispatch,
    dispatch_stream,
    error_body,
    outbound_headers,
)
from app.core.observability import RequestLog, log_request
from app.core.resolver import UnknownProviderError, resolve
from app.core.token_auth import require_shunt_token
from app.core.tokens import estimate_input_tokens
from app.schemas.anthropic import (
    AnthropicErrorResponse,
    AnthropicModel,
    AnthropicModelList,
    AnthropicRequest,
    CountTokensRequest,
    CountTokensResponse,
)
from app.schemas.openai import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    OpenAIErrorResponse,
    OpenAIModel,
    OpenAIModelList,
)

# Teto do fechamento do gerador do stream quando o pedido termina. O
# fechamento e blindado contra cancelamento (e ele que devolve o slot do
# provedor), mas blindado sem prazo prende o pedido para sempre se o provedor
# nunca terminar de fechar a conexao. Passado o teto, desiste: o cancelamento
# atravessa os `finally` do dispatcher e o slot volta mesmo assim.
STREAM_CLOSE_BOUND = 5.0


class ClosingStreamingResponse(StreamingResponse):
    """`StreamingResponse` que SEMPRE fecha o gerador do corpo ao terminar.

    Medido com o Starlette 1.6: quando o cliente desconecta com o gerador
    parado num `yield`, nem o caminho ASGI 2.0 (cancelamento no `send`) nem o
    2.4 (`OSError` -> `ClientDisconnect`) chamam `aclose()` no gerador. O
    `finally` do dispatcher -- que fecha a resposta upstream e devolve o slot
    de concorrencia do provedor -- ficava esperando o finalizador do gerador,
    e o limite do provedor ficava comido ate la.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                # Blindado: o cancelamento de um escopo anyio em volta do
                # pedido (o mecanismo do Starlette) nao interrompe justamente o
                # fechamento que libera o slot. Medido: um `task.cancel()`
                # nativo do asyncio atravessa esta blindagem (so acontece num
                # desligamento forcado). Com teto: ver STREAM_CLOSE_BOUND.
                with anyio.move_on_after(STREAM_CLOSE_BOUND, shield=True):
                    await aclose()


# A dependencia exige o token em toda a familia `/v1`: o envelope do 401/400/
# 503 e decidido pelo handler de `TokenRejected` em `app.main`, pela rota.
router = APIRouter(prefix="/v1", dependencies=[Depends(require_shunt_token)])


class BadBody(ValueError):
    """Corpo de requisicao que nao parseia, ou que parseia para nao-objeto."""

# Claude Code se anuncia como `claude-cli`; os SDKs oficiais, como
# `anthropic-sdk-<linguagem>`.
ANTHROPIC_AGENTS = ("claude-cli", "anthropic")

# Um instante fixo, nao o relogio: o campo existe porque os dois parsers o
# esperam, e um valor que muda a cada chamada faria clientes com cache
# acharem que o catalogo mudou.
MODEL_CREATED = 1700000000
MODEL_CREATED_AT = "2026-01-01T00:00:00Z"


def detect_protocol(headers: Mapping[str, str]) -> str:
    """Em que dialeto este chamador espera a resposta de `/v1/models`?

    A ordem importa: uma assinatura Claude Code autentica com bearer OAuth e
    sem `x-api-key`, entao o agente e consultado ANTES do `authorization` --
    ler aquele bearer como cliente OpenAI responderia o dialeto errado
    justamente ao cliente para quem este proxy existe.
    """
    lower = {k.lower(): v for k, v in headers.items()}
    if "anthropic-version" in lower or "x-api-key" in lower:
        return "anthropic"
    if any(name in lower.get("user-agent", "").lower() for name in ANTHROPIC_AGENTS):
        return "anthropic"
    if "authorization" in lower:
        return "openai"
    return "unknown"


# Que envelope usa o 401/400/503 do token, pela rota que o cliente chamou.
def protocol_of(path: str, headers: Mapping[str, str]) -> str:
    if path.startswith("/v1/messages"):
        return "anthropic"
    if path.startswith("/v1/models"):
        found = detect_protocol(headers)
        return "openai" if found == "unknown" else found
    return "openai"



async def _json_object(request: Request) -> dict:
    """O corpo parseado, ou `BadBody` com a mensagem que descreve a falha.

    Sao duas faltas diferentes e o cliente precisa distinguir: um corpo que
    nao parseia (virgula sobrando, chave nao fechada) e um corpo que parseia
    para algo que nao e objeto (`[1,2]`, `"oi"`, `42`, `null`). Sem esta
    guarda as duas subiam como excecao e viravam 500 -- medido em 25 de 25
    combinacoes de rota e corpo. Nenhuma das duas e falha do servidor.
    """
    try:
        body = await request.json()
    except JSONDecodeError as err:
        raise BadBody(f"o corpo da requisicao nao e JSON valido: {err}") from err
    if not isinstance(body, dict):
        raise BadBody(
            f"o corpo da requisicao precisa ser um objeto JSON, e nao {type(body).__name__}"
        )
    return body


def _validated(body: dict, schema: type[BaseModel]) -> dict:
    """O corpo conferido contra o schema do dialeto, devolvido como dict.

    `exclude_unset=True` e o que mantem o repasse fiel: sem ele todo default
    declarado no schema entraria no corpo que sobe, e o provedor veria
    parametro que o cliente nunca pediu -- `max_tokens` inventado, `stream`
    falso onde nao havia `stream` nenhum.
    """
    try:
        return schema(**body).model_dump(exclude_unset=True)
    except ValidationError as err:
        first = err.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or "corpo"
        raise BadBody(f"{field}: {first['msg']}") from err


# Cada rota tem o schema do seu dialeto e do seu endpoint. A validacao aqui
# nao estreita o que sobe: `extra="allow"` guarda o campo desconhecido e
# `model_dump(exclude_unset=True)` devolve exatamente o que o cliente mandou.
REQUEST_SCHEMAS: dict[tuple[str, str], type[BaseModel]] = {
    ("anthropic", "messages"): AnthropicRequest,
    ("openai", "chat"): ChatCompletionRequest,
    ("openai", "completions"): CompletionRequest,
    ("openai", "embeddings"): EmbeddingRequest,
}


# --- Documentacao OpenAPI -------------------------------------------------
# As rotas POST leem o corpo na mao (`_json_object`) para que um corpo
# invalido vire 400 no envelope do dialeto, e nao o 422 do FastAPI. O custo
# e que o FastAPI nao enxerga corpo nenhum para documentar. O que segue so
# DESCREVE a superficie: nada aqui e lido no caminho da requisicao.

# Modelos que as rotas citam por `$ref` sem que o FastAPI os veja como
# parametro. `app.main` registra cada um em `components/schemas`.
DOCUMENTED_MODELS: tuple[type[BaseModel], ...] = (
    AnthropicRequest,
    CountTokensRequest,
    CountTokensResponse,
    AnthropicErrorResponse,
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    OpenAIErrorResponse,
    AnthropicModel,
    AnthropicModelList,
    OpenAIModel,
    OpenAIModelList,
)


def _ref(model: type[BaseModel]) -> dict:
    return {"$ref": f"#/components/schemas/{model.__name__}"}


# `PROXY_SECURITY` documenta, para o OpenAPI, os veiculos que um token valido
# do Shunt pode chegar em: o cabecalho `x-shunt-token`, o prefixo `/t/<token>/`
# e o `?token=`. A execucao nao depende desta lista: `require_shunt_token`
# (dependencia do router) exige um token valido em toda a familia `/v1` e
# recusa com 401 quando nao ha token em nenhum veiculo.
PROXY_SECURITY: list[dict[str, list[str]]] = [
    {"anthropicApiKey": []},
    {"bearerAuth": []},
    {"shuntToken": []},
    {},
]
# `count_tokens` tambem exige o token Shunt: a dependencia do router cobre a
# rota, e `shuntToken` e uma das formas declaradas aqui.
COUNT_TOKENS_SECURITY: list[dict[str, list[str]]] = [
    {"anthropicApiKey": []},
    {"bearerAuth": []},
    {"shuntToken": []},
    {},
]

_DETAIL = {
    "type": "object",
    "properties": {"detail": {"type": "string"}},
    "required": ["detail"],
}


def _request_body(model: type[BaseModel]) -> dict:
    return {
        "required": True,
        "content": {"application/json": {"schema": _ref(model)}},
    }


def _proxy_responses(error: type[BaseModel], streaming: bool) -> dict:
    ok_content: dict = {"application/json": {"schema": {"type": "object"}}}
    ok = (
        "The provider's answer, translated into this route's dialect."
    )
    if streaming:
        ok_content["text/event-stream"] = {"schema": {"type": "string"}}
        ok += (
            " With `\"stream\": true` in the body, the answer arrives as "
            "server-sent events in the same dialect. In that mode a failure "
            "(including \"no candidate can serve this request\") arrives inside the "
            "stream, because the 200 has already been sent: an Anthropic stream gets "
            "an `event: error` with the Anthropic error envelope, an OpenAI stream "
            "gets a `data:` chunk carrying the OpenAI error envelope."
        )
    return {
        200: {
            "description": ok,
            "content": ok_content,
            "headers": {
                "x-shunt-model": {
                    "description": (
                        "The model that actually answered. The body's `model` field "
                        "still echoes the one you asked for. Sent on JSON answers only, "
                        "not on streams."
                    ),
                    "schema": {"type": "string"},
                }
            },
        },
        400: {
            "description": (
                "The body is not a JSON object, fails validation, names a model no "
                "route resolves, or no candidate can serve this request (every "
                "candidate lacks a capability it needs). A provider's own 400 is also "
                "passed through. The error envelope follows this route's dialect."
            ),
            "content": {"application/json": {"schema": _ref(error)}},
        },
        401: {
            "description": (
                "Either the `x-shunt-token` is unknown or expired (body `detail`), or "
                "the provider rejected the credential and its 401 is passed through "
                "in this route's error envelope."
            ),
            "content": {"application/json": {"schema": {"anyOf": [_DETAIL, _ref(error)]}}},
        },
        502: {
            "description": (
                "No candidate returned a usable answer. The message lists every "
                "candidate tried and why it failed. When the last candidate answered "
                "with an error of its own, that status is returned instead."
            ),
            "content": {"application/json": {"schema": _ref(error)}},
        },
        503: {
            "description": (
                "Either an `x-shunt-token` was sent but Shunt has no database to check "
                "it (body `detail`), or the provider answered 503 and it is passed "
                "through in this route's error envelope."
            ),
            "content": {"application/json": {"schema": {"anyOf": [_DETAIL, _ref(error)]}}},
        },
        "default": {
            "description": (
                "Any other error status the last provider answered (403, 429, 500...), "
                "passed through in this route's error envelope."
            ),
            "content": {"application/json": {"schema": _ref(error)}},
        },
    }


def _proxy_route(
    model: type[BaseModel], error: type[BaseModel], streaming: bool, tag: str
) -> dict:
    return {
        "tags": [tag],
        "responses": _proxy_responses(error, streaming),
        "openapi_extra": {"requestBody": _request_body(model), "security": PROXY_SECURITY},
    }


_ROUTING = (
    " Shunt resolves `model` against its routes and tries each candidate in order, "
    "translating between dialects when the provider speaks the other one."
)


def _record(
    route: str,
    dialect: str,
    status: int,
    started: float,
    *,
    requested_model: str = "",
    rule: str = "none",
    matched: str | None = None,
    candidate: str | None = None,
    provider: str | None = None,
    error_type: str | None = None,
    input_tokens: int = 0,
) -> None:
    """A linha de log e o evento do painel para o que o dispatcher nao ve.

    Tres casos caem aqui: `count_tokens`, a listagem de modelos, e toda recusa
    que acontece ANTES de existir candidato -- corpo malformado, modelo sem
    provedor. Requisicao recusada e justamente a que se quer contar: sem ela o
    painel mostraria um proxy que nunca erra.
    """
    log_request(
        RequestLog(
            request_id=uuid.uuid4().hex,
            requested_model=requested_model,
            rule=rule,
            matched=matched,
            candidate=candidate,
            input_tokens=input_tokens,
            duration_ms=int((time.monotonic() - started) * 1000),
            route=route,
            dialect=dialect,
            status=status,
            error_type=error_type,
            provider=provider,
        )
    )


async def _serve(request: Request, protocol: str, endpoint: str, streaming: bool):
    started = time.monotonic()
    route = ROUTES.get(endpoint, endpoint)
    try:
        body = _validated(await _json_object(request), REQUEST_SCHEMAS[(protocol, endpoint)])
    except BadBody as err:
        _record(route, protocol, 400, started, error_type="invalid_request_error")
        return JSONResponse(status_code=400, content=error_body(protocol, 400, str(err)))
    settings, pool = request.app.state.settings, request.app.state.pool
    # A validacao do token e da dependencia `require_shunt_token` (rodou antes
    # deste handler): `credential_is_token` diz se a credencial apresentada
    # FOI um token do Shunt, e entao a chave configurada do provedor o
    # substitui. O campo `shunt_token` do dataclass foi removido (Task 1.5).
    shunt_request = ShuntRequest(
        protocol,
        body,
        dict(request.headers),
        endpoint=endpoint,
        credential_is_token=request.state.credential_is_token,
    )
    try:
        resolve(body.get("model", ""), settings)
    except UnknownProviderError as err:
        _record(
            route,
            protocol,
            400,
            started,
            requested_model=body.get("model", ""),
            error_type="invalid_request_error",
        )
        return JSONResponse(status_code=400, content=error_body(protocol, 400, str(err)))
    if streaming and body.get("stream"):
        return ClosingStreamingResponse(
            dispatch_stream(shunt_request, settings, pool), media_type="text/event-stream"
        )
    result = await dispatch(shunt_request, settings, pool)
    # `x-shunt-model` names the model that actually ran; the response body's
    # own `model` field still echoes what the client asked for. When no
    # candidate ran at all, `real_model` is None and the header is omitted
    # rather than sent empty.
    headers = {"x-shunt-model": result.real_model} if result.real_model else {}
    return JSONResponse(status_code=result.status, content=result.body, headers=headers)


@router.post(
    "/messages",
    summary="Create a message (Anthropic Messages API)",
    description="Anthropic Messages request, answered in Anthropic's shape." + _ROUTING,
    **_proxy_route(AnthropicRequest, AnthropicErrorResponse, True, "Anthropic"),
)
async def create_message(request: Request):
    return await _serve(request, "anthropic", "messages", streaming=True)


@router.post(
    "/chat/completions",
    summary="Create a chat completion (OpenAI)",
    description="OpenAI Chat Completions request, answered in OpenAI's shape." + _ROUTING,
    **_proxy_route(ChatCompletionRequest, OpenAIErrorResponse, True, "OpenAI"),
)
async def create_chat_completion(request: Request):
    return await _serve(request, "openai", "chat", streaming=True)


@router.post(
    "/completions",
    summary="Create a text completion (OpenAI legacy)",
    description=(
        "OpenAI legacy Completions request. Always answered as one JSON document."
        + _ROUTING
        + " Anthropic providers are skipped: their API has no equivalent."
    ),
    **_proxy_route(CompletionRequest, OpenAIErrorResponse, False, "OpenAI"),
)
async def create_completion(request: Request):
    return await _serve(request, "openai", "completions", streaming=False)


@router.post(
    "/embeddings",
    summary="Create embeddings (OpenAI)",
    description=(
        "OpenAI Embeddings request, answered as one JSON document."
        + _ROUTING
        + " Anthropic providers are skipped: their API has no equivalent."
    ),
    **_proxy_route(EmbeddingRequest, OpenAIErrorResponse, False, "OpenAI"),
)
async def create_embeddings(request: Request):
    return await _serve(request, "openai", "embeddings", streaming=False)


@router.post(
    "/messages/count_tokens",
    summary="Count input tokens (Anthropic)",
    description=(
        "Counts the input tokens of an Anthropic Messages request. When the first "
        "candidate for `model` speaks Anthropic, its own count is returned; otherwise, "
        "or when it fails, Shunt returns a local estimate. Requires a valid Shunt "
        "token (401 without one), like every other `/v1` route."
    ),
    tags=["Anthropic"],
    responses={
        200: {
            "description": "The provider's count, or Shunt's local estimate.",
            "content": {"application/json": {"schema": _ref(CountTokensResponse)}},
        },
        400: {
            "description": (
                "The body is not a JSON object, fails validation, or names a model "
                "no route resolves."
            ),
            "content": {"application/json": {"schema": _ref(AnthropicErrorResponse)}},
        },
    },
    openapi_extra={
        "requestBody": _request_body(CountTokensRequest),
        "security": COUNT_TOKENS_SECURITY,
    },
)
async def count_tokens(request: Request):
    """Encaminha quando da, estima quando nao da.

    A API OpenAI nao tem equivalente deste endpoint. Se o candidato resolvido
    fala Anthropic, a contagem dele e a autoritativa e vai encaminhada; caso
    contrario -- e tambem quando o destino responde qualquer coisa diferente de
    200, como um gateway que se declara Anthropic mas nao implementa a rota --
    a estimativa local e uma resposta util, e um erro nao seria.
    """
    started = time.monotonic()
    route = "/v1/messages/count_tokens"
    try:
        body = _validated(await _json_object(request), CountTokensRequest)
    except BadBody as err:
        _record(route, "anthropic", 400, started, error_type="invalid_request_error")
        return JSONResponse(status_code=400, content=error_body("anthropic", 400, str(err)))
    settings, pool = request.app.state.settings, request.app.state.pool
    try:
        resolution = resolve(body.get("model", ""), settings)
        candidate = resolution.chain[0]
    except UnknownProviderError as err:
        _record(
            route,
            "anthropic",
            400,
            started,
            requested_model=body.get("model", ""),
            error_type="invalid_request_error",
        )
        return JSONResponse(status_code=400, content=error_body("anthropic", 400, str(err)))
    if candidate.protocol == "anthropic":
        # Mesma guarda de `_serve`: o valor do token nunca e credencial que sai.
        shunt_request = ShuntRequest(
            "anthropic",
            body,
            dict(request.headers),
            endpoint="messages",
            credential_is_token=request.state.credential_is_token,
        )
        try:
            async with pool.client(
                candidate.provider, _provider_config(candidate, settings)
            ) as upstream_client:
                upstream = await upstream_client.post(
                    "/v1/messages/count_tokens",
                    json={**body, "model": candidate.model},
                    headers=outbound_headers(shunt_request, candidate, settings),
                )
        except httpx.HTTPError:
            # Conexao que morre e a mesma situacao de um status diferente de
            # 200: a contagem autoritativa nao veio, e a estimativa local e uma
            # resposta util. Sem este ramo a excecao sobe pela pilha do ASGI e
            # o cliente recebe um 500 de `text/plain`, que nem o envelope de
            # erro do protocolo respeita.
            pass
        else:
            if upstream.status_code == 200:
                counted = upstream.json()
                _record(
                    route,
                    "anthropic",
                    200,
                    started,
                    requested_model=body.get("model", ""),
                    rule=resolution.rule,
                    matched=resolution.matched,
                    candidate=candidate.model,
                    provider=candidate.provider,
                    input_tokens=counted.get("input_tokens", 0),
                )
                return counted
    estimated = {"input_tokens": estimate_input_tokens(body)}
    # Estimativa local conta como resposta desta rota, e nao como falha: o
    # candidato fica registrado como nulo porque nenhum provedor respondeu.
    _record(
        route,
        "anthropic",
        200,
        started,
        requested_model=body.get("model", ""),
        rule=resolution.rule,
        matched=resolution.matched,
        input_tokens=estimated["input_tokens"],
    )
    return estimated


_DIALECT = (
    " The answer's shape follows the caller: an `anthropic-version` or `x-api-key` "
    "header, or a Claude Code / Anthropic SDK User-Agent, gets Anthropic's shape; "
    "an `Authorization` header without those gets OpenAI's; no signal gets a superset "
    "carrying the fields of both. No credential is required."
)


def _model_entries(settings: Settings) -> list[dict]:
    """A uniao dos dois formatos. Cada dialeto e uma projecao deste registro."""
    return [
        {
            "id": alias,
            "object": "model",
            "created": MODEL_CREATED,
            "owned_by": model.provider,
            "type": "model",
            "display_name": f"{alias} ({model.model})",
            "created_at": MODEL_CREATED_AT,
        }
        for alias, model in settings.models.items()
    ]


@router.get(
    "/models/{model_id}",
    summary="Get one model",
    description="One configured model alias." + _DIALECT,
    tags=["Models"],
    responses={
        200: {
            "description": "The model, without the list wrapper.",
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [_ref(AnthropicModel), _ref(OpenAIModel), {"type": "object"}]
                    }
                }
            },
        },
        404: {
            "description": (
                "No model is configured under this alias. The error envelope follows "
                "the caller's dialect."
            ),
            "content": {
                "application/json": {
                    "schema": {"anyOf": [_ref(AnthropicErrorResponse), _ref(OpenAIErrorResponse)]}
                }
            },
        },
    },
)
async def get_model(model_id: str, request: Request):
    """Um modelo pelo alias, no dialeto de quem perguntou.

    A projecao e a mesma da listagem, menos o wrapper de lista: cada parser
    le o documento solto sem a envoltura `data`. Sem o alias a resposta e 404
    no envelope de quem perguntou -- nao o `detail` genrico do framework, que
    nenhum SDK dos dois protocolos sabe classificar.
    """
    started = time.monotonic()
    settings = request.app.state.settings
    protocol = detect_protocol(request.headers)
    entries = _model_entries(settings)
    entry = next((e for e in entries if e["id"] == model_id), None)
    if entry is None:
        _record("/v1/models", protocol, 404, started, requested_model=model_id)
        return JSONResponse(
            status_code=404,
            content=error_body(protocol, 404, f"modelo nao encontrado: {model_id}"),
        )
    _record("/v1/models", protocol, 200, started, requested_model=model_id)
    if protocol == "anthropic":
        return AnthropicModel(
            **{k: entry[k] for k in ("id", "display_name", "created_at")}
        ).model_dump()
    if protocol == "openai":
        return OpenAIModel(**{k: entry[k] for k in ("id", "created", "owned_by")}).model_dump()
    return entry


@router.get(
    "/models",
    summary="List models",
    description="Every configured model alias." + _DIALECT,
    tags=["Models"],
    responses={
        200: {
            "description": "The catalog in the caller's dialect.",
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [
                            _ref(AnthropicModelList),
                            _ref(OpenAIModelList),
                            {"type": "object"},
                        ]
                    }
                }
            },
        },
    },
)
async def list_models(request: Request):
    """O catalogo no dialeto de quem perguntou.

    Sem sinal nenhum nos cabecalhos a resposta e o superconjunto: as chaves dos
    dois formatos no mesmo documento. Cada parser le as suas e ignora as
    outras, o que e melhor do que apostar no dialeto errado.
    """
    started = time.monotonic()
    entries = _model_entries(request.app.state.settings)
    # Uma instalacao sem nenhum modelo configurado: `entries[0]` estouraria, e
    # `first_id: null` e o que os dois parsers leem como "nao ha nada aqui".
    # Os dois dialetos resolvem isso dentro do proprio schema; o superconjunto
    # e uniao de formatos e nao tem schema, entao calcula aqui.
    first = entries[0]["id"] if entries else None
    protocol = detect_protocol(request.headers)
    _record("/v1/models", protocol, 200, started)
    if protocol == "anthropic":
        return AnthropicModelList.of(entries).model_dump()
    if protocol == "openai":
        return OpenAIModelList.of(entries).model_dump()
    return {"object": "list", "data": entries, "has_more": False, "first_id": first}
