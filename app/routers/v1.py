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

from collections.abc import Mapping
from json import JSONDecodeError

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError

from app.config.settings import Settings
from app.core.dispatcher import (
    ShuntRequest,
    dispatch,
    dispatch_stream,
    error_body,
    outbound_headers,
)
from app.core.resolver import UnknownProviderError, resolve
from app.core.tokens import estimate_input_tokens
from app.schemas.anthropic import (
    AnthropicModelList,
    AnthropicRequest,
    CountTokensRequest,
)
from app.schemas.openai import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    OpenAIModelList,
)

router = APIRouter(prefix="/v1")


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


async def _serve(request: Request, protocol: str, endpoint: str, streaming: bool):
    try:
        body = _validated(await _json_object(request), REQUEST_SCHEMAS[(protocol, endpoint)])
    except BadBody as err:
        return JSONResponse(status_code=400, content=error_body(protocol, 400, str(err)))
    settings, pool = request.app.state.settings, request.app.state.pool
    shunt_request = ShuntRequest(protocol, body, dict(request.headers), endpoint=endpoint)
    try:
        resolve(body.get("model", ""), settings)
    except UnknownProviderError as err:
        return JSONResponse(status_code=400, content=error_body(protocol, 400, str(err)))
    if streaming and body.get("stream"):
        return StreamingResponse(
            dispatch_stream(shunt_request, settings, pool), media_type="text/event-stream"
        )
    result = await dispatch(shunt_request, settings, pool)
    # `x-shunt-model` names the model that actually ran; the response body's
    # own `model` field still echoes what the client asked for. When no
    # candidate ran at all, `real_model` is None and the header is omitted
    # rather than sent empty.
    headers = {"x-shunt-model": result.real_model} if result.real_model else {}
    return JSONResponse(status_code=result.status, content=result.body, headers=headers)


@router.post("/messages")
async def create_message(request: Request):
    return await _serve(request, "anthropic", "messages", streaming=True)


@router.post("/chat/completions")
async def create_chat_completion(request: Request):
    return await _serve(request, "openai", "chat", streaming=True)


@router.post("/completions")
async def create_completion(request: Request):
    return await _serve(request, "openai", "completions", streaming=False)


@router.post("/embeddings")
async def create_embeddings(request: Request):
    return await _serve(request, "openai", "embeddings", streaming=False)


@router.post("/messages/count_tokens")
async def count_tokens(request: Request):
    """Encaminha quando da, estima quando nao da.

    A API OpenAI nao tem equivalente deste endpoint. Se o candidato resolvido
    fala Anthropic, a contagem dele e a autoritativa e vai encaminhada; caso
    contrario -- e tambem quando o destino responde qualquer coisa diferente de
    200, como um gateway que se declara Anthropic mas nao implementa a rota --
    a estimativa local e uma resposta util, e um erro nao seria.
    """
    try:
        body = _validated(await _json_object(request), CountTokensRequest)
    except BadBody as err:
        return JSONResponse(status_code=400, content=error_body("anthropic", 400, str(err)))
    settings, pool = request.app.state.settings, request.app.state.pool
    try:
        candidate = resolve(body.get("model", ""), settings).chain[0]
    except UnknownProviderError as err:
        return JSONResponse(status_code=400, content=error_body("anthropic", 400, str(err)))
    if candidate.protocol == "anthropic":
        shunt_request = ShuntRequest("anthropic", body, dict(request.headers), endpoint="messages")
        try:
            upstream = await pool.get(candidate.provider).post(
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
                return upstream.json()
    return {"input_tokens": estimate_input_tokens(body)}


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


@router.get("/models")
async def list_models(request: Request):
    """O catalogo no dialeto de quem perguntou.

    Sem sinal nenhum nos cabecalhos a resposta e o superconjunto: as chaves dos
    dois formatos no mesmo documento. Cada parser le as suas e ignora as
    outras, o que e melhor do que apostar no dialeto errado.
    """
    entries = _model_entries(request.app.state.settings)
    # Uma instalacao sem nenhum modelo configurado: `entries[0]` estouraria, e
    # `first_id: null` e o que os dois parsers leem como "nao ha nada aqui".
    # Os dois dialetos resolvem isso dentro do proprio schema; o superconjunto
    # e uniao de formatos e nao tem schema, entao calcula aqui.
    first = entries[0]["id"] if entries else None
    protocol = detect_protocol(request.headers)
    if protocol == "anthropic":
        return AnthropicModelList.of(entries).model_dump()
    if protocol == "openai":
        return OpenAIModelList.of(entries).model_dump()
    return {"object": "list", "data": entries, "has_more": False, "first_id": first}
