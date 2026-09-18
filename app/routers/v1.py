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

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

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

router = APIRouter(prefix="/v1")

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


async def _serve(request: Request, protocol: str, endpoint: str, streaming: bool):
    body = await request.json()
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
    body = await request.json()
    settings, pool = request.app.state.settings, request.app.state.pool
    try:
        candidate = resolve(body.get("model", ""), settings).chain[0]
    except UnknownProviderError as err:
        return JSONResponse(status_code=400, content=error_body("anthropic", 400, str(err)))
    if candidate.protocol == "anthropic":
        shunt_request = ShuntRequest("anthropic", body, dict(request.headers), endpoint="messages")
        upstream = await pool.get(candidate.provider).post(
            "/v1/messages/count_tokens",
            json={**body, "model": candidate.model},
            headers=outbound_headers(shunt_request, candidate, settings),
        )
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
    first = entries[0]["id"] if entries else None
    protocol = detect_protocol(request.headers)
    if protocol == "anthropic":
        return {
            "data": [
                {k: e[k] for k in ("type", "id", "display_name", "created_at")} for e in entries
            ],
            "has_more": False,
            "first_id": first,
        }
    if protocol == "openai":
        return {
            "object": "list",
            "data": [{k: e[k] for k in ("id", "object", "created", "owned_by")} for e in entries],
        }
    return {"object": "list", "data": entries, "has_more": False, "first_id": first}
