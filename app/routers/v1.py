
import httpx
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.config.config import model_sources
from app.schemas.schemas import (
    AnthropicRequest,
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
)
from app.tools.conversors import transform_anthropic_to_openai, transform_openai_to_anthropic

router = APIRouter(
    prefix="/v1",
    tags=["OpenAI"]
)


@router.get("/models")
async def get_models():
    """Return the list of models supported."""
    data = [{"id": item.get("model"), "object": "model", "created": 1700000000, "owned_py": item.get("provider")} for item in model_sources]
    
    return {
        "object": "list",
        "data": data
    }

@router.post("/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest):
    """
    Processa conversas/mensagens no formato Chat.
    Suporta respostas em lote ou em streaming (SSE).
    """
    if body.stream:
        # Exemplo de resposta em streaming simulada
        async def event_generator():
            yield 'data: {"choices": [{"delta": {"content": "Olá!"}}]}\n\n'
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return {
        "id": "chatcmpl-proxy-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Resposta processada pelo seu proxy para chat."
                },
                "finish_reason": "stop"
            }
        ]
    }

@router.post("/completions")
async def create_completion(body: CompletionRequest):
    """
    Processa solicitações de texto simples/autocomplete (Modelos Legados).
    """
    return {
        "id": "cmpl-proxy-123",
        "object": "text_completion",
        "created": 1700000000,
        "model": body.model,
        "choices": [
            {
                "text": " Texto completado pelo proxy.",
                "index": 0,
                "finish_reason": "stop"
            }
        ]
    }

@router.post("/embeddings")
async def create_embeddings(body: EmbeddingRequest):
    """
    Gera vetores numéricos de embedding para textos.
    """
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.012, -0.023, 0.045, 0.089]  # Exemplo de vetor
            }
        ],
        "model": body.model
    }
    
@router.post("/messages")
async def create_anthropic_message(
    body: AnthropicRequest,
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None)
):
    # O Claude Code envia a chave via header 'x-api-key' ou 'Authorization'
    api_key = x_api_key or (authorization.replace("Bearer ", "") if authorization else None)
    
    if not api_key:
        raise HTTPException(status_code=401, detail="Header de autenticação ausente.")

    # Converte para formato OpenAI
    openai_payload = transform_anthropic_to_openai(body)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(TARGET_PROVIDER_URL, json=openai_payload, headers=headers)
            
            if response.status_code != 200:
                return JSONResponse(status_code=response.status_code, content=response.json())

            # Traduz a resposta de volta antes de responder ao Claude Code
            anthropic_response = transform_openai_to_anthropic(response.json())
            return JSONResponse(content=anthropic_response)

        except httpx.RequestError as err:
            raise HTTPException(status_code=502, detail=f"Erro de conexão com o servidor destino: {err!s}")
