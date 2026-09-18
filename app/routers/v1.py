"""The Anthropic-facing surface the dispatcher answers through.

Deliberately minimal: one non-streaming route. `dispatch` never raises on an
upstream failure -- it always returns a `ShuntResult`, already translated
into Anthropic error shape when every candidate failed -- so the only
exception this route needs to catch is `UnknownProviderError`, which escapes
`dispatch` before any candidate is tried (no provider is known for the
requested model, so there is nothing to retry or fall back to). Task 21
widens this to the full surface (streaming, `count_tokens`, the OpenAI
routes).
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.dispatcher import ShuntRequest, dispatch
from app.core.resolver import UnknownProviderError
from app.translate.to_anthropic import openai_error_to_anthropic

router = APIRouter(prefix="/v1")


@router.post("/messages")
async def create_message(request: Request) -> JSONResponse:
    body = await request.json()
    shunt_request = ShuntRequest("anthropic", body, dict(request.headers), endpoint="messages")
    try:
        result = await dispatch(shunt_request, request.app.state.settings, request.app.state.pool)
    except UnknownProviderError as err:
        return JSONResponse(status_code=400, content=openai_error_to_anthropic(400, str(err)))
    # `x-shunt-model` names the model that actually ran; the response body's
    # own `model` field still echoes what the client asked for (see
    # `openai_response_to_anthropic`). When no candidate ran at all,
    # `real_model` is None and the header is omitted rather than sent empty.
    headers = {"x-shunt-model": result.real_model} if result.real_model else {}
    return JSONResponse(status_code=result.status, content=result.body, headers=headers)
