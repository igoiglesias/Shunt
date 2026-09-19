"""Provedor falso scriptado, servido por transporte httpx dentro do processo.

O E2E nao sobe socket: o `ScriptedTransport` entrega a requisicao a um
aplicativo FastAPI em memoria. Isso mantem o teste deterministico e ainda
exerce o caminho real de `httpx`, incluindo cabecalhos, path e corpo.
"""

import asyncio
import json
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


@dataclass
class Scripted:
    """Uma resposta programada. `sse` vence `json_body` quando preenchida."""

    status: int = 200
    json_body: dict | None = None
    sse: list[bytes] = field(default_factory=list)
    chunk_delay: float = 0.0
    headers: dict[str, str] = field(default_factory=dict)
    raise_transport: bool = False


class FakeProvider:
    """Provedor scriptado: cada requisicao consome a proxima resposta da fila."""

    def __init__(self) -> None:
        self.script: list[Scripted] = []
        self.calls: list[httpx.Request] = []
        self.bodies: list[dict] = []
        self.app = self._build()

    def queue(self, *responses: Scripted, **single: object) -> None:
        if single:
            responses = (*responses, Scripted(**single))  # type: ignore[arg-type]
        self.script.extend(responses)

    def peek(self) -> Scripted | None:
        return self.script[0] if self.script else None

    def pop(self) -> Scripted:
        if self.script:
            return self.script.pop(0)
        return Scripted(status=500, json_body={"error": {"message": "script vazio"}})

    def _build(self) -> FastAPI:
        app = FastAPI()

        async def handle(request: Request):
            self.bodies.append(await request.json())
            scripted = self.pop()
            if scripted.sse:

                async def body():
                    for chunk in scripted.sse:
                        if scripted.chunk_delay:
                            await asyncio.sleep(scripted.chunk_delay)
                        yield chunk

                return StreamingResponse(
                    body(),
                    status_code=scripted.status,
                    media_type="text/event-stream",
                    headers=scripted.headers,
                )
            return JSONResponse(
                status_code=scripted.status,
                content=scripted.json_body or {},
                headers=scripted.headers,
            )

        # A `base_url` de um provedor OpenAI ja termina em `/v1`, entao o path
        # real vira `/v1/chat/completions`; registre as duas formas.
        for path in (
            "/chat/completions",
            "/completions",
            "/embeddings",
            "/messages",
            "/messages/count_tokens",
        ):
            app.post(path)(handle)
            app.post(f"/v1{path}")(handle)
        return app


class ScriptedTransport(httpx.AsyncBaseTransport):
    """Transporte que serve o provedor falso, e sabe morrer antes de chegar nele."""

    def __init__(self, provider: FakeProvider) -> None:
        self._provider = provider
        self._inner = httpx.ASGITransport(app=provider.app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._provider.calls.append(request)
        scripted = self._provider.peek()
        if scripted is not None and scripted.raise_transport:
            self._provider.pop()
            raise httpx.ConnectError("connection refused", request=request)
        return await self._inner.handle_async_request(request)


def sse_chunks(*payloads: dict, done: bool = True) -> list[bytes]:
    """Empacota dicionarios como eventos SSE `data:`, com o `[DONE]` no fim."""
    chunks = [f"data: {json.dumps(p, ensure_ascii=False)}\n\n".encode() for p in payloads]
    if done:
        chunks.append(b"data: [DONE]\n\n")
    return chunks
