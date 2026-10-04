"""Provedor falso scriptado, servido por transporte httpx dentro do processo.

O E2E nao sobe socket: o `ScriptedTransport` entrega a requisicao a um
aplicativo FastAPI em memoria. Isso mantem o teste deterministico e ainda
exerce o caminho real de `httpx`, incluindo cabecalhos, path e corpo.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


@dataclass
class Scripted:
    """Uma resposta programada. `sse` vence `json_body` quando preenchida."""

    status: int = 200
    json_body: dict | None = None
    sse: list[bytes] = field(default_factory=list)
    chunk_delay: float = 0.0
    headers: dict[str, str] = field(default_factory=dict)
    raise_transport: bool = False
    # Corpo que nao e JSON e nem SSE: uma pagina HTML de um gateway interposto,
    # um corpo truncado. Serve bruto, com o content-type declarado nos
    # `headers`, para o `response.json()` do dispatcher levantar -- o caminho
    # do `unreadable body` (`dispatcher.py:734-747`). Default neutro: vazio,
    # o handler segue pelo JSON.
    raw_body: bytes | None = None
    # Serve `raise_after` bytes do corpo e a conexao cai. O 200 e os primeiros
    # eventos ja foram para o fio quando o provedor morreu -- um provider real
    # derruba a conexao no meio, e o corte tem que ser no MEIO da resposta.
    # `chunk_delay` nao serve: ele atrasa a PRODUCAO, e o `ASGITransport` do
    # httpx (0.28) bufferiza a resposta inteira antes de devolve-la, entao o
    # atraso some no `send` e a leitura ve um corpo instantaneo. So o
    # `ScriptedTransport` consegue quebrar DEPOIS do comeco do corpo.
    raise_after: int = 0
    # Pausa antes do PRIMEIRO byte do corpo. Mesma razao do `raise_after`: um
    # provedor que aceita a conexao e fica em silencio nao pode ser imitado
    # pelo gerador do FastAPI, porque o `ASGITransport` roda ele ate o fim
    # dentro do `send`.
    first_byte_delay: float = 0.0


class FakeProvider:
    """Provedor scriptado: cada requisicao consome a proxima resposta da fila."""

    def __init__(self) -> None:
        self.script: list[Scripted] = []
        self.calls: list[httpx.Request] = []
        self.bodies: list[dict] = []
        # Corpo CRU dos paths que nao sao de modelo: o relay repassa bytes
        # identicos, e `request.json()` nao serve -- um corpo parcial ou
        # comprimido nao parseia (e nao deve parsear).
        self.raw_bodies: list[bytes] = []
        # A resposta que o handler ACABOU de consumir. O `ScriptedTransport`
        # le depois do `handle_async_request` do ASGITransport voltar, para
        # saber se a resposta precisa de atraso ou de corte no meio do corpo.
        self.consumed: Scripted | None = None
        self.app = self._build()

    def queue(self, *responses: Scripted, **single: object) -> None:
        if single:
            responses = (*responses, Scripted(**single))  # type: ignore[arg-type]
        self.script.extend(responses)

    def peek(self) -> Scripted | None:
        return self.script[0] if self.script else None

    def pop(self) -> Scripted:
        if self.script:
            self.consumed = self.script.pop(0)
            return self.consumed
        self.consumed = Scripted(status=500, json_body={"error": {"message": "script vazio"}})
        return self.consumed

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
            if scripted.raw_body is not None:
                return Response(
                    content=scripted.raw_body,
                    status_code=scripted.status,
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

        # Catch-all DEPOIS dos paths de modelo, com os 7 metodos do relay
        # (`app/routers/relay.py`): uma rota desconhecida pode virar GET, HEAD
        # ou OPTIONS, e `handle` so serve POST de corpo JSON. O corpo e lido
        # CRU: o relay promete bytes identicos, e parsear aqui faria o falso
        # rejeitar um corpo que o upstream verdadeiro aceitaria.
        async def catch_all(request: Request):
            self.raw_bodies.append(await request.body())
            scripted = self.pop()
            if scripted.sse:

                async def sse_body():
                    for chunk in scripted.sse:
                        if scripted.chunk_delay:
                            await asyncio.sleep(scripted.chunk_delay)
                        yield chunk

                return StreamingResponse(
                    sse_body(),
                    status_code=scripted.status,
                    media_type="text/event-stream",
                    headers=scripted.headers,
                )
            if scripted.raw_body is not None:
                return Response(
                    content=scripted.raw_body,
                    status_code=scripted.status,
                    headers=scripted.headers,
                )
            return JSONResponse(
                status_code=scripted.status,
                content=scripted.json_body or {},
                headers=scripted.headers,
            )

        app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
        )(catch_all)
        return app


class _TamperedStream(httpx._types.AsyncByteStream):
    """O corpo da resposta, com atraso no primeiro byte e/ou queda no meio.

    Representa o que a rede real faz e o `ASGITransport` do httpx (0.28) nao
    consegue: ele bufferiza a resposta ASGI inteira antes de voltar, e a partir
    dai o corpo inteiro e um unico chunk instantaneo. O atraso e o corte sao
    aplicados no CONSUMO, que e onde o dispatcher os mede. A heranca e
    obrigatoria: o `_send_single_request` do httpx faz `isinstance` na stream
    (`_client.py:1732`) e duck typing nao passa.
    """

    def __init__(self, body: bytes, first_byte_delay: float = 0.0, raise_after: int = 0) -> None:
        self._chunks = _chunks(body, first_byte_delay, raise_after)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


async def _chunks(
    body: bytes, first_byte_delay: float, raise_after: int
) -> AsyncIterator[bytes]:
    if first_byte_delay:
        await asyncio.sleep(first_byte_delay)
    served = 0
    for chunk in _split(body):
        if raise_after and served + len(chunk) > raise_after:
            head = chunk[: raise_after - served]
            if head:
                yield head
            raise httpx.ReadError("connection dropped mid stream")
        served += len(chunk)
        yield chunk


def _split(body: bytes, size: int = 4096) -> Iterator[bytes]:
    for start in range(0, len(body), size):
        yield body[start : start + size]


class ScriptedTransport(httpx.AsyncBaseTransport):
    """Transporte que serve o provedor falso, e sabe morrer antes e durante a resposta."""

    def __init__(self, provider: FakeProvider) -> None:
        self._provider = provider
        self._inner = httpx.ASGITransport(app=provider.app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._provider.calls.append(request)
        scripted = self._provider.peek()
        if scripted is not None and scripted.raise_transport:
            self._provider.pop()
            raise httpx.ConnectError("connection refused", request=request)
        response = await self._inner.handle_async_request(request)
        tampered = self._provider.consumed
        if tampered is None or not (tampered.raise_after or tampered.first_byte_delay):
            return response
        body = await response.aread()
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=_TamperedStream(
                body,
                first_byte_delay=tampered.first_byte_delay,
                raise_after=tampered.raise_after,
            ),
            request=request,
            extensions=response.extensions,
        )


def sse_chunks(*payloads: dict, done: bool = True) -> list[bytes]:
    """Empacota dicionarios como eventos SSE `data:`, com o `[DONE]` no fim."""
    chunks = [f"data: {json.dumps(p, ensure_ascii=False)}\n\n".encode() for p in payloads]
    if done:
        chunks.append(b"data: [DONE]\n\n")
    return chunks
