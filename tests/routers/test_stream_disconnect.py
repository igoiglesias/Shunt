"""Cliente que desconecta no meio de um stream: o slot do provedor tem de voltar.

O `StreamingResponse` do Starlette nao chama `aclose()` no gerador do corpo
quando o cliente vai embora com o gerador parado num `yield` (o cancelamento
acerta o `send`, nao o gerador). Sem um fechamento explicito, a resposta
upstream e o slot de concorrencia so voltam quando o finalizador do gerador
rodar -- e o limite do provedor fica comido ate la.
"""

import asyncio
import contextlib
import json

import anyio
import httpx
import pytest
import respx
from starlette.requests import ClientDisconnect

from app.config.settings import ModelConfig, ProviderConfig, Settings
from app.core.upstream import UpstreamPool
from app.main import app
from tests.conftest import TEST_SHUNT_TOKEN

LIMITED = Settings(
    providers={
        "local": ProviderConfig(
            base_url="http://local.test/v1", protocol="openai", api_key="k", max_concurrency=1
        )
    },
    models={
        "loc": ModelConfig(
            provider="local", model="vendor/local", context_window=64000, max_output_tokens=8192
        )
    },
    routes=[("opus", ["loc"])],
    default_model=None,
)

ASK = {
    "model": "claude-opus-4-5",
    "max_tokens": 32,
    "stream": True,
    "messages": [{"role": "user", "content": "oi"}],
}


class _Endless(httpx.AsyncByteStream):
    """Provedor que manda um evento valido e depois segue gerando devagar."""

    def __init__(self) -> None:
        self.closed = False
        self.closing = asyncio.Event()

    async def __aiter__(self):
        yield b'data: {"choices": [{"delta": {"content": "a"}}]}\n\n'
        while True:
            await asyncio.sleep(0.001)
            yield b'data: {"choices": [{"delta": {"content": "b"}}]}\n\n'

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("spec_version", ["2.0", "2.4"])
@respx.mock
async def test_a_client_disconnect_mid_stream_returns_the_slot_without_gc(
    monkeypatch, spec_version
):
    upstream = _Endless()
    await _disconnect_mid_stream(monkeypatch, spec_version, upstream)


class _Stuck(_Endless):
    """Provedor cujo fechamento nunca termina (conexao presa num proxy)."""

    async def aclose(self) -> None:
        self.closing.set()
        await asyncio.Event().wait()


class _Slow(_Endless):
    """Provedor que demora um pouco (mas termina) para fechar."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self._delay = delay

    async def aclose(self) -> None:
        self.closing.set()
        await asyncio.sleep(self._delay)
        self.closed = True


@respx.mock
async def test_a_close_that_never_ends_is_bounded_and_still_returns_the_slot(monkeypatch):
    from app.routers import v1

    monkeypatch.setattr(v1, "STREAM_CLOSE_BOUND", 0.05)
    await _disconnect_mid_stream(monkeypatch, "2.0", _Stuck(), expect_closed=False)


@respx.mock
async def test_a_slow_close_inside_the_bound_is_allowed_to_finish(monkeypatch):
    # O teto padrao nao pode cortar um fechamento normal, so um preso.
    await _disconnect_mid_stream(monkeypatch, "2.0", _Slow(0.02))


@respx.mock
async def test_cancelling_the_request_does_not_interrupt_the_close(monkeypatch):
    # Blindagem: um escopo anyio cancelado em volta do app (o tipo de
    # cancelamento que o Starlette usa) no meio do fechamento nao o
    # interrompe. Medido: um `task.cancel()` NATIVO do asyncio atravessa o
    # shield do anyio -- esse caso (desligamento forcado do servidor) nao e
    # coberto, e o processo esta morrendo de qualquer jeito.
    await _disconnect_mid_stream(monkeypatch, "2.0", _Slow(0.05), cancel_during_close=True)


async def _disconnect_mid_stream(
    monkeypatch, spec_version, upstream, expect_closed=True, cancel_during_close=False
):
    respx.post("http://local.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=upstream)
    )
    pool = UpstreamPool(LIMITED)
    monkeypatch.setattr(app.state, "settings", LIMITED, raising=False)
    monkeypatch.setattr(app.state, "pool", pool, raising=False)

    body = json.dumps(ASK).encode()
    first_body = asyncio.Event()
    gone = asyncio.Event()
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_body.set()
            if gone.is_set():
                # Servidor 2.4: escrever num socket fechado levanta OSError.
                raise OSError("cliente foi embora")
            # O cliente para de ler: o `send` fica parado, e o gerador do
            # corpo fica suspenso no `yield` -- e o caso que vaza.
            await gone.wait()
            if spec_version == "2.4":
                raise OSError("cliente foi embora")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"host", b"test"), (b"x-shunt-token", TEST_SHUNT_TOKEN.encode())],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
        "state": {},
    }

    holder: dict[str, anyio.CancelScope] = {}

    async def serve():
        with anyio.CancelScope() as around:
            holder["scope"] = around
            await app(scope, receive, send)

    task = asyncio.ensure_future(serve())
    try:
        await asyncio.wait_for(first_body.wait(), 5.0)
        assert pool.in_use("local") == 1
        gone.set()
        if cancel_during_close:
            await asyncio.wait_for(upstream.closing.wait(), 2.0)
            holder["scope"].cancel()
        # `asyncio.wait`, nao `wait_for`: TimeoutError e subclasse de OSError
        # e seria engolido pelo `suppress` abaixo -- um app preso passava.
        done, _ = await asyncio.wait({task}, timeout=2.0)
        assert task in done, "o app nao terminou: fechamento preso sem teto"
        # Caminho 2.4: o `OSError` do `send` vira ClientDisconnect, e o
        # middleware de erro ainda tenta responder 500 pelo socket fechado --
        # o `send` deste teste levanta OSError de novo. Nenhum dos dois e o
        # que se mede aqui.
        with contextlib.suppress(ClientDisconnect, OSError):
            task.result()
        # Sem gc.collect(): quando o app devolve, o slot ja voltou e a
        # resposta upstream ja foi fechada.
        assert pool.in_use("local") == 0
        assert upstream.closed is expect_closed
    finally:
        if not task.done():
            task.cancel()
        await pool.aclose()
