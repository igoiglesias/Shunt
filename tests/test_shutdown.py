"""O desligamento termina mesmo com o SSE do painel aberto.

Medido em 2026-09-23: `make prod` ficou preso porque uma aba mantinha
`/api/stats/stream` aberto e o uvicorn esperava sem prazo. Na versao instalada
(uvicorn 0.53.0) `Server.shutdown` chama `connection.shutdown()` em cada
conexao, e para uma resposta em curso isso so marca `keep_alive = False`
(`protocols/http/httptools_impl.py:349-356`): nenhum `http.disconnect`, nenhum
sinal chega ao gerador, e o `lifespan` de shutdown so roda depois da espera
(`server.py:297-311`). O que encerra a resposta e o cancelamento da task
quando `timeout_graceful_shutdown` vence (`server.py:302-307`). Por isso o
alvo `prod` passa o teto, e este teste sobe o servidor real, em processo,
para provar que ele sai dentro dele.
"""

import asyncio
import re
import socket
import time
from pathlib import Path

import httpx
import uvicorn

from app.core.security import issue_jwt
from app.main import app

SECRET = "segredo-do-teste-de-desligamento"
MAKEFILE = Path(__file__).resolve().parent.parent / "Makefile"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_the_prod_target_caps_the_graceful_shutdown():
    recipe = MAKEFILE.read_text().split("prod:")[1].split("\n\n")[0]
    found = re.search(r"--timeout-graceful-shutdown (\d+)", recipe)
    assert found is not None, recipe
    assert 1 <= int(found.group(1)) <= 10


async def test_an_open_sse_does_not_hold_the_shutdown_past_the_cap():
    app.state.admin_session_secret = SECRET
    port = _free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, timeout_graceful_shutdown=1, log_level="warning"
    )
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve())
    try:
        while not server.started:
            assert not serving.done(), serving.result()
            await asyncio.sleep(0.01)
        cookie = {"shunt_admin": issue_jwt(user_id=1, secret=SECRET, ttl_seconds=60)}
        async with (
            httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}", cookies=cookie, timeout=5
            ) as client,
            client.stream(
                "GET", "/api/stats/stream", headers={"accept": "text/event-stream"}
            ) as response,
        ):
            assert response.status_code == 200
            started = time.monotonic()
            server.should_exit = True
            await asyncio.wait_for(serving, 10)
            elapsed = time.monotonic() - started
        assert elapsed < 5, elapsed
    finally:
        if not serving.done():
            server.force_exit = True
            await asyncio.wait_for(serving, 5)
        for attr in ("settings", "pool", "recorder", "admin_session_secret", "config_watcher"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)
