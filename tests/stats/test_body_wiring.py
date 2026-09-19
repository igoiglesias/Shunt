"""A conversa gravada de ponta a ponta, nos dois modos.

Medido contra o proxy real: em streaming a resposta gravava VAZIA enquanto o
pedido gravava certo. Estes testes existem para que isso nao volte.
"""

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.observability import set_recorder
from app.core.upstream import UpstreamPool
from app.main import app
from app.stats.models import RequestBody
from app.stats.recorder import Recorder
from tests.core.test_dispatcher import SETTINGS

ASK = {
    "model": "claude-opus-4-5",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "conte ate tres"}],
}


@pytest.fixture
def gravando(make_engine, tmp_path, monkeypatch):
    monkeypatch.setenv("SHUNT_STORE_BODIES", "1")
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'conversa.db'}")
    recorder = Recorder(engine, batch_size=500, interval=0.01)
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    app.state.recorder = recorder

    def stored():
        with Session(engine) as session:
            return list(session.scalars(select(RequestBody)))

    yield stored
    del app.state.recorder
    set_recorder(Recorder(None))


@respx.mock
def test_a_buffered_request_stores_both_sides(gravando):
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "vendor/free",
                "choices": [{"message": {"content": "um dois tres"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )
    )
    with TestClient(app) as c:
        assert c.post("/v1/messages", json=ASK).status_code == 200
    (body,) = gravando()
    assert "user: conte ate tres" in body.prompt
    assert body.answer == "um dois tres"


@respx.mock
def test_a_streaming_request_stores_the_answer_too(gravando):
    """Medido no proxy real: gravava o pedido e deixava a resposta vazia."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices": [{"delta": {"content": "um"}}]}\n\n'
                b'data: {"choices": [{"delta": {"content": " dois"}}]}\n\n'
                b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )
    )
    with TestClient(app) as c, c.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as r:
        "".join(r.iter_text())
    (body,) = gravando()
    assert "user: conte ate tres" in body.prompt
    assert body.answer == "um dois", f"resposta gravada: {body.answer!r}"


@respx.mock
def test_nothing_is_stored_when_the_feature_is_off(gravando, monkeypatch):
    monkeypatch.setenv("SHUNT_STORE_BODIES", "")
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "c",
                "model": "vendor/free",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )
    )
    with TestClient(app) as c:
        c.post("/v1/messages", json=ASK)
    assert gravando() == []


@respx.mock
def test_the_reasoning_is_stored_in_streaming_just_like_buffered(gravando):
    """Medido contra o Groq: uma resposta que gastou todos os tokens pensando
    gravava completa sem stream e VAZIA com stream."""
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices": [{"delta": {"reasoning_content": "vou contar"}}]}\n\n'
                b'data: {"choices": [{"delta": {"content": "um"}}]}\n\n'
                b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )
    )
    with TestClient(app) as c, c.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as r:
        "".join(r.iter_text())
    (body,) = gravando()
    assert "[pensando] vou contar" in body.answer
    assert body.answer.endswith("um")
