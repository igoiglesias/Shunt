"""Toda rota vira evento, e nenhuma delas paga por isso.

Os testes aqui sobem o aplicativo inteiro com um gravador de verdade apontado
para um SQLite temporario, e conferem o que ficou gravado.
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
from app.stats.models import RequestEvent
from app.stats.recorder import Recorder
from tests.core.test_dispatcher import SETTINGS

ASK = {
    "model": "claude-opus-4-5",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "oi"}],
}


@pytest.fixture
def stored(make_engine, tmp_path):
    """O aplicativo com gravador real; devolve quem le a tabela depois."""
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'stats.db'}")
    recorder = Recorder(engine, batch_size=500, interval=0.01)
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    app.state.recorder = recorder

    def rows():
        with Session(engine) as session:
            return list(session.scalars(select(RequestEvent).order_by(RequestEvent.id)))

    yield rows
    del app.state.recorder
    set_recorder(Recorder(None))


def client():
    return TestClient(app)


def answer(model="vendor/free"):
    return {
        "id": "chatcmpl-1",
        "model": model,
        "choices": [{"message": {"content": "pronto"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }


@respx.mock
def test_a_served_request_is_stored_with_provider_candidate_and_tokens(stored):
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=answer())
    )
    with client() as c:
        assert c.post("/v1/messages", json=ASK).status_code == 200
    (event,) = stored()
    assert event.route == "/v1/messages"
    assert event.dialect == "anthropic"
    assert event.provider == "openrouter"
    assert event.candidate_model == "vendor/free"
    assert event.status == 200
    assert (event.input_tokens, event.output_tokens) == (7, 3)
    assert event.stream is False
    assert event.fell_back is False


@respx.mock
def test_a_fallback_is_visible_in_the_stored_event(stored):
    respx.post("https://api.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "nao"}}),
            httpx.Response(200, json=answer("vendor/cheap")),
        ]
    )
    with client() as c:
        c.post("/v1/messages", json=ASK)
    (event,) = stored()
    assert event.fell_back is True
    assert event.candidate_model == "vendor/cheap"
    assert len(event.attempts) == 1


@respx.mock
def test_the_tools_offered_and_the_tools_called_are_both_stored(stored):
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "vendor/free",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "read", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )
    )
    with client() as c:
        c.post(
            "/v1/messages",
            json={
                **ASK,
                "tools": [
                    {"name": "read", "input_schema": {"type": "object"}},
                    {"name": "write", "input_schema": {"type": "object"}},
                ],
            },
        )
    (event,) = stored()
    assert event.tools_offered == ["read", "write"]
    assert event.tools_called == ["read"]


@respx.mock
def test_a_stream_is_stored_with_its_time_to_first_token(stored):
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices": [{"delta": {"content": "oi"}}]}\n\n'
                b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )
    )
    with client() as c, c.stream("POST", "/v1/messages", json={**ASK, "stream": True}) as response:
        "".join(response.iter_text())
    (event,) = stored()
    assert event.stream is True
    assert event.provider == "openrouter"
    assert event.ttft_ms is not None


def test_a_refused_request_is_stored_with_no_candidate(stored):
    with client() as c:
        c.post("/v1/messages", json={**ASK, "model": "modelo-sem-dono"})
    (event,) = stored()
    assert event.status == 400
    assert event.error_type == "invalid_request_error"
    assert event.candidate_model is None
    assert event.requested_model == "modelo-sem-dono"


def test_a_malformed_body_is_stored_too(stored):
    with client() as c:
        c.post(
            "/v1/messages", content=b"{nao e json", headers={"content-type": "application/json"}
        )
    (event,) = stored()
    assert event.status == 400
    assert event.route == "/v1/messages"


def test_count_tokens_and_the_model_listing_are_stored(stored):
    with client() as c:
        c.post("/v1/messages/count_tokens", json={"model": "claude-opus-4-5", **ASK})
        c.get("/v1/models", headers={"anthropic-version": "2023-06-01"})
    routes = [event.route for event in stored()]
    assert routes == ["/v1/messages/count_tokens", "/v1/models"]
    assert stored()[0].input_tokens > 0


@respx.mock
def test_every_openai_route_is_stored_under_its_own_path(stored):
    for path, payload, answer_body in (
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "oi"}]}, answer()),
        ("/v1/completions", {"prompt": "oi"}, {"model": "vendor/free", "choices": []}),
        ("/v1/embeddings", {"input": "oi"}, {"model": "vendor/free", "data": []}),
    ):
        respx.post(f"https://api.test{path}").mock(
            return_value=httpx.Response(200, json=answer_body)
        )
        with client() as c:
            c.post(path, json={"model": "claude-opus-4-5", **payload})
    routes = [event.route for event in stored()]
    assert routes == ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"]
    assert {event.dialect for event in stored()} == {"openai"}
