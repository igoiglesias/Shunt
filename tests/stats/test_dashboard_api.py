"""A API do painel: cacheada, tolerante a banco ausente, e ao vivo pelo barramento."""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.observability import set_recorder
from app.core.upstream import UpstreamPool
from app.main import app
from app.routers.dashboard import stats_stream
from app.stats import queries
from app.stats.recorder import Recorder
from tests.core.test_dispatcher import SETTINGS
from tests.stats.test_queries import row


class _FakeApp:
    def __init__(self, recorder):
        self.state = type("State", (), {})()
        if recorder is not None:
            self.state.recorder = recorder


class _FakeRequest:
    """O minimo que as rotas do painel leem de uma requisicao."""

    def __init__(self, recorder):
        self.app = _FakeApp(recorder)
        self.query_params: dict[str, str] = {}


def _request_with(recorder):
    return _FakeRequest(recorder)


def _json(payload):
    """Substitui `Request.json`, que e assincrono."""

    async def read():
        return payload

    return read


@pytest.fixture
def panel(make_engine, tmp_path):
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'stats.db'}")
    with Session(engine) as session:
        session.add_all([row(request_id="a"), row(request_id="b", status=500)])
        session.commit()
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    app.state.recorder = Recorder(engine, interval=0.01)
    yield app.state.recorder
    del app.state.recorder
    set_recorder(Recorder(None))


@pytest.fixture(autouse=True)
def clear_cache():
    from app.routers import dashboard

    dashboard._cache.clear()
    yield
    dashboard._cache.clear()


def test_the_summary_carries_every_section_and_the_health_footer(panel):
    with TestClient(app) as c:
        response = c.get("/api/stats")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["totals"]["requests"] == 2
    assert body["totals"]["errors"] == 1
    assert body["health"] == {
        "enabled": True,
        "queued": 0,
        "dropped": 0,
        "failures": 0,
        "commits": 0,
    }


def test_the_second_call_inside_the_cache_window_does_not_query_again(panel, monkeypatch):
    calls = []
    original = queries.snapshot

    def counted(engine, hours):
        calls.append(hours)
        return original(engine, hours)

    monkeypatch.setattr(queries, "snapshot", counted)
    with TestClient(app) as c:
        c.get("/api/stats")
        c.get("/api/stats")
    assert calls == [24], "a segunda chamada tinha de vir do cache"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", 1),
        ("720", 720),
        ("99999", 720),
        ("0", 1 / 60),
        ("-5", 1 / 60),
        ("0.0833", 0.0833),  # os cinco minutos do painel
        ("abacaxi", 24),
        ("nan", 24),
        (None, 24),
    ],
)
def test_the_window_is_clamped_between_one_minute_and_a_month(panel, raw, expected):
    with TestClient(app) as c:
        url = "/api/stats" if raw is None else f"/api/stats?window={raw}"
        assert c.get(url).json()["window_hours"] == expected


def test_with_no_database_the_panel_answers_an_empty_summary_instead_of_failing():
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    app.state.recorder = Recorder(None)
    with TestClient(app) as c:
        body = c.get("/api/stats").json()
    del app.state.recorder
    assert body["totals"]["requests"] == 0
    assert body["health"]["enabled"] is False
    assert body["recent"] == []


async def test_the_stream_delivers_an_event_without_touching_the_database(panel, monkeypatch):
    """O SSE le o barramento em memoria, e nao a tabela.

    O teste roda o gerador no MESMO laco de eventos de quem grava, que e como
    acontece em producao: `record()` e chamado de dentro do laco. Uma
    `asyncio.Queue` alimentada de outra thread nao acordaria o leitor -- e um
    teste com `TestClient` faria exatamente isso, e ficaria pendurado.
    """

    def explode(*args, **kwargs):
        raise AssertionError("o SSE nao pode consultar o banco")

    monkeypatch.setattr(queries, "snapshot", explode)
    response = await stats_stream(_request_with(panel))
    events = response.body_iterator
    panel.record({"request_id": "ao-vivo", "route": "/v1/messages"})
    chunk = await anext(events)
    assert json.loads(chunk.decode().removeprefix("data: "))["request_id"] == "ao-vivo"
    await events.aclose()


async def test_closing_the_browser_releases_the_subscription(panel):
    response = await stats_stream(_request_with(panel))
    events = response.body_iterator
    panel.record({"request_id": "x"})
    await anext(events)
    await events.aclose()
    assert panel._subscribers == [], "a fila do navegador ficou para tras"


async def test_a_quiet_stream_sends_a_keepalive_instead_of_dying(panel, monkeypatch):
    """Proxy no meio do caminho derruba conexao ociosa; o painel nao pode morrer sozinho."""
    monkeypatch.setattr("app.routers.dashboard.KEEPALIVE_SECONDS", 0.01)
    response = await stats_stream(_request_with(panel))
    events = response.body_iterator
    assert await anext(events) == b": keepalive\n\n"
    # A segunda batida prova que o stream SEGUE depois de uma pausa, em vez de
    # entregar uma e terminar.
    assert await anext(events) == b": keepalive\n\n"
    await events.aclose()


async def test_a_stream_with_no_recorder_answers_instead_of_failing():
    response = await stats_stream(_request_with(None))
    assert response.media_type == "text/event-stream"


async def test_the_health_footer_answers_even_with_no_recorder_at_all():
    """Processo sem gravador -- um teste, um script -- ainda responde o painel."""
    from app.routers.dashboard import stats

    body = await stats(_request_with(None))
    assert body["health"] == {
        "enabled": False,
        "queued": 0,
        "dropped": 0,
        "failures": 0,
        "commits": 0,
    }


async def test_a_subscriber_that_went_away_does_not_break_the_stream(panel, monkeypatch):
    """Fila do navegador cheia: ele perde a linha ao vivo, ninguem mais paga."""
    monkeypatch.setattr("app.stats.recorder.SUBSCRIBER_QUEUE", 1)
    response = await stats_stream(_request_with(panel))
    events = response.body_iterator
    panel.record({"request_id": "primeiro"})
    panel.record({"request_id": "perdido"})
    chunk = await anext(events)
    assert b"primeiro" in chunk
    await events.aclose()


def test_the_stream_serialises_instants_the_same_way_the_summary_does():
    """Dois formatos de data na mesma fita a deixam fora de ordem."""
    from datetime import UTC, datetime

    from app.routers.dashboard import _iso, _sse

    moment = datetime(2026, 9, 19, 8, 30, tzinfo=UTC)
    assert _iso(moment) == "2026-09-19T08:30:00+00:00"
    assert _iso("ja e texto") == "ja e texto"
    chunk = _sse({"started_at": moment}).decode()
    assert "2026-09-19T08:30:00+00:00" in chunk
    assert "2026-09-19 08:30" not in chunk


async def test_clearing_needs_an_explicit_confirmation(panel):
    from app.routers.dashboard import clear_stats

    request = _request_with(panel)
    request.json = _json({})
    refused = await clear_stats(request)
    assert refused.status_code == 400
    assert b"confirm" in refused.body

    request = _request_with(panel)
    request.json = _json({"confirm": True})
    done = await clear_stats(request)
    assert done.status_code == 200
    assert json.loads(done.body)["deleted"] == 2


async def test_clearing_with_no_database_says_so_instead_of_pretending():
    from app.routers.dashboard import clear_stats

    request = _request_with(Recorder(None))
    request.json = _json({"confirm": True})
    answer = await clear_stats(request)
    assert answer.status_code == 409


async def test_clearing_only_the_past_is_accepted(panel):
    from app.routers.dashboard import clear_stats

    request = _request_with(panel)
    request.json = _json({"confirm": True, "older_than_hours": 24})
    answer = await clear_stats(request)
    assert answer.status_code == 200
    assert json.loads(answer.body)["older_than_hours"] == 24


async def test_clearing_drops_the_cached_summary(panel):
    """Sem isso o painel mostraria por segundos um resumo de linhas que sumiram."""
    from app.routers import dashboard

    await dashboard.stats(_request_with(panel))
    assert dashboard._cache, "o resumo tinha de estar em cache"
    request = _request_with(panel)
    request.json = _json({"confirm": True})
    await dashboard.clear_stats(request)
    assert dashboard._cache == {}


async def test_a_body_that_is_not_an_object_is_refused(panel):
    from app.routers.dashboard import clear_stats

    request = _request_with(panel)
    request.json = _json(["confirm"])
    assert (await clear_stats(request)).status_code == 400

    request = _request_with(panel)

    async def broken():
        raise ValueError("corpo ilegivel")

    request.json = broken
    assert (await clear_stats(request)).status_code == 400
