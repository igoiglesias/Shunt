from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.core.upstream import UpstreamPool
from app.main import app
from app.routers import v1
from app.routers.v1 import detect_protocol
from tests.core.test_dispatcher import SETTINGS

EMPTY = Settings(providers={}, models={}, routes=[], default_model=None)


def client(settings=SETTINGS):
    app.state.settings = settings
    app.state.pool = UpstreamPool(settings)
    return TestClient(app)


def test_detect_protocol_by_header():
    assert detect_protocol({"anthropic-version": "2023-06-01"}) == "anthropic"
    assert detect_protocol({"x-api-key": "sk"}) == "anthropic"
    assert detect_protocol({"user-agent": "claude-cli/2.0"}) == "anthropic"
    assert detect_protocol({"authorization": "Bearer sk"}) == "openai"
    assert detect_protocol({}) == "unknown"


def test_detect_protocol_ignores_header_and_agent_casing():
    """A header name arrives in whatever case the client typed it; only
    Starlette's own mapping is already lower-cased."""
    assert detect_protocol({"Anthropic-Version": "2023-06-01"}) == "anthropic"
    assert detect_protocol({"X-Api-Key": "sk"}) == "anthropic"
    assert detect_protocol({"Authorization": "Bearer sk"}) == "openai"
    assert detect_protocol({"User-Agent": "Claude-CLI/2.0"}) == "anthropic"


def test_an_anthropic_sdk_user_agent_is_also_anthropic():
    """Both names in `ANTHROPIC_AGENTS` are load-bearing: the official SDKs
    send `anthropic-sdk-python/...`, not `claude-cli`."""
    assert detect_protocol({"user-agent": "anthropic-sdk-python/0.40"}) == "anthropic"


def test_an_anthropic_agent_wins_over_a_bearer_token():
    """Claude Code authenticates a subscription with an OAuth bearer token and
    no `x-api-key`. Reading that bearer as an OpenAI caller would answer the
    wrong dialect to the one client this proxy exists for."""
    assert (
        detect_protocol({"authorization": "Bearer oauth", "user-agent": "claude-cli/2.0"})
        == "anthropic"
    )


def test_an_api_key_wins_over_a_bearer_token_sent_alongside_it():
    """Um gateway na frente do proxy pode repassar os dois cabecalhos. O
    `x-api-key` e sinal exclusivo do dialeto Anthropic -- nenhum cliente
    OpenAI legitimo o manda -- enquanto `Authorization: Bearer` sozinho e
    ambiguo. Quem manda os dois esta se identificando como Anthropic."""
    assert (
        detect_protocol({"x-api-key": "sk-ant-x", "authorization": "Bearer sk-openai"})
        == "anthropic"
    )


def test_a_bearer_token_alone_is_still_read_as_openai():
    """A contraparte da assercao acima: sem o sinal inequivoco, o bearer
    sozinho continua valendo como OpenAI. Sem este par, inverter a ordem da
    deteccao passaria despercebido em uma das duas direcoes."""
    assert detect_protocol({"authorization": "Bearer sk-openai"}) == "openai"


def test_models_answers_anthropic_shape_for_an_anthropic_caller():
    with client() as c:
        body = c.get("/v1/models", headers={"anthropic-version": "2023-06-01"}).json()
    assert body["data"][0]["type"] == "model"
    assert "display_name" in body["data"][0]
    assert body["has_more"] is False


def test_the_anthropic_shape_carries_exactly_the_anthropic_keys():
    with client() as c:
        body = c.get("/v1/models", headers={"anthropic-version": "2023-06-01"}).json()
    assert set(body) == {"data", "has_more", "first_id"}
    assert set(body["data"][0]) == {"type", "id", "display_name", "created_at"}
    assert body["data"][0]["id"] == "opus"
    assert body["data"][0]["display_name"] == "opus -> free, cheap"
    assert body["data"][0]["created_at"] == "2026-01-01T00:00:00Z"
    assert body["first_id"] == "opus"


def test_models_answers_openai_shape_for_an_openai_caller():
    with client() as c:
        body = c.get("/v1/models", headers={"authorization": "Bearer sk"}).json()
    assert body["object"] == "list"
    assert body["data"][0]["object"] == "model"
    assert body["data"][0]["owned_by"]
    assert "type" not in body["data"][0]


def test_the_openai_shape_carries_exactly_the_openai_keys():
    with client() as c:
        body = c.get("/v1/models", headers={"authorization": "Bearer sk"}).json()
    assert set(body) == {"object", "data"}
    assert set(body["data"][0]) == {"id", "object", "created", "owned_by"}
    assert body["data"][0]["id"] == "opus"
    assert body["data"][0]["owned_by"] == "shunt"
    assert body["data"][0]["created"] == 1700000000


def test_models_falls_back_to_a_superset_both_parsers_accept():
    with client() as c:
        body = c.get("/v1/models").json()
    entry = body["data"][0]
    assert body["object"] == "list" and body["has_more"] is False
    assert entry["object"] == "model" and entry["type"] == "model"
    assert entry["owned_by"] and entry["display_name"]


def test_the_superset_lists_every_configured_model_in_order():
    with client() as c:
        body = c.get("/v1/models").json()
    assert [e["id"] for e in body["data"]] == ["opus"]
    assert body["first_id"] == "opus"


def test_an_installation_with_no_models_has_no_first_id():
    """`entries[0]` would raise on an empty table; the guard answers None, and
    `first_id: null` is what both parsers read as "nothing here"."""
    with client(EMPTY) as c:
        body = c.get("/v1/models", headers={"anthropic-version": "2023-06-01"}).json()
    assert body["data"] == []
    assert body["first_id"] is None


def test_get_one_model_answers_anthropic_shape_for_an_anthropic_caller():
    with client() as c:
        resp = c.get("/v1/models/opus", headers={"anthropic-version": "2023-06-01"})
    assert resp.status_code == 200
    entry = resp.json()
    assert entry["id"] == "opus"
    assert entry["type"] == "model"
    assert entry["display_name"] == "opus -> free, cheap"
    assert entry["created_at"] == "2026-01-01T00:00:00Z"
    assert set(entry) == {"type", "id", "display_name", "created_at"}


def test_get_one_model_answers_openai_shape_for_an_openai_caller():
    with client() as c:
        resp = c.get("/v1/models/opus", headers={"authorization": "Bearer sk"})
    assert resp.status_code == 200
    entry = resp.json()
    assert entry["id"] == "opus"
    assert entry["object"] == "model"
    assert entry["owned_by"] == "shunt"
    assert entry["created"] == 1700000000
    assert set(entry) == {"id", "object", "created", "owned_by"}


def test_get_one_model_falls_back_to_a_superset_when_the_dialect_is_unknown():
    with client() as c:
        resp = c.get("/v1/models/opus")
    assert resp.status_code == 200
    entry = resp.json()
    assert entry["id"] == "opus"
    assert entry["object"] == "model" and entry["type"] == "model"
    assert entry["owned_by"] and entry["display_name"]


def test_get_an_unknown_model_answers_404_in_the_callers_dialect():
    with client() as c:
        resp = c.get("/v1/models/nao-existe", headers={"anthropic-version": "2023-06-01"})
    assert resp.status_code == 404
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "not_found_error"
    assert "nao-existe" in body["error"]["message"]


def test_an_openai_caller_gets_an_openai_envelope_for_an_unknown_model():
    with client() as c:
        resp = c.get("/v1/models/nao-existe", headers={"authorization": "Bearer sk"})
    assert resp.status_code == 404
    body = resp.json()
    assert set(body) == {"error"}
    assert body["error"]["type"] == "not_found_error"
    assert "nao-existe" in body["error"]["message"]


def test_get_one_model_when_there_are_no_models_is_a_404_not_a_crash():
    with client(EMPTY) as c:
        resp = c.get("/v1/models/qualquer", headers={"anthropic-version": "2023-06-01"})
    assert resp.status_code == 404
    assert resp.json()["type"] == "error"


def test_get_one_model_records_observability_like_the_listing(monkeypatch):
    """O log e a mesma linha que a listagem: rota, dialeto de quem perguntou,
    e o alias pedido -- tanto quando responde 200 quanto quando responde 404.
    Sem este teste um `_record` com status ou route errado passa em silencio."""
    calls: list = []
    monkeypatch.setattr(v1, "log_request", lambda entry: calls.append(entry))
    with client() as c:
        assert c.get("/v1/models/opus", headers={"x-api-key": "sk"}).status_code == 200
        assert c.get("/v1/models/nao-existe", headers={"x-api-key": "sk"}).status_code == 404
    ok, miss = calls
    assert ok.route == "/v1/models" and ok.dialect == "anthropic" and ok.status == 200
    assert ok.requested_model == "opus"
    assert miss.route == "/v1/models" and miss.dialect == "anthropic" and miss.status == 404
    assert miss.requested_model == "nao-existe"
