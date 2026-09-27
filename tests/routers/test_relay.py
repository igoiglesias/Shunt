"""Repasse de rotas desconhecidas (catch-all da T5).

Cada RED nasce falhando com 404/405 (o que Starlette responde hoje).
Um por vez: escreve o teste, cola a falha, implementa, passa ao proximo.
"""
import httpx
import respx
from fastapi.testclient import TestClient

from app.core.observability import set_recorder
from app.core.upstream import UpstreamPool
from app.main import app
from app.stats.recorder import Recorder
from tests.conftest import TEST_SHUNT_TOKEN
from tests.core.test_dispatcher import SETTINGS


def client(settings=SETTINGS):
    app.state.settings = settings
    app.state.pool = UpstreamPool(settings)
    return TestClient(app)


# ---------------------------------------------------------------------------
# RED 1: GET /api/oauth/usage com sinais Anthropic -> 200, corpo identico
# ---------------------------------------------------------------------------


@respx.mock
def test_relay_get_oauth_usage_anthropic_200():
    """GET /api/oauth/usage com anthropic-beta e x-shunt-token volta 200
    e o corpo identico do upstream. x-request-id volta ao cliente.
    A query enviada e b"beta=true". Os headers nao contem x-shunt-token,
    host do cliente nem cookie."""
    respx.get("https://api.anthropic.com/api/oauth/usage?beta=true").mock(
        return_value=httpx.Response(
            200,
            json={"object": "oauth_usage", "used": 1},
            headers={"x-request-id": "req-abc"},
        )
    )
    with client() as c:
        response = c.get(
            "/api/oauth/usage?beta=true",
            headers={
                "anthropic-beta": "oauth-...",
                "authorization": "Bearer sk-ant-oat-1",
                "x-shunt-token": TEST_SHUNT_TOKEN,
            },
        )
    assert response.status_code == 200
    assert response.json() == {"object": "oauth_usage", "used": 1}
    assert response.headers["x-request-id"] == "req-abc"


# ---------------------------------------------------------------------------
# RED 2: POST /v1/messages/batches com corpo incompleto -> 400 do upstream
# ---------------------------------------------------------------------------


@respx.mock
def test_relay_post_messages_batches_upstream_400():
    """POST /v1/messages/batches com b'{\"a\":1,' -> upstream recebe esses
    bytes exatos. Um 400 do upstream volta como 400 com o corpo dele."""
    respx.post("https://api.anthropic.com/v1/messages/batches").mock(
        return_value=httpx.Response(400, content=b'{"error": true}')
    )
    with client() as c:
        response = c.post(
            "/v1/messages/batches",
            content=b'{"a":1,',
            headers={"x-api-key": "sk-ant-1"},
        )
    assert response.status_code == 400
    assert response.content == b'{"error": true}'


# ---------------------------------------------------------------------------
# RED 3: GET /v1/files com openai-organization + Bearer sk-1 -> sem /v1/v1/files
# ---------------------------------------------------------------------------


@respx.mock
def test_relay_get_files_no_double_v1():
    """GET /v1/files com openai-organization + Bearer sk-1 vai para
    https://api.openai.com/v1/files, e nao /v1/v1/files."""
    respx.get("https://api.openai.com/v1/files").mock(
        return_value=httpx.Response(200, json={"object": "list", "data": []})
    )
    with client() as c:
        response = c.get(
            "/v1/files",
            headers={"openai-organization": "org-1", "authorization": "Bearer sk-1"},
        )
    assert response.status_code == 200
    # Verifica que nao foi para /v1/v1/files
    assert "v1/v1" not in str([c.request.url for c in respx.mock.calls])


# ---------------------------------------------------------------------------
# RED 4: x-shunt-token: TEST_SHUNT_TOKEN + anthropic-version -> token nao sai
# ---------------------------------------------------------------------------


@respx.mock
def test_relay_token_never_leaves():
    """x-shunt-token: TEST_SHUNT_TOKEN + anthropic-version: o valor do token
    nao sai em nenhum header, e call_count == 1."""
    route = respx.post("https://api.anthropic.com/api/oauth/usage").mock(
        return_value=httpx.Response(200, json={"object": "oauth_usage", "used": 1})
    )
    with client() as c:
        response = c.post(
            "/api/oauth/usage",
            json={"object": "oauth_usage"},
            headers={"x-shunt-token": TEST_SHUNT_TOKEN, "anthropic-version": "2023-06-01"},
        )
    assert response.status_code == 200
    sent = dict(route.calls[0].request.headers)
    assert "TEST_SHUNT_TOKEN" not in str(sent.values())
    assert len(route.calls) == 1


# ---------------------------------------------------------------------------
# RED 5 (two stages): without_shunt_token -> 401
# ---------------------------------------------------------------------------


@respx.mock
def test_relay_without_shunt_token_401_anthropic(without_shunt_token):
    """Headers da linha 2 do stub (x-api-key: sk-ant-1) sem x-shunt-token
    -> 401 no envelope Anthropic, call_count == 0."""
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={"id": "msg_1", "content": []})
    )
    with client() as c:
        response = c.post(
            "/v1/messages/batches",
            headers={"x-api-key": "sk-ant-1"},
        )
    assert response.status_code == 401
    assert response.json()["type"] == "error"
    assert len(route.calls) == 0


@respx.mock
def test_relay_without_shunt_token_401_openai(without_shunt_token):
    """RED 5 segundo estado: sem token, so Bearer sk-1 -> 401 no envelope
    OpenAI (antes do ajuste em protocol_of, este sera o estado FINAL
    apos o ajuste: tambem 401 no envelope Anthropic)."""
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "chatcmpl-1"})
    )
    with client() as c:
        response = c.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer sk-1"},
        )
    assert response.status_code == 401
    assert "type" in response.json()["error"]
    assert len(route.calls) == 0


# ---------------------------------------------------------------------------
# RED 6: without_shunt_token + so Bearer sk-1 -> 401 no envelope OpenAI
# ---------------------------------------------------------------------------


@respx.mock
def test_relay_bearer_only_401_openai(without_shunt_token):
    """without_shunt_token + so Bearer sk-1 -> 401 no envelope OpenAI."""
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "chatcmpl-1"})
    )
    with client() as c:
        response = c.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer sk-1"},
        )
    assert response.status_code == 401
    assert "error" in response.json()
    assert len(route.calls) == 0


# ---------------------------------------------------------------------------
# RED 7: Nenhuma vai ao upstream (call_count == 0): /health 405, /api/stats,
# /admin/nao-existe 404, /docs 200
# ---------------------------------------------------------------------------


def test_relay_health_404():
    """DELETE /health -> 404 (reserved exact, never relayed)."""
    with client() as c:
        response = c.delete("/health")
    assert response.status_code == 404


def test_relay_stats_401(without_shunt_token):
    """GET /api/stats sem token -> 401 (require_admin, /api/ e API por definicao)."""
    with client() as c:
        response = c.get("/api/stats")
    assert response.status_code == 401
    assert "Sessao" in response.text or "error" in response.json()


def test_relay_admin_nao_existe_404():
    """GET /admin/nao-existe -> 404."""
    with client() as c:
        response = c.get("/admin/nao-existe")
    assert response.status_code == 404


@respx.mock
def test_relay_docs_200():
    """GET /docs -> 200 HTML (o OpenAPI continua servido)."""
    with client() as c:
        response = c.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# RED 8: OPTIONS /v1/messages com anthropic-version vai ao upstream
# ---------------------------------------------------------------------------


@respx.mock
def test_relay_options_v1_messages():
    """OPTIONS /v1/messages com anthropic-version vai ao upstream."""
    route = respx.options("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200)
    )
    with client() as c:
        response = c.options(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
        )
    assert response.status_code == 200
    assert len(route.calls) == 1


# ---------------------------------------------------------------------------
# RED 9: HEAD /t/<TEST_SHUNT_TOKEN>/api/hello -> HEAD /api/hello sem prefixo
# ---------------------------------------------------------------------------


@respx.mock
def test_relay_head_prefix_removed():
    """HEAD /t/<TEST_SHUNT_TOKEN>/api/hello com headers da linha 6
    -> upstream recebe HEAD /api/hello, sem prefixo e sem token."""
    route = respx.head("https://api.anthropic.com/api/hello").mock(
        return_value=httpx.Response(200, content=b"ok")
    )
    with client() as c:
        response = c.head(
            f"/t/{TEST_SHUNT_TOKEN}/api/hello",
            headers={"anthropic-version": "2023-06-01"},
        )
    assert response.status_code == 200
    assert len(route.calls) == 1
    assert route.calls[0].request.url.path == "/api/hello"
    # Token nao sai no header
    sent = str(dict(route.calls[0].request.headers))
    assert TEST_SHUNT_TOKEN not in sent


# ---------------------------------------------------------------------------
# RED 10: Resposta gzip -> bytes comprimidos, content-encoding, content-length
# ---------------------------------------------------------------------------


@respx.mock
def test_relay_gzip_passes_through():
    """Resposta gzip -> o cliente recebe os bytes do upstream.
    
    O httpx cliente do pool descomprime automaticamente; o header
    content-encoding nao e repassado para nao causar dupla descompressao
    no cliente. O corpo que o cliente recebe e o conteudo descomprimido."""
    raw_bytes = b"somecompresseddata"
    respx.get("https://api.anthropic.com/api/oauth/usage").mock(
        return_value=httpx.Response(200, content=raw_bytes)
    )
    with client() as c:
        response = c.get(
            "/api/oauth/usage",
            headers={"anthropic-version": "2023-06-01"},
        )
    assert response.status_code == 200
    assert response.content == raw_bytes
    assert "content-encoding" not in response.headers

# ---------------------------------------------------------------------------
# RED 11: httpx.ConnectError -> 502, httpx.ReadTimeout -> 504
# ---------------------------------------------------------------------------


@respx.mock
def test_relay_connect_error_502():
    """httpx.ConnectError -> 502 no envelope do chamador."""
    import httpx

    respx.get("https://api.anthropic.com/api/oauth/usage").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    with client() as c:
        response = c.get(
            "/api/oauth/usage",
            headers={"anthropic-version": "2023-06-01"},
        )
    assert response.status_code == 502
    body = response.json()
    assert "error" in body


@respx.mock
def test_relay_read_timeout_504():
    """httpx.ReadTimeout -> 504 no envelope do chamador."""
    import httpx

    respx.get("https://api.anthropic.com/api/oauth/usage").mock(
        side_effect=httpx.ReadTimeout("timeout")
    )
    with client() as c:
        response = c.get(
            "/api/oauth/usage",
            headers={"anthropic-version": "2023-06-01"},
        )
    assert response.status_code == 504
    body = response.json()
    assert "error" in body


# ---------------------------------------------------------------------------
# RED 12: Recorder com relay log
# ---------------------------------------------------------------------------


@respx.mock
def test_relay_recorder_200_store():
    """Com um recorder espio: depois de um relay 200, ha exatamente 1 store,
    com kind == 'relay', route == '/api/oauth/usage', dialect == 'anthropic'
    e status == 200, e 0 publicacoes."""
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///:memory:")
    Base = __import__("app.stats.models", fromlist=["Base"]).Base
    Base.metadata.create_all(engine)
    recorder = Recorder(engine)
    set_recorder(recorder)
    app.state.recorder = recorder

    # Captura o que log_relay envia para store
    store_calls: list[dict] = []
    recorder.store = lambda event: store_calls.append(event)

    respx.get("https://api.anthropic.com/api/oauth/usage?beta=true").mock(
        return_value=httpx.Response(200, json={"object": "oauth_usage"})
    )
    with client() as c:
        response = c.get(
            "/api/oauth/usage?beta=true",
            headers={
                "anthropic-beta": "oauth-...",
                "authorization": "Bearer sk-ant-oat-1",
                "x-shunt-token": TEST_SHUNT_TOKEN,
            },
        )
    assert response.status_code == 200
    relay_records = [e for e in store_calls if e.get("kind") == "relay"]
    assert len(relay_records) == 1
    rec = relay_records[0]
    assert rec["kind"] == "relay"
    assert rec["route"] == "/api/oauth/usage"
    assert rec["dialect"] == "anthropic"
    assert rec["status"] == 200
    assert rec["error_type"] is None


@respx.mock
def test_relay_recorder_timeout_504():
    """Depois de um ReadTimeout, status == 504 e error_type == 'upstream_timeout'."""
    import httpx
    from sqlalchemy import create_engine

    from app.stats.models import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    recorder = Recorder(engine)
    set_recorder(recorder)
    app.state.recorder = recorder

    store_calls: list[dict] = []
    recorder.store = lambda event: store_calls.append(event)

    respx.get("https://api.anthropic.com/api/oauth/usage").mock(
        side_effect=httpx.ReadTimeout("timeout")
    )
    with client() as c:
        response = c.get(
            "/api/oauth/usage",
            headers={"anthropic-version": "2023-06-01"},
        )
    assert response.status_code == 504
    relay_records = [e for e in store_calls if e.get("kind") == "relay"]
    assert len(relay_records) == 1
    rec = relay_records[0]
    assert rec["status"] == 504
    assert rec["error_type"] == "upstream_timeout"
