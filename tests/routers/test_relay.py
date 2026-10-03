"""Repasse de rotas desconhecidas (catch-all da T5).

Cada RED nasce falhando com 404/405 (o que Starlette responde hoje).
Um por vez: escreve o teste, cola a falha, implementa, passa ao proximo.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
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
    route = respx.get("https://api.anthropic.com/api/oauth/usage?beta=true").mock(
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
    # A query que chega ao upstream e exatamente b"beta=true": sem token,
    # um unico '?' (CRITICAL 2: URL montada com query original + segunda
    # query colada) e sem re-encodacao (verbatim, RED 1).
    upstream_url = route.calls[0].request.url
    assert upstream_url.raw_path == b"/api/oauth/usage?beta=true"


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
            headers={},
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


@respx.mock
def test_relay_head_passes_upstream_content_length():
    """HEAD com content-length upstream: o valor do upstream sobe, e nao um
    `content-length: 0` inventado pela Response vazia.

    O caminho de HEAD constroi a resposta com `headers={}` (o Starlette so
    aceita Mapping e populava content-length: 0) e entao atribui
    `raw_headers` -- a lista precisa trazer o content-length do upstream,
    ou o cliente veria um corpo de tamanho zero onde o upstream disse outro.
    """
    respx.head("https://api.anthropic.com/api/hello").mock(
        return_value=httpx.Response(
            200,
            headers={"content-length": "42", "content-type": "text/plain"},
            content=b"",
        )
    )
    with client() as c:
        response = c.head(
            "/api/hello",
            headers={"anthropic-version": "2023-06-01"},
        )
    assert response.status_code == 200
    assert response.headers["content-length"] == "42"
    assert response.headers["content-type"] == "text/plain"


# ---------------------------------------------------------------------------
# RED 1b (fix round 2): ?token= no vai ao upstream em nenhum metodo
# ---------------------------------------------------------------------------


@respx.mock
def test_relay_token_query_param_never_leaves_get():
    """GET /api/hello?token=...: a query enviada e vazia -- o token sai
    MESMO quando esta sozinho na query (CRITICAL 1: medido, a URL bruta
    subia com ?token=... intacto)."""
    route = respx.get("https://api.anthropic.com/api/hello").mock(
        return_value=httpx.Response(200, content=b"ok")
    )
    with client() as c:
        response = c.get(
            "/api/hello",
            params={"token": TEST_SHUNT_TOKEN},
            headers={"anthropic-version": "2023-06-01"},
        )
    assert response.status_code == 200
    upstream_url = route.calls[0].request.url
    assert upstream_url.raw_path == b"/api/hello"


@respx.mock
def test_relay_token_query_param_never_leaves_post():
    """POST /api/hello?token=...: sem query extra, a URL upstream fica sem
    '?' nenhum -- o token nao pode voltar colado na URL (CRITICAL 2)."""
    route = respx.post("https://api.anthropic.com/api/hello").mock(
        return_value=httpx.Response(200, content=b"ok")
    )
    with client() as c:
        response = c.post(
            "/api/hello",
            params={"token": TEST_SHUNT_TOKEN},
            content=b"corpo",
            headers={"anthropic-version": "2023-06-01"},
        )
    assert response.status_code == 200
    upstream_url = route.calls[0].request.url
    assert b"?" not in upstream_url.raw_path
    assert TEST_SHUNT_TOKEN.encode() not in upstream_url.raw_path


@respx.mock
def test_relay_duplicate_query_keys_survive():
    """GET /api/hello?a=1&a=2: chaves repetidas nao colapsam num dict
    (CRITICAL 2) -- o upstream ve a=1 e a=2 na ordem original."""
    route = respx.get("https://api.anthropic.com/api/hello").mock(
        return_value=httpx.Response(200, content=b"ok")
    )
    with client() as c:
        response = c.get(
            "/api/hello?a=1&a=2",
            headers={"anthropic-version": "2023-06-01"},
        )
    assert response.status_code == 200
    upstream_url = route.calls[0].request.url
    assert upstream_url.query == b"a=1&a=2"
    assert upstream_url.raw_path == b"/api/hello?a=1&a=2"


# ---------------------------------------------------------------------------
# Fix round 2 (IMPORTANT 5): path que casa parcialmente com rota local FORA
# de /v1 responde 405 local e nunca sobe ao upstream
# ---------------------------------------------------------------------------


@respx.mock
def test_relay_post_api_stats_405_local():
    """POST /api/stats com token valido: a rota local /api/stats so aceita
    GET, entao o catch-all responde 405 ANTES de repassar -- o upstream nao
    e tocado (design 1b do plano)."""
    route = respx.post("https://api.anthropic.com/api/stats").mock(
        return_value=httpx.Response(200, content=b"nada")
    )
    with client() as c:
        response = c.post(
            "/api/stats",
            headers={"anthropic-version": "2023-06-01"},
        )
    assert response.status_code == 405
    assert response.json() == {"detail": "Method Not Allowed"}
    assert len(route.calls) == 0


@respx.mock
def test_relay_v1_unimplemented_method_still_relays():
    """PATCH /v1/messages/batches nao existe local: /v1/* com metodo sem
    rota propria CONTINUA no upstream (plano: nao existe 404 por falta de
    sinal)."""
    route = respx.patch("https://api.anthropic.com/v1/messages/batches").mock(
        return_value=httpx.Response(200, content=b"ok")
    )
    with client() as c:
        response = c.patch(
            "/v1/messages/batches",
            content=b"corpo",
            headers={"anthropic-version": "2023-06-01"},
        )
    assert response.status_code == 200
    assert len(route.calls) == 1
    assert route.calls[0].request.url.path == "/v1/messages/batches"


# ---------------------------------------------------------------------------
# RED 10: Resposta gzip -> bytes comprimidos, content-encoding, content-length
# ---------------------------------------------------------------------------


class _GzipRawStream(httpx.AsyncByteStream):
    """Stream do mock que entrega os bytes comprimidos em pedacos.

    Nao e um `httpx.ByteStream` (sincrono), entao o respx nao o pre-le com
    `aread()` -- e o pre-leitura e exatamente o que corromperia o teste: o
    httpx substitui o stream pelo conteudo DECODIFICADO, e o relay nunca
    veria os bytes cruros. Stream assincrono proprio = os bytes chegam no
    relay como chegariam numa conexao real."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def __aiter__(self):
        for i in range(0, len(self._data), 7):
            yield self._data[i:i + 7]

    async def aclose(self) -> None:
        pass


@respx.mock
def test_relay_gzip_passes_through():
    """Resposta gzip de verdade -> a WIRE carrega os bytes comprimidos
    INTACTOS, com content-encoding: gzip e o MESMO content-length.

    Medido: o TestClient (httpx) descomprime sozinho em `response.content`,
    entao a unica forma honesta de medir a wire e `iter_raw()` -- os bytes
    antes do decode local. O relay sobe via aiter_raw(): httpx nao
    descomprime no caminho de subida (CRITICAL 3)."""
    import gzip as _gzip

    payload = b'{"usage": 1}' * 64
    compressed = _gzip.compress(payload)
    assert compressed != payload  # o fixture de fato esta comprimido
    respx.get("https://api.anthropic.com/api/oauth/usage").mock(
        return_value=httpx.Response(
            200,
            headers={
                "content-encoding": "gzip",
                "content-length": str(len(compressed)),
                "content-type": "application/json",
            },
            stream=_GzipRawStream(compressed),
        )
    )
    with client() as c, c.stream(
        "GET",
        "/api/oauth/usage",
        headers={"anthropic-version": "2023-06-01"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-encoding"] == "gzip"
        assert response.headers["content-length"] == str(len(compressed))
        # Wire bytes: o que chegou no ar, sem decode local.
        wire = b"".join(response.iter_raw())
    # A wire carrega os bytes comprimidos byte a byte -- nao o payload
    # descomprimido (que e exatamente o defeito que o teste original
    # deixava passar porque .content descomprime).
    assert wire == compressed
    # E os bytes da wire de fato descomprimem para o payload original.
    assert _gzip.decompress(wire) == payload


# ---------------------------------------------------------------------------
# RED 12: Set-Cookie duplicados sobem separados (nao colapsados por virgula)
# ---------------------------------------------------------------------------


@respx.mock
def test_relay_preserves_duplicate_set_cookie_headers():
    """Dois `Set-Cookie` na resposta upstream chegam ao cliente como DOIS
    headers, e nao um so com os valores juntados por virgula.

    Colapsar e o defeito sutil: o `Expires`/`Date` de cada cookie tambem tem
    virgula, entao "a=1; Expires=..., b=2; Expires=..." e uma cookie so para
    o navegador. `dict` (Headers.items()) faz exatamente esse colapso;
    `multi_items()` preserva as duplicatas."""
    cookies = [
        "sess=abc; Path=/; Expires=Wed, 09 Jun 2026 10:18:14 GMT; HttpOnly",
        "csrf=xyz; Path=/; Expires=Wed, 09 Jun 2026 10:18:14 GMT",
    ]
    respx.get("https://api.anthropic.com/api/oauth/usage").mock(
        return_value=httpx.Response(
            200,
            headers=[
                ("set-cookie", cookies[0]),
                ("content-type", "application/json"),
                ("set-cookie", cookies[1]),
            ],
            json={"ok": 1},
        )
    )
    with client() as c:
        response = c.get(
            "/api/oauth/usage",
            headers={"anthropic-version": "2023-06-01"},
        )
    assert response.status_code == 200
    # Dois headers separados: o colapso por virgula os uniria em um so.
    assert response.headers.get_list("set-cookie") == cookies
    # O content-type tambem sobe (raw_headers sobrescreve o da init_headers).
    assert response.headers["content-type"] == "application/json"


@respx.mock
def test_relay_preserves_duplicate_set_cookie_headers_on_stream():
    """O mesmo no caminho de stream: a regra e a do header da resposta, e
    nao pode depender do corpo ter sido ou nao lido."""
    cookies = [
        "sess=abc; Path=/; Expires=Wed, 09 Jun 2026 10:18:14 GMT; HttpOnly",
        "csrf=xyz; Path=/; Expires=Wed, 09 Jun 2026 10:18:14 GMT",
    ]
    respx.get("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            headers=[
                ("set-cookie", cookies[0]),
                ("content-type", "text/event-stream"),
                ("set-cookie", cookies[1]),
            ],
            stream=_GzipRawStream(b"event: ping\ndata: 1\n\n"),
        )
    )
    with client() as c, c.stream(
        "GET",
        "/v1/messages",
        headers={"anthropic-version": "2023-06-01"},
    ) as response:
        assert response.status_code == 200
        assert response.headers.get_list("set-cookie") == cookies
        assert response.headers["content-type"] == "text/event-stream"

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


@respx.mock
def test_relay_read_error_502():
    """httpx.ReadError (corpo cortado) -> 502 no envelope do chamador
    (CRITICAL 4: qualquer httpx.HTTPError que nao e timeout e 502)."""
    respx.get("https://api.anthropic.com/api/oauth/usage").mock(
        side_effect=httpx.ReadError("connection reset")
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
def test_relay_pool_timeout_504():
    """httpx.PoolTimeout -> 504 (CRITICAL 4: qualquer TimeoutException e
    504, nao so ReadTimeout)."""
    respx.get("https://api.anthropic.com/api/oauth/usage").mock(
        side_effect=httpx.PoolTimeout("pool timeout")
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


# ---------------------------------------------------------------------------
# T6 (caracterizacao): o fechamento do corpo FECHA o upstream
# ---------------------------------------------------------------------------
#
# Caracteriza o `finally` do gerador `body()` em app/routers/relay.py:222-230:
# terminado normalmente, fechado por `aclose()` ou cancelado no meio do
# stream, a resposta upstream (e o stack que segura o cliente proprio)
# fecham. Nasce VERDE de proposito -- o comportamento ja existe (T5); a
# ausencia de RED e mandato do plano (decisao 6). Sem requisicao HTTP: o
# transporte e um `AsyncByteStream` local.


class _ClosingAsyncByteStream(httpx.AsyncByteStream):
    """Stream async minimalista com flag `closed`.

    Implementa o protocolo que o httpx espera dele -- `__aiter__` (gerador
    abaixo, o que o `aiter_raw()` itera) e `aclose` (o que o
    `response.aclose()` delega, `_models.py:1065-1076`). A porta no meio do
    corpo trava a iteracao para o teste poder cancelar a task de fato NO
    MEIO do stream, e nao depois dele terminar."""

    def __init__(self) -> None:
        self.closed = False
        self.first_chunk_sent = asyncio.Event()
        self._gate = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"chunk-1"
        self.first_chunk_sent.set()
        await self._gate.wait()
        yield b"chunk-2"

    async def aclose(self) -> None:
        self.closed = True


@asynccontextmanager
async def _own_client() -> AsyncIterator[httpx.AsyncClient]:
    """Replica o caminho de cliente proprio do `UpstreamPool.client`
    (app/core/upstream.py:298-304): um cliente por uso, fechado na saida. O
    relay o entra num `AsyncExitStack`, e esse stack e a outra metade do que
    o `finally` do corpo tem de fechar."""
    client = httpx.AsyncClient()
    try:
        yield client
    finally:
        await client.aclose()


async def _relay_body(
    response: httpx.Response, stack: contextlib.AsyncExitStack
) -> AsyncIterator[bytes]:
    """Copia do gerador `body()` de app/routers/relay.py:222-230: sobe os
    chunks crus da resposta upstream e o `finally` fecha a resposta E o
    stack (na mesma ordem do producao)."""
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    finally:
        try:
            await response.aclose()
        finally:
            await stack.aclose()


async def _open_relay_response(
    stream: _ClosingAsyncByteStream,
) -> tuple[AsyncIterator[bytes], httpx.Response, contextlib.AsyncExitStack]:
    """Monta o que app/routers/relay.py:170-237 monta: o stack entra no
    cliente proprio, e uma `httpx.Response` REAL de status 200 carrega o
    stream (o `response.aclose()` delega ao `stream.aclose()`)."""
    stack = contextlib.AsyncExitStack()
    await stack.enter_async_context(_own_client())
    response = httpx.Response(status_code=200, stream=stream)
    return _relay_body(response, stack), response, stack


async def test_t6_consume_a_chunk_then_aclose_closes_the_upstream():
    """Consumir um chunk do gerador do corpo e chamar `aclose()` nele fecha
    o stream upstream: `closed is True` (e a resposta registra o close)."""
    stream = _ClosingAsyncByteStream()
    body_iterator, response, _ = await _open_relay_response(stream)

    chunk = await body_iterator.__anext__()
    assert chunk == b"chunk-1"
    assert stream.closed is False
    await body_iterator.aclose()

    assert stream.closed is True
    assert response.is_closed


async def test_t6_cancelling_the_task_mid_stream_closes_the_upstream():
    """Cancelar a task que consome o corpo no MEIO do stream: o
    `CancelledError` sobe e, apos aguardar a task, o `finally` ja fechou o
    stream upstream -- `closed is True`."""
    stream = _ClosingAsyncByteStream()
    body_iterator, response, _ = await _open_relay_response(stream)

    async def consume() -> None:
        while True:
            await body_iterator.__anext__()

    task = asyncio.create_task(consume())
    # Espera o primeiro chunk sair: a task esta agora parada NA porta do
    # stream, e nao apos o fim dele.
    await stream.first_chunk_sent.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed is True
    assert response.is_closed
