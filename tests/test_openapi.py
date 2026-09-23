"""O documento OpenAPI de `/docs` descreve a superficie `/v1` de verdade.

As rotas POST de `/v1` leem o corpo na mao (`await request.json()`) de
proposito: um corpo invalido tem que virar 400 no envelope do dialeto de quem
perguntou, e nao o 422 generico do FastAPI. O preco era um `/docs` sem corpo
de requisicao, sem autenticacao e com toda resposta mostrada como "string".
Estes testes prendem a documentacao sem deixar a correcao mexer no runtime.
"""

import tomllib
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.core.upstream import UpstreamPool
from app.main import app
from tests.core.test_dispatcher import SETTINGS

TAGS = ["Anthropic", "OpenAI", "Models", "Panel", "Requests", "Analysis", "Admin", "System"]

REQUEST_MODELS = {
    "/v1/messages": "AnthropicRequest",
    "/v1/messages/count_tokens": "CountTokensRequest",
    "/v1/chat/completions": "ChatCompletionRequest",
    "/v1/completions": "CompletionRequest",
    "/v1/embeddings": "EmbeddingRequest",
}

PROXIED = ["/v1/messages", "/v1/chat/completions", "/v1/completions", "/v1/embeddings"]


@pytest.fixture(scope="module")
def spec() -> dict:
    app.openapi_schema = None
    return app.openapi()


def test_app_metadata_names_shunt_and_the_pyproject_version(spec):
    pyproject = tomllib.loads(
        (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert spec["info"]["title"] == "Shunt"
    assert spec["info"]["version"] == pyproject["project"]["version"]
    description = spec["info"]["description"]
    assert "Anthropic" in description and "OpenAI" in description
    assert "shunt_admin" in description


def test_tags_are_declared_in_order_with_descriptions(spec):
    assert [tag["name"] for tag in spec["tags"]] == TAGS
    assert all(tag.get("description") for tag in spec["tags"])


def test_security_schemes_have_the_exact_names_and_locations(spec):
    schemes = spec["components"]["securitySchemes"]
    assert schemes["anthropicApiKey"] == {
        "type": "apiKey",
        "in": "header",
        "name": "x-api-key",
        "description": schemes["anthropicApiKey"]["description"],
    }
    assert schemes["bearerAuth"]["type"] == "http"
    assert schemes["bearerAuth"]["scheme"] == "bearer"
    assert (schemes["shuntToken"]["type"], schemes["shuntToken"]["in"]) == ("apiKey", "header")
    assert schemes["shuntToken"]["name"] == "x-shunt-token"
    assert (schemes["adminSession"]["type"], schemes["adminSession"]["in"]) == (
        "apiKey",
        "cookie",
    )
    assert schemes["adminSession"]["name"] == "shunt_admin"


@pytest.mark.parametrize("path", PROXIED)
def test_proxied_routes_accept_any_of_the_three_credentials_or_none(spec, path):
    assert spec["paths"][path]["post"]["security"] == [
        {"anthropicApiKey": []},
        {"bearerAuth": []},
        {"shuntToken": []},
        {},
    ]


def test_count_tokens_does_not_claim_to_check_the_shunt_token(spec):
    """`count_tokens` nunca le `x-shunt-token` (v1.py, rota `count_tokens`)."""
    security = spec["paths"]["/v1/messages/count_tokens"]["post"]["security"]
    assert {"shuntToken": []} not in security
    assert {} in security


@pytest.mark.parametrize("path,model", REQUEST_MODELS.items())
def test_each_post_route_documents_its_request_model(spec, path, model):
    body = spec["paths"][path]["post"]["requestBody"]
    assert body["required"] is True
    assert body["content"]["application/json"]["schema"] == {
        "$ref": f"#/components/schemas/{model}"
    }
    assert model in spec["components"]["schemas"]


@pytest.mark.parametrize(
    "path,tag",
    [
        ("/v1/messages", "Anthropic"),
        ("/v1/messages/count_tokens", "Anthropic"),
        ("/v1/chat/completions", "OpenAI"),
        ("/v1/completions", "OpenAI"),
        ("/v1/embeddings", "OpenAI"),
    ],
)
def test_post_routes_carry_their_dialect_tag(spec, path, tag):
    assert spec["paths"][path]["post"]["tags"] == [tag]


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/chat/completions"])
def test_streaming_routes_document_server_sent_events(spec, path):
    content = spec["paths"][path]["post"]["responses"]["200"]["content"]
    assert "text/event-stream" in content
    assert "application/json" in content


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/embeddings", "/v1/messages/count_tokens"])
def test_non_streaming_routes_do_not_claim_server_sent_events(spec, path):
    content = spec["paths"][path]["post"]["responses"]["200"]["content"]
    assert "text/event-stream" not in content


@pytest.mark.parametrize("path", PROXIED)
def test_proxied_routes_declare_every_status_the_code_returns(spec, path):
    responses = spec["paths"][path]["post"]["responses"]
    assert set(responses) == {"200", "400", "401", "502", "503", "default"}
    assert "x-shunt-model" in responses["200"]["headers"]


def test_count_tokens_declares_only_200_and_400(spec):
    responses = spec["paths"]["/v1/messages/count_tokens"]["post"]["responses"]
    assert set(responses) == {"200", "400"}
    assert responses["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/CountTokensResponse"
    }


@pytest.mark.parametrize(
    "path,envelope",
    [
        ("/v1/messages", "AnthropicErrorResponse"),
        ("/v1/chat/completions", "OpenAIErrorResponse"),
        ("/v1/completions", "OpenAIErrorResponse"),
        ("/v1/embeddings", "OpenAIErrorResponse"),
        ("/v1/messages/count_tokens", "AnthropicErrorResponse"),
    ],
)
def test_400_uses_the_error_envelope_of_the_callers_dialect(spec, path, envelope):
    schema = spec["paths"][path]["post"]["responses"]["400"]["content"]["application/json"]
    assert schema["schema"] == {"$ref": f"#/components/schemas/{envelope}"}


def test_models_routes_are_tagged_and_explain_dialect_negotiation(spec):
    listing = spec["paths"]["/v1/models"]["get"]
    single = spec["paths"]["/v1/models/{model_id}"]["get"]
    assert listing["tags"] == ["Models"] and single["tags"] == ["Models"]
    for operation in (listing, single):
        assert "anthropic-version" in operation["description"]
        assert "Authorization" in operation["description"]
    assert "404" in single["responses"]
    assert "404" not in listing["responses"]


def test_health_is_a_system_route_and_the_stylesheet_is_hidden(spec):
    assert spec["paths"]["/health"]["get"]["tags"] == ["System"]
    assert spec["paths"]["/health"]["get"]["description"]
    assert "/shunt.css" not in spec["paths"]


@pytest.mark.parametrize(
    "path,method",
    [
        ("/v1/messages", "post"),
        ("/v1/messages/count_tokens", "post"),
        ("/v1/chat/completions", "post"),
        ("/v1/completions", "post"),
        ("/v1/embeddings", "post"),
        ("/v1/models", "get"),
        ("/v1/models/{model_id}", "get"),
        ("/health", "get"),
    ],
)
def test_public_routes_have_an_explicit_summary_and_no_internal_docstring(spec, path, method):
    operation = spec["paths"][path][method]
    assert operation["summary"]
    # As docstrings internas estao em portugues sem acento; a descricao
    # publica e em ingles. Nenhuma palavra das notas internas pode vazar.
    for leaked in ("Encaminha", "dialeto", "projecao", "Kept deliberately"):
        assert leaked not in operation.get("description", "")


def _walk_refs(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref":
                yield value
            else:
                yield from _walk_refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_refs(item)


def test_every_ref_in_the_document_resolves(spec):
    refs = list(_walk_refs(spec))
    assert refs, "o documento deveria ter ao menos uma $ref"
    for ref in refs:
        assert ref.startswith("#/"), ref
        target = spec
        for part in ref[2:].split("/"):
            assert part in target, f"$ref pendurada: {ref}"
            target = target[part]


def test_every_security_requirement_names_a_declared_scheme(spec):
    schemes = set(spec["components"]["securitySchemes"])
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            for requirement in operation.get("security", []):
                assert set(requirement) <= schemes, f"{method} {path}: {requirement}"


def test_docs_page_and_schema_are_served(spec):
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 200
        served = client.get("/openapi.json").json()
    assert served["info"]["title"] == "Shunt"


def test_invalid_body_is_still_a_400_in_the_anthropic_envelope_not_a_422():
    """A documentacao nao pode ter trocado a leitura manual por parametro Pydantic."""
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/messages", content='{"model": ', headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    missing = client.post("/v1/chat/completions", json={"messages": []})
    assert missing.status_code == 400
    assert set(missing.json()["error"]) == {"message", "type", "param", "code"}


def test_no_v1_route_advertises_the_framework_422(spec):
    """`/v1` nunca responde 422: o corpo e lido na mao (vira 400) e `model_id`
    e um `str` sem restricao. O 422 automatico do FastAPI seria uma promessa
    falsa -- os clientes dos dois protocolos nao sabem ler o formato dele."""
    for path, operations in spec["paths"].items():
        if path.startswith("/v1"):
            for method, operation in operations.items():
                assert "422" not in operation["responses"], f"{method} {path}"


def test_an_unconstrained_model_id_answers_404_not_422():
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    response = TestClient(app).get("/v1/models/%20%2F..%00")
    assert response.status_code == 404


SUMMARIES = {
    ("/v1/messages", "post"): "Create a message (Anthropic Messages API)",
    ("/v1/messages/count_tokens", "post"): "Count input tokens (Anthropic)",
    ("/v1/chat/completions", "post"): "Create a chat completion (OpenAI)",
    ("/v1/completions", "post"): "Create a text completion (OpenAI legacy)",
    ("/v1/embeddings", "post"): "Create embeddings (OpenAI)",
    ("/v1/models", "get"): "List models",
    ("/v1/models/{model_id}", "get"): "Get one model",
    ("/health", "get"): "Liveness check",
}


@pytest.mark.parametrize("key,summary", SUMMARIES.items())
def test_public_routes_carry_their_written_summary_and_a_description(spec, key, summary):
    """O FastAPI inventa um summary do nome da funcao ("Create Message") e usa
    a docstring interna como descricao; os dois tem que ser os escritos."""
    path, method = key
    operation = spec["paths"][path][method]
    assert operation["summary"] == summary
    assert len(operation["description"]) > 40


PROXIED_ENVELOPES = [
    ("/v1/messages", "AnthropicErrorResponse"),
    ("/v1/chat/completions", "OpenAIErrorResponse"),
    ("/v1/completions", "OpenAIErrorResponse"),
    ("/v1/embeddings", "OpenAIErrorResponse"),
]


@pytest.mark.parametrize("path,envelope", PROXIED_ENVELOPES)
def test_502_and_default_use_the_dialect_envelope(spec, path, envelope):
    """Status que o provedor devolveu passa adiante no envelope do dialeto
    (dispatcher: `ShuntResult(tally.last_status, error_body(...))`): o
    `default` cobre 403, 429, 500 e o que mais vier de la."""
    responses = spec["paths"][path]["post"]["responses"]
    ref = {"$ref": f"#/components/schemas/{envelope}"}
    assert responses["502"]["content"]["application/json"]["schema"] == ref
    assert responses["default"]["content"]["application/json"]["schema"] == ref


@pytest.mark.parametrize("path,envelope", PROXIED_ENVELOPES)
@pytest.mark.parametrize("status", ["401", "503"])
def test_401_and_503_are_either_shunt_detail_or_the_upstream_envelope(
    spec, path, envelope, status
):
    """Duas origens para o mesmo status: o `x-shunt-token` (corpo `detail`)
    e o provedor, cujo status passa adiante no envelope do dialeto."""
    response = spec["paths"][path]["post"]["responses"][status]
    schema = response["content"]["application/json"]["schema"]
    assert schema["anyOf"] == [
        {"type": "object", "properties": {"detail": {"type": "string"}}, "required": ["detail"]},
        {"$ref": f"#/components/schemas/{envelope}"},
    ]
    assert "x-shunt-token" in response["description"]
    assert "provider" in response["description"]


@respx.mock
@pytest.mark.parametrize("status,error_type", [(401, "authentication_error"), (503, None)])
def test_an_upstream_401_or_503_really_arrives_in_the_dialect_envelope(
    monkeypatch, status, error_type
):
    """O contrato documentado acima, medido: o status do provedor chega ao
    cliente no envelope OpenAI, e nao como `detail`."""
    monkeypatch.setattr("app.core.dispatcher.backoff", lambda attempt: 0)
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(status, json={"error": {"message": "no"}})
    )
    app.state.settings = SETTINGS
    app.state.pool = UpstreamPool(SETTINGS)
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "opus", "messages": [{"role": "user", "content": "oi"}]},
    )
    assert response.status_code == status
    body = response.json()
    assert "detail" not in body
    assert set(body["error"]) == {"message", "type", "param", "code"}
    if error_type:
        assert body["error"]["type"] == error_type


def test_stream_failure_is_described_for_both_dialects(spec):
    anthropic = spec["paths"]["/v1/messages"]["post"]["responses"]["200"]["description"]
    openai = spec["paths"]["/v1/chat/completions"]["post"]["responses"]["200"]["description"]
    for description in (anthropic, openai):
        assert "`event: error`" in description
        assert "`data:` chunk" in description


def test_models_negotiation_names_any_authorization_header(spec):
    """`detect_protocol` aceita qualquer `authorization`, nao so Bearer."""
    for path in ("/v1/models", "/v1/models/{model_id}"):
        description = spec["paths"][path]["get"]["description"]
        assert "an `Authorization` header" in description
        assert "Bearer" not in description


@pytest.mark.parametrize("path", PROXIED)
def test_400_names_the_capability_filter(spec, path):
    description = spec["paths"][path]["post"]["responses"]["400"]["description"]
    assert "no candidate can serve" in description


def test_a_route_added_after_the_first_build_shows_up(spec):
    """Mesmo contrato de cache do `FastAPI.openapi`: rota nova invalida."""
    app.openapi()

    async def late() -> dict:
        return {}

    app.add_api_route("/late-route-for-test", late)
    try:
        rebuilt = app.openapi()
        assert "/late-route-for-test" in rebuilt["paths"]
        assert rebuilt["components"]["securitySchemes"]["shuntToken"]["name"] == "x-shunt-token"
        assert "AnthropicRequest" in rebuilt["components"]["schemas"]
        assert "422" not in rebuilt["paths"]["/v1/models/{model_id}"]["get"]["responses"]
    finally:
        app.router.routes[:] = [
            r for r in app.router.routes if getattr(r, "path", "") != "/late-route-for-test"
        ]
        app.openapi_schema = None
    assert "/late-route-for-test" not in app.openapi()["paths"]


def test_app_level_openapi_arguments_are_passed_through(monkeypatch):
    """O override recebe o que o `FastAPI.openapi` original recebe."""
    monkeypatch.setattr(app, "servers", [{"url": "https://shunt.example"}])
    monkeypatch.setattr(app, "summary", "resumo")
    monkeypatch.setattr(app, "openapi_version", "3.1.1")
    app.openapi_schema = None
    try:
        built = app.openapi()
        assert built["servers"] == [{"url": "https://shunt.example"}]
        assert built["info"]["summary"] == "resumo"
        assert built["openapi"] == "3.1.1"
    finally:
        app.openapi_schema = None


def test_the_document_does_not_share_the_module_security_dicts():
    from app import main

    app.openapi_schema = None
    built = app.openapi()
    built["components"]["securitySchemes"]["shuntToken"]["name"] = "mudado"
    assert main.SECURITY_SCHEMES["shuntToken"]["name"] == "x-shunt-token"
    app.openapi_schema = None


def test_version_prefers_the_installed_distribution(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "version", lambda name: "7.7.7" if name == "shunt" else "")
    assert main._version() == "7.7.7"


def test_version_falls_back_to_zero_when_pyproject_is_unreadable(monkeypatch):
    from app import main

    def not_installed(name):
        raise main.PackageNotFoundError(name)

    def broken(_fh):
        raise tomllib.TOMLDecodeError("broken", "", 0)

    monkeypatch.setattr(main, "version", not_installed)
    monkeypatch.setattr(main.tomllib, "load", broken)
    assert main._version() == "0.0.0"


def test_version_falls_back_to_zero_when_pyproject_cannot_be_opened(monkeypatch):
    from app import main

    def not_installed(name):
        raise main.PackageNotFoundError(name)

    def unreadable(*_args, **_kwargs):
        raise PermissionError("sem leitura")

    monkeypatch.setattr(main, "version", not_installed)
    monkeypatch.setattr(main.Path, "open", unreadable)
    assert main._version() == "0.0.0"


def test_version_falls_back_to_zero_when_pyproject_has_no_version(monkeypatch):
    from app import main

    def not_installed(name):
        raise main.PackageNotFoundError(name)

    monkeypatch.setattr(main, "version", not_installed)
    monkeypatch.setattr(main.tomllib, "load", lambda _fh: {"project": {"name": "shunt"}})
    assert main._version() == "0.0.0"


def test_the_document_is_built_once_and_cached(spec):
    """O `/openapi.json` e servido a cada abertura de `/docs`: o documento e
    montado uma vez e reaproveitado, como no `FastAPI.openapi` original."""
    assert app.openapi() is app.openapi()


def test_a_cached_document_is_returned_untouched():
    """Como no `FastAPI.openapi`: quem ajusta `app.openapi_schema` depois da
    primeira geracao ve o ajuste mantido, e nao sobrescrito a cada chamada."""
    app.openapi_schema = None
    first = app.openapi()
    first["components"]["securitySchemes"]["shuntToken"]["description"] = "ajustado"
    try:
        assert app.openapi()["components"]["securitySchemes"]["shuntToken"][
            "description"
        ] == "ajustado"
    finally:
        app.openapi_schema = None


def _probe_route(path, endpoint):
    app.add_api_route(path, endpoint)
    app.openapi_schema = None
    try:
        return app.openapi()["paths"][path]["get"]["responses"]
    finally:
        app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", "") != path]
        app.openapi_schema = None


def test_a_plain_string_path_param_does_not_advertise_422():
    """Medido: `str` sem restricao gera `{"type": "string", "title": ...}` e
    nunca falha a validacao -- o 422 automatico seria promessa falsa."""

    async def plain(key: str) -> dict:
        return {}

    assert "422" not in _probe_route("/probe-plain/{key}", plain)


def test_a_constrained_string_path_param_keeps_422():
    """Medido: com `pattern`/`max_length` o FastAPI devolve 422 de verdade."""
    from fastapi import Path as PathParam

    async def constrained(key: str = PathParam(pattern="^a+$")) -> dict:
        return {}

    assert "422" in _probe_route("/probe-constrained/{key}", constrained)


def test_an_integer_path_param_keeps_422():
    async def number(key: int) -> dict:
        return {}

    assert "422" in _probe_route("/probe-int/{key}", number)


def test_a_plain_string_query_param_keeps_422():
    """Mesmo schema de string pura, mas em query: ausente, e 422 de verdade."""

    async def query(key: str, q: str) -> dict:
        return {}

    assert "422" in _probe_route("/probe-query/{key}", query)


def test_a_validated_body_keeps_422_even_with_a_plain_path_param():
    from pydantic import BaseModel

    class Payload(BaseModel):
        n: int

    async def with_body(key: str, payload: Payload) -> dict:
        return {}

    app.add_api_route("/probe-body/{key}", with_body, methods=["POST"])
    app.openapi_schema = None
    try:
        responses = app.openapi()["paths"]["/probe-body/{key}"]["post"]["responses"]
        assert "422" in responses
    finally:
        app.router.routes[:] = [
            r for r in app.router.routes if getattr(r, "path", "") != "/probe-body/{key}"
        ]
        app.openapi_schema = None
