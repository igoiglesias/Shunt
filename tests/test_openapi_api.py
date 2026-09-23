"""O /docs descreve a API /api como ela e, e nao mostra as telas do admin.

O que se trava aqui e o CONTRATO publicado, derivado do que os handlers leem:
os nomes de parametro documentados tem de ser os que `filters_from` le de fato,
e os status os que o codigo devolve (409 sem banco, 401 sem sessao, 400 sem
`confirm`). O comportamento em si continua coberto pelos testes de
`tests/stats/test_*_api.py`; nada aqui troca a validacao manual por `Query`.
"""

import pytest

from app.main import app
from app.routers import audit

SESSION = [{"adminSession": []}]

# (metodo, caminho) -> tag
API_OPERATIONS = {
    ("get", "/api/stats"): "Panel",
    ("post", "/api/stats/clear"): "Panel",
    ("get", "/api/stats/stream"): "Panel",
    ("get", "/api/requests"): "Requests",
    ("get", "/api/requests/export"): "Requests",
    ("get", "/api/requests/facets"): "Requests",
    ("get", "/api/requests/{request_id}"): "Requests",
    ("get", "/api/requests/{request_id}/body"): "Requests",
    ("get", "/api/analysis"): "Analysis",
    ("post", "/api/analysis"): "Analysis",
    ("get", "/api/analysis/{analysis_id}"): "Analysis",
}

FILTERS = set(audit.filters_from({}))
ANALYSIS_FILTERS = set(audit._analysis_filters({}))

# Parametros de query que cada handler le de `request.query_params`.
QUERY_PARAMS = {
    ("get", "/api/stats"): {"window"},
    ("post", "/api/stats/clear"): set(),
    ("get", "/api/stats/stream"): set(),
    ("get", "/api/requests"): FILTERS | {"limit", "cursor"},
    ("get", "/api/requests/export"): FILTERS,
    ("get", "/api/requests/facets"): set(),
    ("get", "/api/requests/{request_id}"): set(),
    ("get", "/api/requests/{request_id}/body"): set(),
    ("get", "/api/analysis"): set(),
    ("post", "/api/analysis"): ANALYSIS_FILTERS,
    ("get", "/api/analysis/{analysis_id}"): set(),
}

# Status que o codigo devolve. 422 so onde o FastAPI valida parametro de
# caminho de verdade (`analysis_id: int`); um `str` sem restricao nunca falha,
# e `app.main._openapi` tira o 422 que o FastAPI acrescenta sozinho.
STATUSES = {
    ("get", "/api/stats"): {"200", "401"},
    ("post", "/api/stats/clear"): {"200", "400", "401", "409"},
    ("get", "/api/stats/stream"): {"200", "401"},
    ("get", "/api/requests"): {"200", "401", "409"},
    ("get", "/api/requests/export"): {"200", "401", "409"},
    ("get", "/api/requests/facets"): {"200", "401", "409"},
    ("get", "/api/requests/{request_id}"): {"200", "401", "404", "409"},
    ("get", "/api/requests/{request_id}/body"): {"200", "401", "404", "409"},
    ("get", "/api/analysis"): {"200", "401", "409"},
    ("post", "/api/analysis"): {"200", "401", "409", "502", "default"},
    ("get", "/api/analysis/{analysis_id}"): {"200", "401", "404", "409", "422"},
}

SUCCESS_MEDIA = {
    ("get", "/api/stats/stream"): "text/event-stream",
    ("get", "/api/requests/export"): "text/csv",
}


@pytest.fixture(scope="module")
def schema() -> dict:
    app.openapi_schema = None
    return app.openapi()


def _op(schema: dict, method: str, path: str) -> dict:
    return schema["paths"][path][method]


@pytest.mark.parametrize(("method", "path"), sorted(API_OPERATIONS))
def test_api_operation_carries_its_tag(schema, method, path):
    assert _op(schema, method, path).get("tags") == [API_OPERATIONS[(method, path)]]


@pytest.mark.parametrize(("method", "path"), sorted(API_OPERATIONS))
def test_api_operation_requires_admin_session(schema, method, path):
    assert _op(schema, method, path).get("security") == SESSION


@pytest.mark.parametrize(("method", "path"), sorted(API_OPERATIONS))
def test_api_operation_has_public_summary_and_description(schema, method, path):
    op = _op(schema, method, path)
    assert op.get("summary")
    assert op.get("description")


@pytest.mark.parametrize(("method", "path"), sorted(API_OPERATIONS))
def test_documented_query_params_match_what_the_handler_reads(schema, method, path):
    params = _op(schema, method, path).get("parameters", [])
    documented = {p["name"] for p in params if p["in"] == "query"}
    assert documented == QUERY_PARAMS[(method, path)]
    for param in params:
        assert param.get("schema"), param["name"]
        assert param.get("description"), param["name"]


@pytest.mark.parametrize(("method", "path"), sorted(API_OPERATIONS))
def test_documented_statuses_are_the_ones_the_code_returns(schema, method, path):
    assert set(_op(schema, method, path)["responses"]) == STATUSES[(method, path)]


@pytest.mark.parametrize(("method", "path"), sorted(API_OPERATIONS))
def test_success_media_type(schema, method, path):
    content = _op(schema, method, path)["responses"]["200"]["content"]
    assert set(content) == {SUCCESS_MEDIA.get((method, path), "application/json")}


# O servidor PRENDE `limit` e `window` entre limites, e nao recusa: publicar
# `minimum`/`maximum` faria o Swagger (e cliente gerado) recusar valor que o
# servidor aceita. A faixa vai no texto, com o numero configurado.


def test_limit_param_declares_default_and_states_the_clamp(schema):
    from app.config.config import MAX_SEARCH_LIMIT, SEARCH_LIMIT

    params = {p["name"]: p for p in _op(schema, "get", "/api/requests")["parameters"]}
    limit = params["limit"]
    assert limit["schema"]["default"] == SEARCH_LIMIT
    assert "minimum" not in limit["schema"]
    assert "maximum" not in limit["schema"]
    assert f"between 1 and {MAX_SEARCH_LIMIT}" in limit["description"]
    assert params["order_by"]["schema"]["enum"] == ["time", "duration"]


def test_window_param_declares_default_and_states_the_clamp(schema):
    from app.config.config import DEFAULT_HOURS, MAX_HOURS

    (window,) = _op(schema, "get", "/api/stats")["parameters"]
    assert window["schema"]["default"] == DEFAULT_HOURS
    assert "minimum" not in window["schema"]
    assert "maximum" not in window["schema"]
    assert f"between one minute and {MAX_HOURS:g} hours" in window["description"]


def test_clear_body_requires_confirm(schema):
    body = _op(schema, "post", "/api/stats/clear")["requestBody"]
    assert body["required"] is True
    shape = body["content"]["application/json"]["schema"]
    assert shape["required"] == ["confirm"]
    assert set(shape["properties"]) == {"confirm", "older_than_hours"}


def test_analysis_body_is_optional(schema):
    body = _op(schema, "post", "/api/analysis")["requestBody"]
    assert body["required"] is False
    shape = body["content"]["application/json"]["schema"]
    assert set(shape["properties"]) == {"model", "refresh"}
    assert "required" not in shape


def test_no_admin_html_screen_in_the_schema(schema):
    admin = {
        (method, path)
        for path, ops in schema["paths"].items()
        if path.startswith("/admin")
        for method in ops
    }
    assert admin == {("post", "/admin/config/reload")}


def test_config_reload_is_documented(schema):
    op = _op(schema, "post", "/admin/config/reload")
    assert op["tags"] == ["Admin"]
    assert op["security"] == SESSION
    assert op.get("summary") and op.get("description")
    assert set(op["responses"]) == {"200", "303", "401", "503"}
    ok = op["responses"]["200"]["content"]["application/json"]["schema"]
    assert set(ok["properties"]) == {"status", "default_model"}


# Tipo publicado de cada parametro: o que o handler de fato converte.
PARAM_TYPES = {
    "since": ("string", "date-time"),
    "until": ("string", "date-time"),
    "text": ("string", None),
    "route": ("string", None),
    "dialect": ("string", None),
    "provider": ("string", None),
    "candidate_model": ("string", None),
    "requested_model": ("string", None),
    "error_type": ("string", None),
    "project": ("string", None),
    "status_min": ("integer", None),
    "status_max": ("integer", None),
    "stream": ("boolean", None),
    "fell_back": ("boolean", None),
    "has_tools": ("boolean", None),
    "min_duration_ms": ("integer", None),
    "min_tokens": ("integer", None),
    "order_by": ("string", None),
    "limit": ("integer", None),
    "cursor": ("string", None),
    "window": ("number", None),
    "request_id": ("string", None),
    "analysis_id": ("integer", None),
}


@pytest.mark.parametrize(("method", "path"), sorted(API_OPERATIONS))
def test_every_param_has_the_type_the_handler_converts_to(schema, method, path):
    for param in _op(schema, method, path).get("parameters", []):
        kind, fmt = PARAM_TYPES[param["name"]]
        assert param["schema"]["type"] == kind, param["name"]
        assert param["schema"].get("format") == fmt, param["name"]


def test_clear_confirm_is_a_literal_true(schema):
    shape = _op(schema, "post", "/api/stats/clear")["requestBody"]["content"]["application/json"][
        "schema"
    ]
    assert shape["properties"]["confirm"]["type"] == "boolean"
    assert shape["properties"]["confirm"]["const"] is True
    assert shape["properties"]["older_than_hours"]["type"] == "number"


def test_export_documents_the_truncation_headers(schema):
    headers = _op(schema, "get", "/api/requests/export")["responses"]["200"]["headers"]
    assert set(headers) == {"x-shunt-exported", "x-shunt-total", "x-shunt-truncated"}


def test_analysis_refresh_defaults_to_false(schema):
    shape = _op(schema, "post", "/api/analysis")["requestBody"]["content"]["application/json"][
        "schema"
    ]
    assert shape["properties"]["refresh"] == {
        "type": "boolean",
        "default": False,
        "description": shape["properties"]["refresh"]["description"],
    }
    assert shape["properties"]["model"]["type"] == "string"


def _endpoint(method: str, path: str):
    from app.routers import admin_config, dashboard

    for route in [*audit.router.routes, *dashboard.router.routes, *admin_config.router.routes]:
        if getattr(route, "path", None) == path and method.upper() in getattr(route, "methods", ()):
            return route.endpoint
    raise LookupError((method, path))


@pytest.mark.parametrize(
    ("method", "path"), [*sorted(API_OPERATIONS), ("post", "/admin/config/reload")]
)
def test_public_text_is_not_the_framework_fallback(schema, method, path):
    """Sem `summary`/`description` o FastAPI usa o nome da funcao e a docstring
    em portugues -- o que e nota de implementacao, e nao texto para quem integra."""
    import inspect

    endpoint = _endpoint(method, path)
    op = _op(schema, method, path)
    assert op["summary"] != endpoint.__name__.replace("_", " ").title()
    assert op["description"] != inspect.cleandoc(endpoint.__doc__ or "")


@pytest.mark.parametrize(("method", "path"), sorted(API_OPERATIONS))
def test_query_params_are_optional(schema, method, path):
    for param in _op(schema, method, path).get("parameters", []):
        if param["in"] == "query":
            assert param["required"] is False, param["name"]


def test_analysis_upstream_failure_is_the_default_response(schema):
    """`analyse` devolve o status do modelo quando ele falha (analysis.py), qualquer
    4xx/5xx: o `default` e o que cobre esse leque, com o corpo que a funcao monta."""
    default = _op(schema, "post", "/api/analysis")["responses"]["default"]
    assert "passed through" in default["description"]
    shape = default["content"]["application/json"]["schema"]
    assert set(shape["properties"]) == {"status", "error", "text", "cached", "dossier", "trace"}
    assert shape["properties"]["status"]["type"] == "integer"
    assert shape["properties"]["error"]["type"] == "string"


def test_analysis_id_keeps_the_real_422(schema):
    """`analysis_id` e `int`: `/api/analysis/abc` falha a validacao de verdade."""
    assert "422" in _op(schema, "get", "/api/analysis/{analysis_id}")["responses"]


FAILURE_FIELDS = {"status", "error", "text", "cached", "dossier", "trace"}


def _analysis_response(schema: dict, status: str) -> dict:
    return _op(schema, "post", "/api/analysis")["responses"][status]


def _variants(response: dict) -> list[dict]:
    return response["content"]["application/json"]["schema"]["anyOf"]


def test_analysis_401_covers_the_session_and_the_upstream_key(schema):
    """401 tem duas origens: sessao do admin (`{"detail"}` do HTTPException) e o
    provedor recusando a chave, cujo status `analyse` repassa com o corpo dela."""
    response = _analysis_response(schema, "401")
    session, upstream = _variants(response)
    assert session["required"] == ["detail"]
    assert set(session["properties"]) == {"detail"}
    assert set(upstream["properties"]) == FAILURE_FIELDS
    assert "session" in response["description"]
    assert "model" in response["description"]


def test_analysis_409_covers_no_database_and_the_model_side(schema):
    response = _analysis_response(schema, "409")
    no_database, model_side = _variants(response)
    assert no_database["required"] == ["error"]
    assert set(no_database["properties"]) == {"error"}
    assert set(model_side["properties"]) == FAILURE_FIELDS
    assert "database" in response["description"]
    assert "default_model" in response["description"]
    assert "provider" in response["description"]


def test_analysis_502_carries_the_failure_body(schema):
    shape = _analysis_response(schema, "502")["content"]["application/json"]["schema"]
    assert set(shape["properties"]) == FAILURE_FIELDS
