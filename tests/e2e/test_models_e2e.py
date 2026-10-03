"""Endpoints /v1/models — lista de rotas do Shunt (nao modelos dos providers)."""

import pytest


def test_models_endpoint_lists_routes_not_provider_models(shunt, settings):
    """GET /v1/models deve listar as rotas do Shunt, nao os modelos brutos."""
    settings.routes = [("gpt-4o", ["fable"]), ("claude-opus", ["free"])]
    settings.models["fable"] = settings.models["free"].model_copy(
        update={"provider": "anthropic", "model": "claude-fable-5-1"}
    )
    settings.models["free"] = settings.models["free"].model_copy(
        update={"provider": "local", "model": "qwen3-8b"}
    )

    anthropic = shunt.get("/v1/models", headers={"anthropic-version": "2023-06-01"}).json()
    openai = shunt.get("/v1/models", headers={"authorization": "Bearer sk"}).json()

    # Resposta no dialeto Anthropic: ModelInfo so tem type/id/display_name/
    # created_at (app/schemas/anthropic.py:75-79). Nao existe `owned_by` no
    # formato Anthropic -- essa chave e uma extensao OpenAI.
    assert anthropic["data"] is not None
    assert len(anthropic["data"]) == 2
    for entry in anthropic["data"]:
        assert entry["type"] == "model"
        assert entry["id"] in ("gpt-4o", "claude-opus")
        assert "display_name" in entry
        assert "owned_by" not in entry
    assert anthropic["has_more"] is False

    # Resposta no dialeto OpenAI: Model so tem id/object/created/owned_by
    # (app/schemas/openai.py:64-68).
    assert openai["object"] == "list"
    assert len(openai["data"]) == 2
    for entry in openai["data"]:
        assert entry["object"] == "model"
        assert entry["owned_by"] == "shunt"
        assert entry["id"] in ("gpt-4o", "claude-opus")
        assert "candidates" not in entry
    # ListModelsResponse required = object + data; sem has_more/first_id.
    assert "has_more" not in openai


def test_models_endpoint_empty_when_no_routes(shunt, settings):
    """Sem rotas configuradas, lista vazia nos dois dialetos."""
    settings.routes = []
    anthropic = shunt.get("/v1/models", headers={"anthropic-version": "2023-06-01"}).json()
    openai = shunt.get("/v1/models", headers={"authorization": "Bearer sk"}).json()

    assert anthropic["data"] == []
    assert anthropic["has_more"] is False
    # `first_id: null` e o sinal de "catalogo vazio" do padrao Anthropic
    # (AnthropicModelList.of, app/schemas/anthropic.py:96).
    assert anthropic["first_id"] is None
    # Padrao OpenAI: ListModelsResponse so tem `object` e `data` -- sem
    # has_more/first_id, que sao extensoes do formato Anthropic.
    assert openai["object"] == "list"
    assert openai["data"] == []


def test_models_single_endpoint_returns_route_detail(shunt, settings):
    """GET /v1/models/{pattern} retorna a rota especifica."""
    settings.routes = [("gpt-4o", ["fable", "free"])]
    settings.models["fable"] = settings.models["free"].model_copy(
        update={"provider": "anthropic", "model": "claude-fable-5-1"}
    )

    anthropic = shunt.get(
        "/v1/models/gpt-4o", headers={"anthropic-version": "2023-06-01"}
    ).json()
    openai = shunt.get(
        "/v1/models/gpt-4o", headers={"authorization": "Bearer sk"}
    ).json()

    # Anthropic: ModelInfo -- type/id/display_name/created_at. A rota e a
    # candidata ficam codificadas no display_name; `candidates` e `owned_by`
    # nao existem neste dialeto.
    assert anthropic["type"] == "model"
    assert anthropic["id"] == "gpt-4o"
    assert anthropic["display_name"] == "gpt-4o -> fable, free"
    assert "candidates" not in anthropic
    assert "owned_by" not in anthropic

    # OpenAI: Model -- id/object/created/owned_by. Sem display_name e sem
    # candidates.
    assert openai["id"] == "gpt-4o"
    assert openai["object"] == "model"
    assert openai["owned_by"] == "shunt"
    assert "display_name" not in openai
    assert "candidates" not in openai


def test_models_single_404_for_unknown_route(shunt):
    """Rota inexistente retorna 404 no envelope do dialeto de quem pergunta.

    Os dois padroes classificam um recurso ausente como `not_found_error`:
    a tabela Anthropic (app/translate/to_anthropic.py:57) e a OpenAI
    (app/core/dispatcher.py:107) mapeiam 404 para o mesmo nome.
    """
    anthro = shunt.get(
        "/v1/models/inexistente", headers={"anthropic-version": "2023-06-01"}
    )
    assert anthro.status_code == 404
    # Envelope Anthropic: {"type": "error", "error": {...}}
    assert anthro.json()["type"] == "error"
    assert anthro.json()["error"]["type"] == "not_found_error"

    openai = shunt.get("/v1/models/inexistente", headers={"authorization": "Bearer sk"})
    assert openai.status_code == 404
    # Envelope OpenAI: {"error": {...}}
    assert openai.json()["error"]["type"] == "not_found_error"


def test_models_endpoint_superset_when_no_dialect(shunt, settings):
    """Sem cabecalho de dialeto, retorna superconjunto (uniao dos dois formatos)."""
    settings.routes = [("gpt-4o", ["fable"])]
    resp = shunt.get("/v1/models").json()
    assert resp["object"] == "list"
    assert "data" in resp
    assert "has_more" in resp
    assert "first_id" in resp
    assert len(resp["data"]) == 1
    assert resp["data"][0]["id"] == "gpt-4o"
    assert "candidates" in resp["data"][0]
    assert resp["data"][0]["owned_by"] == "shunt"


@pytest.mark.parametrize(
    "path,headers",
    [
        ("/v1/models", {"anthropic-version": "2023-06-01"}),
        ("/v1/models", {"authorization": "Bearer sk"}),
    ],
)
def test_models_endpoint_content_type_json(shunt, path, headers):
    """Resposta sempre application/json."""
    resp = shunt.get(path, headers=headers)
    assert resp.headers["content-type"].startswith("application/json")