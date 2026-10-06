"""A analise do periodo de ponta a ponta: rota real, dispatcher real, banco real.

O que so aparece aqui: a analise SAI PELO PROPRIO PROXY. O pedido dela atravessa
resolucao, cadeia, traducao e transporte como qualquer requisicao do harness --
e, por isso, ela mesma vira uma linha no historico. Um teste de unidade com
dublê de `dispatch` nunca mostraria isso.
"""

import json
import time
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.observability import set_recorder
from app.core.security import issue_jwt
from app.core.upstream import UpstreamPool
from app.main import app as shunt_app
from app.stats.models import Analysis, Base, RequestEvent
from app.stats.recorder import Recorder
from tests.e2e.conftest import E2E_SETTINGS
from tests.e2e.fake_provider import ScriptedTransport

NOW = datetime.now(UTC)

ADMIN_SESSION_SECRET = "segredo-de-teste-do-admin"


def seed(engine) -> None:
    with Session(engine) as session:
        session.add_all(
            [
                RequestEvent(
                    request_id=f"req-{index}",
                    started_at=NOW - timedelta(minutes=index + 1),
                    route="/v1/messages",
                    dialect="anthropic",
                    stream=False,
                    requested_model="claude-opus-5",
                    rule="family",
                    matched="opus",
                    provider="local",
                    candidate_model="qwen3-8b",
                    status=200 if index else 429,
                    error_type=None if index else "rate_limit_error",
                    input_tokens=1000 * (index + 1),
                    output_tokens=100,
                    ttft_ms=None,
                    duration_ms=300 * (index + 1),
                    attempts=["free: context window too small (90000 > 64000)"],
                    fell_back=True,
                    tools_offered=["Read", "Bash"],
                    tools_called=["Read"],
                    thinking_blocks=0,
                )
                for index in range(3)
            ]
        )
        session.commit()


@pytest.fixture
def shunt_with_history(provider, tmp_path):
    """O aplicativo real com banco de verdade, provedor falso e default declarado."""
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'stats.db'}")
    Base.metadata.create_all(engine)
    seed(engine)
    settings = E2E_SETTINGS.model_copy(deep=True, update={"default_model": "qwen"})
    shunt_app.state.settings = settings
    shunt_app.state.pool = UpstreamPool(settings, transport=ScriptedTransport(provider))
    shunt_app.state.recorder = Recorder(engine, interval=0.01)
    shunt_app.state.admin_session_secret = ADMIN_SESSION_SECRET
    with TestClient(shunt_app) as client:
        client.cookies.set(
            "shunt_admin",
            issue_jwt(user_id=1, secret=ADMIN_SESSION_SECRET, ttl_seconds=3600),
        )
        yield client, engine
    del shunt_app.state.recorder
    set_recorder(Recorder(None))
    engine.dispose()


def resposta_do_modelo(texto="1. Tire `Bash` do catálogo: oferecida 3, chamada 0."):
    return {
        "id": "chatcmpl-analise",
        "model": "qwen3-8b",
        "choices": [{"message": {"role": "assistant", "content": texto}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 40},
    }


def test_the_analysis_goes_out_through_the_proxy_and_comes_back_read(
    shunt_with_history, provider
):
    client, engine = shunt_with_history
    provider.queue(json_body=resposta_do_modelo())

    response = client.post("/api/analysis?provider=local")

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"].startswith("1. Tire `Bash`")
    assert payload["cached"] is False
    # Foi o proxy que resolveu o candidato: o corpo que chegou ao provedor
    # carrega o modelo REAL, e nao o alias que a rota pediu.
    assert provider.bodies[0]["model"] == "qwen3-8b"
    # E o dossie que ele mandou tem os numeros do periodo filtrado.
    enviado = provider.bodies[0]["messages"][-1]["content"]
    dossie = json.loads(enviado[enviado.index("{"): enviado.rindex("}") + 1])
    assert dossie["volume"]["requests"] == 3
    assert dossie["volume"]["errors"] == 1
    assert dossie["chain"]["skips"][0]["reason"] == "não coube"

    # E, porque saiu pelo proxy, a propria analise virou uma linha do historico:
    # ela aparece no painel como qualquer outra requisicao, e o token que ela
    # gastou entra na conta do periodo seguinte.
    # O gravador escreve em lote numa thread de fundo, entao a leitura espera --
    # sem isso o teste passaria ou falharia pelo relogio, e nao pelo codigo.
    gravadas = []
    limite = time.monotonic() + 5
    while time.monotonic() < limite and not gravadas:
        with Session(engine) as session:
            gravadas = session.query(RequestEvent).filter(RequestEvent.input_tokens == 900).all()
        if not gravadas:
            time.sleep(0.05)
    assert len(gravadas) == 1
    assert gravadas[0].candidate_model == "qwen3-8b"


def test_the_analysis_is_stored_with_its_dossier(shunt_with_history, provider):
    client, engine = shunt_with_history
    provider.queue(json_body=resposta_do_modelo())

    primeira = client.post("/api/analysis").json()
    segunda = client.post("/api/analysis").json()

    assert segunda["id"] == primeira["id"]
    assert segunda["cached"] is True
    # Uma unica chamada ao provedor para duas leituras: analise custa token.
    assert len(provider.bodies) == 1
    with Session(engine) as session:
        linha = session.get(Analysis, primeira["id"])
        assert linha.candidate_model == "qwen3-8b"
        assert linha.input_tokens == 900
        assert json.loads(linha.dossier_json)["volume"]["requests"] == 3

    listagem = client.get("/api/analysis").json()
    assert [item["id"] for item in listagem["analyses"]] == [primeira["id"]]
    assert client.get(f"/api/analysis/{primeira['id']}").json()["dossier"]["volume"]["requests"] == 3


def test_an_upstream_failure_is_reported_and_not_cached(shunt_with_history, provider):
    client, engine = shunt_with_history
    provider.queue(status=502, json_body={"error": {"message": "provedor caiu"}})
    provider.queue(status=502, json_body={"error": {"message": "provedor caiu"}})

    falhou = client.post("/api/analysis")

    assert falhou.status_code >= 400
    with Session(engine) as session:
        assert session.query(Analysis).count() == 0

    provider.queue(json_body=resposta_do_modelo())
    assert client.post("/api/analysis").status_code == 200


def test_a_failed_request_stores_the_upstream_error_body(shunt_with_history, provider):
    """O 422 do provedor chega ao banco: e o que a aba "Erro" da auditoria le.

    A analise e a rota que passa pelo dispatcher com banco ligado, entao e aqui
    que o par `prompt`/`error` se prova de ponta a ponta.
    """
    from app.stats.models import RequestBody

    client, engine = shunt_with_history
    provider.queue(
        status=422,
        json_body={"error": {"message": "prompt too long", "code": "context_length_exceeded"}},
    )
    provider.queue(
        status=422,
        json_body={"error": {"message": "prompt too long", "code": "context_length_exceeded"}},
    )

    resposta = client.post("/api/analysis")
    assert resposta.status_code >= 400

    def bodies():
        with Session(engine) as session:
            return list(session.scalars(select(RequestBody)))

    deadline = time.monotonic() + 2
    gravados = bodies()
    while not gravados and time.monotonic() < deadline:
        time.sleep(0.02)
        gravados = bodies()
    (gravado,) = gravados
    assert "prompt too long" in gravado.error
    assert "context_length_exceeded" in gravado.error
    assert gravado.error_bytes > 0


def test_without_a_declared_model_the_analysis_refuses(shunt_with_history, provider):
    client, _ = shunt_with_history
    shunt_app.state.settings = shunt_app.state.settings.model_copy(
        update={"default_model": None}
    )

    resposta = client.post("/api/analysis")

    assert resposta.status_code == 409
    assert "modelo" in resposta.json()["error"]
    assert provider.bodies == []
