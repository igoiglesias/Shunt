"""As rotas da analise: pedir, listar e reabrir uma analise ja paga."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.core.dispatcher import ShuntResult
from app.core.observability import set_recorder
from app.routers import audit
from app.stats import analysis
from app.stats.recorder import Recorder
from tests.core.test_dispatcher import SETTINGS
from tests.stats.test_dashboard_api import _request_with
from tests.stats.test_dossier import row

NOW = datetime.now(UTC)


@pytest.fixture
def store(make_engine, tmp_path):
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'stats.db'}")
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="a", started_at=NOW - timedelta(minutes=2)),
                row(request_id="b", started_at=NOW - timedelta(minutes=1), status=500),
            ]
        )
        session.commit()
    recorder = Recorder(engine)
    yield recorder
    set_recorder(Recorder(None))


def request_for(recorder, *, params=None, body=None):
    request = _request_with(recorder)
    request.query_params = {k: str(v) for k, v in (params or {}).items()}
    request.app.state.settings = SETTINGS.model_copy(update={"default_model": "cheap"})
    request.app.state.pool = None

    async def read_json():
        if body is None:
            raise ValueError("sem corpo")
        return body

    request.json = read_json
    return request


@pytest.fixture
def dublê(monkeypatch):
    enviados = []

    async def fake_dispatch(req, settings, pool):
        enviados.append(req)
        return ShuntResult(
            status=200,
            body={
                "content": [{"type": "text", "text": "1. Faça X (2 requisições)."}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            real_model="vendor/cheap",
            real_provider="openrouter",
        )

    monkeypatch.setattr(analysis, "dispatch", fake_dispatch)
    return enviados


def body_of(response):
    return json.loads(response.body)


def test_analise_herda_os_filtros_da_tela(store, dublê):
    resposta = asyncio.run(
        audit.analyse_period(request_for(store, params={"provider": "groq", "status_min": 500}))
    )

    assert resposta.status_code == 200
    payload = body_of(resposta)
    assert payload["text"].startswith("1. Faça X")
    # O periodo analisado e o periodo que a pessoa esta olhando.
    assert payload["dossier"]["period"]["filters"] == {"provider": "groq", "status_min": 500}


def test_sem_banco_a_analise_recusa_em_vez_de_analisar_o_vazio(dublê):
    resposta = asyncio.run(audit.analyse_period(request_for(None)))

    assert resposta.status_code == 409
    assert dublê == []


def test_erro_do_modelo_vira_status_do_erro(store, monkeypatch):
    async def falha(req, settings, pool):
        return ShuntResult(status=502, body={"error": {"message": "sem upstream"}})

    monkeypatch.setattr(analysis, "dispatch", falha)

    resposta = asyncio.run(audit.analyse_period(request_for(store)))

    assert resposta.status_code == 502
    assert "sem upstream" in body_of(resposta)["error"]


def test_segunda_chamada_devolve_a_analise_ja_paga(store, dublê):
    primeira = body_of(asyncio.run(audit.analyse_period(request_for(store))))
    segunda = body_of(asyncio.run(audit.analyse_period(request_for(store))))

    assert segunda["id"] == primeira["id"]
    assert segunda["cached"] is True
    assert len(dublê) == 1


def test_refresh_no_corpo_paga_de_novo(store, dublê):
    body_of(asyncio.run(audit.analyse_period(request_for(store))))
    segunda = body_of(
        asyncio.run(audit.analyse_period(request_for(store, body={"refresh": True})))
    )

    assert segunda["cached"] is False
    assert len(dublê) == 2


def test_modelo_pode_vir_do_corpo(store, dublê):
    asyncio.run(audit.analyse_period(request_for(store, body={"model": "claude-opus-5"})))

    assert dublê[0].body["model"] == "claude-opus-5"


def test_lista_e_detalhe_da_analise(store, dublê):
    salva = body_of(asyncio.run(audit.analyse_period(request_for(store))))

    listagem = body_of(asyncio.run(audit.list_analyses(request_for(store))))
    completa = body_of(asyncio.run(audit.one_analysis(request_for(store), salva["id"])))

    assert [item["id"] for item in listagem["analyses"]] == [salva["id"]]
    assert completa["dossier"]["volume"]["requests"] == 2


def test_analise_inexistente_e_404(store, dublê):
    resposta = asyncio.run(audit.one_analysis(request_for(store), 999))

    assert resposta.status_code == 404


def test_listar_e_abrir_sem_banco_recusam_tambem(dublê):
    listagem = asyncio.run(audit.list_analyses(request_for(None)))
    detalhe = asyncio.run(audit.one_analysis(request_for(None), 1))

    assert listagem.status_code == 409
    assert detalhe.status_code == 409


def test_a_listagem_oferece_os_modelos_configurados(store, dublê):
    listagem = body_of(asyncio.run(audit.list_analyses(request_for(store))))

    # A escolha de quem vai ler o período aparece ANTES de gastar: sem isso,
    # uma instalação sem `default_model` só descobre o 409 depois do clique.
    assert listagem["models"] == ["cheap", "free"]
    assert listagem["default_model"] == "cheap"
