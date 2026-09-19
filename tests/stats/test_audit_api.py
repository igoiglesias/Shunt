"""As rotas de auditoria: busca, detalhe e exportacao.

Um filtro que nao parseia e um filtro ausente, e nao um 400: quem esta
investigando prefere um resultado mais largo a uma recusa por causa de uma data
digitada pela metade.
"""

import csv
import io
import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.core.observability import set_recorder
from app.routers import audit
from app.stats.recorder import Recorder
from tests.stats.test_dashboard_api import _request_with
from tests.stats.test_queries import row

NOW = datetime.now(UTC)


@pytest.fixture
def store(make_engine, tmp_path):
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'auditoria.db'}")
    with Session(engine) as session:
        session.add_all(
            [
                row(
                    request_id="ok",
                    started_at=NOW - timedelta(minutes=1),
                    provider="groq",
                    candidate_model="openai/gpt-oss-120b",
                    project="/home/x/agenda",
                    session_id="s-1",
                    tools_offered=["Read"],
                    tools_called=["Read"],
                ),
                row(
                    request_id="falhou",
                    started_at=NOW - timedelta(minutes=5),
                    status=429,
                    error_type="rate_limit_error",
                    provider="openrouter",
                    candidate_model="openrouter/free",
                    attempts=["qwen: 502 (attempt 1)"],
                    fell_back=True,
                ),
            ]
        )
        session.commit()
    recorder = Recorder(engine)
    yield recorder
    set_recorder(Recorder(None))


def with_params(recorder, **params):
    request = _request_with(recorder)
    request.query_params = {k: str(v) for k, v in params.items()}
    return request


def body_of(response):
    return json.loads(response.body)


async def test_the_search_answers_a_page_with_its_total(store):
    answer = await audit.search_requests(with_params(store))
    page = body_of(answer)
    assert [event["request_id"] for event in page["events"]] == ["ok", "falhou"]
    assert page["total"] == 2
    assert page["next_cursor"] is None


async def test_filters_arrive_from_the_query_string(store):
    answer = await audit.search_requests(with_params(store, status_min=400))
    assert [event["request_id"] for event in body_of(answer)["events"]] == ["falhou"]

    answer = await audit.search_requests(with_params(store, has_tools="true"))
    assert [event["request_id"] for event in body_of(answer)["events"]] == ["ok"]

    answer = await audit.search_requests(with_params(store, text="rate_limit"))
    assert [event["request_id"] for event in body_of(answer)["events"]] == ["falhou"]


@pytest.mark.parametrize(
    "params",
    [
        {"since": "nao e data"},
        {"status_min": "muitos"},
        {"limit": "todas"},
        {"stream": ""},
        {"order_by": "abacaxi"},
    ],
)
async def test_a_filter_that_does_not_parse_is_simply_absent(store, params):
    answer = await audit.search_requests(with_params(store, **params))
    assert answer.status_code == 200
    assert body_of(answer)["total"] == 2


async def test_a_flag_left_out_does_not_mean_false(store):
    """Caixa desmarcada na tela nao pode virar "so as que NAO fizeram streaming"."""
    assert audit._flag({}, "stream") is None
    assert audit._flag({"stream": ""}, "stream") is None
    assert audit._flag({"stream": "true"}, "stream") is True
    assert audit._flag({"stream": "0"}, "stream") is False


async def test_paging_walks_with_the_cursor_from_the_previous_page(store):
    first = body_of(await audit.search_requests(with_params(store, limit=1)))
    assert first["next_cursor"]
    second = body_of(
        await audit.search_requests(with_params(store, limit=1, cursor=first["next_cursor"]))
    )
    assert [event["request_id"] for event in first["events"]] == ["ok"]
    assert [event["request_id"] for event in second["events"]] == ["falhou"]


async def test_one_request_comes_back_with_its_whole_chain(store):
    answer = await audit.one_request(with_params(store), "falhou")
    event = body_of(answer)
    assert event["attempts"] == ["qwen: 502 (attempt 1)"]
    assert event["fell_back"] is True
    assert event["error_type"] == "rate_limit_error"


async def test_an_unknown_request_answers_404(store):
    answer = await audit.one_request(with_params(store), "nunca-existiu")
    assert answer.status_code == 404


async def test_the_export_is_a_csv_with_a_row_per_request(store):
    answer = await audit.export_requests(with_params(store))
    assert answer.media_type.startswith("text/csv")
    assert "attachment" in answer.headers["content-disposition"]
    text = "".join([chunk async for chunk in answer.body_iterator])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert [r["request_id"] for r in rows] == ["ok", "falhou"]
    assert rows[1]["attempts"] == "qwen: 502 (attempt 1)"
    assert rows[0]["tools_called"] == "Read"


async def test_the_export_respects_the_same_filters(store):
    answer = await audit.export_requests(with_params(store, status_min=400))
    text = "".join([chunk async for chunk in answer.body_iterator])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert [r["request_id"] for r in rows] == ["falhou"]


@pytest.mark.parametrize(
    "call",
    [
        lambda request: audit.search_requests(request),
        lambda request: audit.export_requests(request),
        lambda request: audit.one_request(request, "seja-qual-for"),
    ],
)
async def test_without_a_database_every_route_says_so(call):
    answer = await call(with_params(Recorder(None)))
    assert answer.status_code == 409
    assert b"sem banco" in answer.body


async def test_an_instant_with_no_offset_is_read_as_utc(store):
    """A tela manda o que o campo de data der; sem fuso, vale o do gravador."""
    naive = audit._moment({"since": "2026-09-19T08:00:00"}, "since")
    aware = audit._moment({"since": "2026-09-19T08:00:00+00:00"}, "since")
    zulu = audit._moment({"since": "2026-09-19T08:00:00Z"}, "since")
    assert naive == aware == zulu
    assert naive.tzinfo is not None


async def test_the_facets_only_offer_what_exists(store):
    """Oferecer um provedor que nunca respondeu e um filtro que devolve vazio."""
    answer = await audit.facets(with_params(store))
    facets = body_of(answer)
    assert [f["value"] for f in facets["providers"]] == ["groq", "openrouter"]
    assert {f["value"] for f in facets["candidate_models"]} == {
        "openai/gpt-oss-120b",
        "openrouter/free",
    }
    assert [f["value"] for f in facets["error_types"]] == ["rate_limit_error"]
    assert facets["providers"][0]["count"] == 1


async def test_the_body_comes_back_when_it_was_stored(store, make_engine, tmp_path):
    from sqlalchemy.orm import Session

    from app.stats.models import RequestBody

    with Session(store.engine) as session:
        session.add(
            RequestBody(
                request_id="ok",
                prompt="user: leia a.txt",
                answer="pronto",
                prompt_bytes=16,
                answer_bytes=6,
                truncated=False,
            )
        )
        session.commit()
    answer = await audit.request_body(with_params(store), "ok")
    body = body_of(answer)
    assert body["prompt"] == "user: leia a.txt"
    assert body["answer"] == "pronto"
    assert body["truncated"] is False


async def test_a_request_without_a_stored_body_says_so(store):
    answer = await audit.request_body(with_params(store), "falhou")
    assert answer.status_code == 404
    assert b"nao gravada" in answer.body


async def test_without_a_database_the_body_and_the_facets_say_so():
    assert (await audit.request_body(with_params(Recorder(None)), "x")).status_code == 409
    assert (await audit.facets(with_params(Recorder(None)))).status_code == 409


async def test_the_facets_never_offer_an_empty_value(store, make_engine, tmp_path):
    """Mutante B10: sem o filtro de vazio, o seletor ganhava uma opcao em branco.

    `/v1/models` grava modelo pedido vazio, e uma opcao sem rotulo na lista e
    um filtro que ninguem consegue explicar.
    """
    from sqlalchemy.orm import Session

    from tests.stats.test_queries import row

    with Session(store.engine) as session:
        session.add(row(request_id="listagem", route="/v1/models", requested_model=""))
        session.commit()
    facets = body_of(await audit.facets(with_params(store)))
    for lista in facets.values():
        assert all(item["value"] for item in lista), lista


async def test_the_export_says_when_it_cut_the_result(store, monkeypatch):
    """Quem exporta precisa saber que levou uma parte."""
    monkeypatch.setattr(audit, "EXPORT_LIMIT", 1)
    answer = await audit.export_requests(with_params(store))
    assert answer.headers["x-shunt-truncated"] == "1"
    assert answer.headers["x-shunt-exported"] == "1"
    assert answer.headers["x-shunt-total"] == "2"


async def test_a_whole_export_says_it_is_whole(store):
    answer = await audit.export_requests(with_params(store))
    assert answer.headers["x-shunt-truncated"] == "0"
    assert answer.headers["x-shunt-exported"] == answer.headers["x-shunt-total"]


async def test_a_busca_filtra_por_projeto_e_o_csv_leva_a_coluna(store):
    pagina = body_of(await audit.search_requests(with_params(store, project="/home/x/agenda")))

    assert [e["request_id"] for e in pagina["events"]] == ["ok"]

    exportado = await audit.export_requests(with_params(store))
    partes = [parte async for parte in exportado.body_iterator]
    texto = "".join(p if isinstance(p, str) else p.decode() for p in partes)
    linhas = list(csv.DictReader(io.StringIO(texto)))

    # A planilha leva o projeto: a investigacao continua fora da tela.
    assert "project" in linhas[0]
    assert {linha["project"] for linha in linhas} == {"/home/x/agenda", ""}


async def test_as_facetas_oferecem_os_projetos_que_existem(store):
    facetas = body_of(await audit.facets(with_params(store)))

    assert facetas["projects"] == [{"value": "/home/x/agenda", "count": 1}]
