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
from tests.core.test_dispatcher import SETTINGS
from tests.stats.test_dashboard_api import _json, _request_with
from tests.stats.test_queries import relay_row, row

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
                    # Medidas reais de proposito: `None if relay else ...` so e
                    # distinguivel de "sempre None" quando a linha de modelo
                    # carrega de fato os valores.
                    input_tokens=10,
                    output_tokens=5,
                    ttft_ms=120,
                    cached_input_tokens=8,
                    cache_write_tokens=2,
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


@pytest.fixture
def store_with_relay(store, tmp_path):
    """A loja de cima, mais uma linha de relay.

    O relay e gravado com `input_tokens` e `output_tokens` ZERO porque a coluna
    e NOT NULL (`app/stats/models.py`): zero e o que o banco aceita, e a API e
    quem distingue "nao mede tokens" de "mediu zero". Por isso a afirmacao dos
    testes e `is None`, e nao `== 0`.
    """
    with Session(store.engine) as session:
        session.add(
            relay_row(
                request_id="repasse",
                started_at=NOW - timedelta(seconds=30),
                # Valores NAO nulos de proposito: o banco aceita os dois (sao
                # `nullable=True`, `app/stats/models.py`), e e justamente o
                # `kind` quem tem de deciding -- se a API copiasse o valor, a
                # linha do relay afirmaria "cacheou 8 tokens" sobre um trafego
                # de rede que nunca cacheou nada.
                input_tokens=0,
                output_tokens=0,
                ttft_ms=42,
                cached_input_tokens=8,
                cache_write_tokens=2,
            )
        )
        session.commit()
    return store


def body_of(response):
    return json.loads(response.body)


@pytest.fixture
def duble_do_modelo(monkeypatch):
    """O modelo que le o periodo: uma resposta de mentir, sem rede.

    O que interessa aqui e o CAMINHO ate o dossie; chamar um provedor de verdade
    pagaria uma requisicao a cada teste e tornaria a afirmacao flutuante.
    """
    from app.core.dispatcher import ShuntResult
    from app.stats import analysis

    async def fake_dispatch(req, settings, pool):
        return ShuntResult(
            status=200,
            body={
                "content": [{"type": "text", "text": "1. So ha modelo neste periodo."}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            real_model="vendor/cheap",
            real_provider="openrouter",
        )

    monkeypatch.setattr(analysis, "dispatch", fake_dispatch)


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


async def test_o_detalhe_e_o_csv_levam_o_cache_informado(store):
    """Sem isto, a tela nao distingue "nao cacheou" de "o provedor nao disse"."""
    pagina = body_of(await audit.search_requests(with_params(store)))
    evento = {e["request_id"]: e for e in pagina["events"]}

    assert evento["ok"]["cached_input_tokens"] == 8
    assert evento["ok"]["cache_write_tokens"] == 2
    # A requisicao cujo provedor nao informou continua NULA, e nao zero.
    assert evento["falhou"]["cached_input_tokens"] is None

    exportado = await audit.export_requests(with_params(store))
    partes = [parte async for parte in exportado.body_iterator]
    texto = "".join(p if isinstance(p, str) else p.decode() for p in partes)
    linhas = {linha["request_id"]: linha for linha in csv.DictReader(io.StringIO(texto))}

    assert linhas["ok"]["cached_input_tokens"] == "8"
    assert linhas["falhou"]["cached_input_tokens"] == ""


async def test_a_linha_de_relay_vem_com_kind_e_sem_medidas(store_with_relay):
    """RED 1: o relay e rastro de rede, e nao uma chamada de modelo.

    Tokens, TTFT e cache nao existem para ele. O banco guarda valores de
    proposito (0 nos tokens NOT NULL, reais nas colunas `nullable`), e a API e
    quem zera tudo olhando o `kind`: sem isso a linha afirmaria "oito tokens em
    cache" sobre um trafego de rede que nunca cacheou nada.
    """
    eventos = {
        e["request_id"]: e for e in body_of(await audit.search_requests(with_params(store_with_relay)))[
            "events"
        ]
    }
    relay = eventos["repasse"]

    assert relay["kind"] == "relay"
    # As CINCO chaves de medicao: as duas NOT NULL chegam como 0 e as outras
    # como valores reais no banco, e nenhuma pode vazar para a tela.
    assert relay["input_tokens"] is None
    assert relay["output_tokens"] is None
    assert relay["ttft_ms"] is None
    assert relay["cached_input_tokens"] is None
    assert relay["cache_write_tokens"] is None
    # A linha de modelo continua medindo, e continua sendo `model`.
    assert eventos["ok"]["kind"] == "model"
    assert eventos["ok"]["input_tokens"] == 10


async def test_o_filtro_kind_seleciona_o_tipo_de_trafego(store_with_relay):
    """RED 2: `kind` filtra a busca; valor desconhecido e filtro ausente."""
    ids = lambda pagina: [e["request_id"] for e in pagina["events"]]

    so_relay = body_of(await audit.search_requests(with_params(store_with_relay, kind="relay")))
    assert ids(so_relay) == ["repasse"]

    so_modelo = body_of(await audit.search_requests(with_params(store_with_relay, kind="model")))
    assert sorted(ids(so_modelo)) == ["falhou", "ok"]

    # `qualquer` nao e um tipo que existe: devolve tudo, e nao um 4xx.
    todos = await audit.search_requests(with_params(store_with_relay, kind="qualquer"))
    assert todos.status_code == 200
    assert sorted(ids(body_of(todos))) == ["falhou", "ok", "repasse"]


async def test_o_csv_leva_a_coluna_kind_e_deixa_as_medidas_em_branco(store_with_relay):
    """RED 3: a planilha do relay traz `kind` e a celula de tokens VAZIA."""
    exportado = await audit.export_requests(with_params(store_with_relay))
    partes = [parte async for parte in exportado.body_iterator]
    texto = "".join(p if isinstance(p, str) else p.decode() for p in partes)
    leitor = csv.DictReader(io.StringIO(texto))

    assert "kind" in leitor.fieldnames
    linhas = {linha["request_id"]: linha for linha in leitor}
    assert linhas["repasse"]["kind"] == "relay"
    assert linhas["repasse"]["input_tokens"] == ""
    assert linhas["ok"]["kind"] == "model"


async def test_a_analise_filha_de_kind_desconhecido_nao_quebra(
    store_with_relay, duble_do_modelo
):
    """RED 4: o dossie nao entende `kind`, e o POST /api/analysis nao pode 500.

    O dossie recebe `kind="model"` FIXO (`app/stats/dossier.py:345`), e `kind`
    nao esta em `FILTER_FIELDS`, que recusa filtro desconhecido com `ValueError`.
    O `pop` em `_analysis_filters` tira a chave ANTES dela chegar la.
    """
    # Pre-condicao: sem o `kind` em `filters_from` este teste nao prova nada -- o
    # dossie nunca veria a chave, e o 500 seria impossivel por outro motivo.
    assert "kind" in audit.filters_from({"kind": "relay"})

    request = with_params(store_with_relay, kind="relay", route="/api/oauth/usage")
    request.app.state.settings = SETTINGS.model_copy(update={"default_model": "cheap"})
    request.app.state.pool = None
    # POST sem corpo, o caminho normal do botao: `payload vazio` (nao
    # exercita o ValueError de `_body_of` -- o dossie so nao recebe chaves).
    request.json = _json({})

    resposta = await audit.analyse_period(request)

    assert resposta.status_code == 200
    corpo = body_of(resposta)
    # O dossie ecoa so o que ele entende: `kind` nao aparece no recorte...
    filtros = corpo["dossier"]["period"]["filters"]
    assert "kind" not in filtros
    # ...mas ecoa de fato o resto que a tela pediu. Sem isto, um `pop` a mais
    # (tirar `route`, por exemplo) passaria despercebido.
    assert filtros["route"] == "/api/oauth/usage"
