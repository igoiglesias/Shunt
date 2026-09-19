"""As consultas do painel, contra uma tabela semeada com numeros conhecidos.

Cada asserticao e um numero exato. "Maior que zero" passaria com a consulta
somando a coluna errada.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.stats import queries
from app.stats.models import RequestEvent

NOW = datetime.now(UTC)


def row(**over):
    base = {
        "request_id": "r",
        "started_at": NOW - timedelta(minutes=5),
        "route": "/v1/messages",
        "dialect": "anthropic",
        "stream": False,
        "requested_model": "claude-haiku-4-5",
        "rule": "family",
        "matched": "haiku",
        "provider": "groq",
        "candidate_model": "gpt-oss-120b",
        "status": 200,
        "error_type": None,
        "input_tokens": 10,
        "output_tokens": 5,
        "ttft_ms": None,
        "duration_ms": 100,
        "attempts": [],
        "fell_back": False,
        "tools_offered": [],
        "tools_called": [],
        "thinking_blocks": 0,
    }
    base.update(over)
    return RequestEvent(**base)


@pytest.fixture
def seeded(make_engine, tmp_path):
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'stats.db'}")
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="a", duration_ms=100, input_tokens=10, output_tokens=5),
                row(request_id="b", duration_ms=200, input_tokens=20, output_tokens=10),
                row(
                    request_id="c",
                    duration_ms=300,
                    status=429,
                    error_type="rate_limit_error",
                    provider="openrouter",
                    candidate_model="vendor/free",
                ),
                row(
                    request_id="d",
                    stream=True,
                    ttft_ms=50,
                    duration_ms=400,
                    attempts=["free: 400 (attempt 1)"],
                    fell_back=True,
                    tools_offered=["read", "write"],
                    tools_called=["read"],
                    thinking_blocks=2,
                ),
                # Fora da janela de 1 hora, e dentro da de 24.
                row(request_id="antigo", started_at=NOW - timedelta(hours=5), duration_ms=999),
            ]
        )
        session.commit()
    return engine


def test_totals_count_only_what_is_inside_the_window(seeded):
    one_hour = queries.totals(seeded, hours=1)
    assert one_hour["requests"] == 4
    assert one_hour["input_tokens"] == 10 + 20 + 10 + 10
    assert one_hour["output_tokens"] == 5 + 10 + 5 + 5
    assert one_hour["errors"] == 1
    assert one_hour["error_rate"] == 0.25
    assert one_hour["fallbacks"] == 1
    assert one_hour["streams"] == 1
    assert queries.totals(seeded, hours=24)["requests"] == 5


def test_the_latency_percentiles_are_observed_values(seeded):
    totals = queries.totals(seeded, hours=1)
    assert totals["p50_duration_ms"] in (200, 300)
    assert totals["p95_duration_ms"] == 400
    assert totals["p50_ttft_ms"] == 50


def test_an_empty_window_answers_zeros_and_no_percentile(make_engine, tmp_path):
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'vazio.db'}")
    totals = queries.totals(engine)
    assert totals["requests"] == 0
    assert totals["error_rate"] == 0.0
    assert totals["p50_duration_ms"] is None


def test_the_series_has_one_bucket_per_hour_that_saw_traffic(seeded):
    series = queries.per_hour(seeded, hours=24)
    assert len(series) == 2
    assert sum(point["requests"] for point in series) == 5
    assert series[0]["hour"] < series[1]["hour"]


def test_grouping_by_provider_splits_requests_and_errors(seeded):
    groups = {group["provider"]: group for group in queries.by_provider(seeded, hours=1)}
    assert groups["groq"]["requests"] == 3
    assert groups["openrouter"]["requests"] == 1
    assert groups["openrouter"]["errors"] == 1
    assert groups["groq"]["errors"] == 0


def test_grouping_by_model_counts_tokens_of_both_directions(seeded):
    groups = {group["model"]: group for group in queries.by_model(seeded, hours=1)}
    assert groups["gpt-oss-120b"]["tokens"] == (10 + 5) + (20 + 10) + (10 + 5)


def test_the_requested_model_is_a_different_question_from_the_served_one(seeded):
    asked = queries.by_requested_model(seeded, hours=1)
    assert asked[0]["requested_model"] == "claude-haiku-4-5"
    assert asked[0]["requests"] == 4


def test_routes_are_listed_with_their_own_traffic(make_engine, tmp_path):
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'rotas.db'}")
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="m", route="/v1/messages"),
                row(request_id="t", route="/v1/messages/count_tokens", candidate_model=None),
                row(request_id="l", route="/v1/models", candidate_model=None),
            ]
        )
        session.commit()
    routes = {group["route"]: group["requests"] for group in queries.by_route(engine)}
    assert routes == {"/v1/messages": 1, "/v1/messages/count_tokens": 1, "/v1/models": 1}


def test_errors_are_grouped_by_status_and_type(seeded):
    errors = queries.errors_by_type(seeded, hours=1)
    assert errors == [{"status": 429, "error_type": "rate_limit_error", "requests": 1}]


def test_chain_health_separates_a_clean_answer_from_a_fallback(seeded):
    chain = queries.chain_health(seeded, hours=1)
    assert chain["requests"] == 4
    assert chain["first_candidate_answered"] == 3
    assert chain["first_candidate_rate"] == 0.75
    assert chain["skips"] == [{"candidate": "free", "count": 1}]


def test_tools_are_listed_by_what_was_actually_called(make_engine, tmp_path):
    """Um harness oferece o catalogo inteiro toda vez.

    Ordenar por oferta enche a tabela com as ferramentas que ninguem usou --
    medido no painel ao vivo: onze linhas, todas com 0%. Quem foi chamada vem
    primeiro.
    """
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'ordem-tools.db'}")
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="a", tools_offered=["Read", "Bash", "Edit"], tools_called=["Edit"]),
                row(request_id="b", tools_offered=["Read", "Bash", "Edit"], tools_called=["Edit"]),
                row(request_id="c", tools_offered=["Read", "Bash", "Edit"], tools_called=["Bash"]),
            ]
        )
        session.commit()
    tools = queries.tool_usage(engine, limit=2)
    assert [tool["tool"] for tool in tools] == ["Edit", "Bash"]
    assert tools[0]["called"] == 2


def test_tool_usage_shows_what_was_offered_against_what_was_called(seeded):
    tools = {tool["tool"]: tool for tool in queries.tool_usage(seeded, hours=1)}
    assert tools["read"] == {"tool": "read", "offered": 1, "called": 1, "call_rate": 1.0}
    assert tools["write"]["called"] == 0
    assert tools["write"]["call_rate"] == 0.0


def test_a_tool_called_without_ever_being_offered_still_shows_up(make_engine, tmp_path):
    """Modelo que inventa ferramenta e coisa que se quer ver, nao esconder."""
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'tools.db'}")
    with Session(engine) as session:
        session.add(row(request_id="x", tools_called=["inventada"]))
        session.commit()
    tools = queries.tool_usage(engine)
    assert tools == [{"tool": "inventada", "offered": 0, "called": 1, "call_rate": None}]


def test_recent_comes_back_newest_first_and_capped(seeded):
    recent = queries.recent(seeded, limit=2)
    assert len(recent) == 2
    assert recent[0]["request_id"] == "antigo"
    assert recent[0]["started_at"] is not None


def test_the_snapshot_carries_every_section_the_panel_draws(seeded):
    snapshot = queries.snapshot(seeded, hours=1)
    assert set(snapshot) == {
        "window_hours",
        "totals",
        "per_hour",
        "by_model",
        "by_provider",
        "by_route",
        "by_requested_model",
        "pairs",
        "errors",
        "chain",
        "tools",
        "recent",
    }
    assert snapshot["window_hours"] == 1


def test_the_percentile_sorts_before_it_picks(make_engine, tmp_path):
    """Mutante M11: sem `sorted`, a suite passava.

    As linhas semeadas estavam em ordem crescente de duracao por acidente, e o
    percentil por indice acertava sem ordenar nada. Aqui a ordem de insercao e
    embaralhada de proposito: a mediana de 10,20,...,90 e 50, venha na ordem
    que vier.
    """
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'ordem.db'}")
    durations = [90, 10, 50, 70, 30, 80, 20, 60, 40]
    with Session(engine) as session:
        session.add_all(
            [row(request_id=f"d{value}", duration_ms=value) for value in durations]
        )
        session.commit()
    totals = queries.totals(engine, hours=1)
    assert totals["p50_duration_ms"] == 50
    assert totals["p95_duration_ms"] == 90


def test_a_route_that_asks_for_no_model_is_not_a_requested_model(make_engine, tmp_path):
    """Medido com trafego real: `/v1/models` grava modelo vazio.

    Vazio nao e NULL, entao o filtro de nulos deixava passar uma linha sem nome
    no diagrama do painel.
    """
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'vazio.db'}")
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="m", requested_model="claude-haiku-4-5"),
                row(request_id="l", route="/v1/models", requested_model="", candidate_model=None),
            ]
        )
        session.commit()
    asked = queries.by_requested_model(engine)
    assert [group["requested_model"] for group in asked] == ["claude-haiku-4-5"]
    # A rota em si continua contada: o que sai e so o "modelo" que nao existe.
    assert len(queries.by_route(engine)) == 2


def test_a_stored_instant_comes_back_saying_it_is_utc(make_engine, tmp_path):
    """O SQLite devolve `datetime` ingenuo mesmo em coluna com `timezone=True`.

    Medido no painel: a fita misturava o ISO sem offset do banco com o ISO em
    UTC do SSE, lia o primeiro como hora local e saia tres horas fora de ordem.
    """
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'fuso.db'}")
    with Session(engine) as session:
        session.add(row(request_id="z", started_at=datetime(2026, 9, 19, 8, 30, tzinfo=UTC)))
        session.commit()
    (stored,) = queries.recent(engine)
    assert stored["started_at"].endswith("+00:00")
    assert stored["started_at"].startswith("2026-09-19T08:30")


def test_a_missing_instant_comes_back_as_null_instead_of_raising():
    from app.stats.queries import _utc

    assert _utc(None) is None
    assert _utc(datetime(2026, 9, 19, 8, 30, tzinfo=UTC)).endswith("+00:00")


def test_clearing_removes_everything_and_says_how_much(seeded):
    assert queries.totals(seeded, hours=24)["requests"] == 5
    assert queries.delete_events(seeded) == 5
    assert queries.totals(seeded, hours=24)["requests"] == 0
    assert queries.delete_events(seeded) == 0, "limpar duas vezes nao inventa linhas"


def test_clearing_only_the_past_keeps_what_is_recent(seeded):
    """Zerar o passado sem perder o que esta acontecendo agora."""
    removed = queries.delete_events(seeded, older_than_hours=1)
    assert removed == 1, "so a linha de cinco horas atras era antiga"
    assert queries.totals(seeded, hours=24)["requests"] == 4


def test_clearing_takes_the_conversation_with_it(make_engine, tmp_path):
    """Texto sem o evento dele fica invisivel para sempre na tela de auditoria."""
    from sqlalchemy import func, select

    from app.stats.models import RequestBody

    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'com-corpo.db'}")
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="velha", started_at=NOW - timedelta(hours=5)),
                row(request_id="nova"),
                RequestBody(request_id="velha", prompt="antiga", answer="ok"),
                RequestBody(request_id="nova", prompt="recente", answer="ok"),
            ]
        )
        session.commit()

    queries.delete_events(engine, older_than_hours=1)
    with Session(engine) as session:
        restantes = session.scalars(select(RequestBody.request_id)).all()
    assert restantes == ["nova"]

    queries.delete_events(engine)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(RequestBody)) == 0


def test_the_tool_table_stops_at_eight_rows(make_engine, tmp_path):
    """Um harness oferece o catalogo inteiro; a nona nunca foi chamada.

    Este teste existe porque o limite se perdeu uma vez ao restaurar arquivos
    depois de uma varredura de mutacao, e so o painel ao vivo mostrou.
    """
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'muitas.db'}")
    catalogo = [f"Ferramenta{i:02d}" for i in range(15)]
    with Session(engine) as session:
        session.add_all(
            [row(request_id=f"r{i}", tools_offered=catalogo, tools_called=[]) for i in range(3)]
        )
        session.commit()
    assert len(queries.tool_usage(engine)) == 8
    assert len(queries.snapshot(engine)["tools"]) == 8


def test_the_pair_says_which_request_landed_on_which_model(make_engine, tmp_path):
    """As duas listas separadas nao dizem se foi o opus que caiu no local."""
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'pares.db'}")
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="a", requested_model="claude-opus-5", candidate_model="qwen3.8-27b"),
                row(request_id="b", requested_model="claude-opus-5", candidate_model="qwen3.8-27b"),
                row(request_id="c", requested_model="claude-opus-5", candidate_model="gpt-oss-120b"),
                row(request_id="d", requested_model="claude-haiku-4-5", candidate_model="gpt-oss-120b"),
                # Recusada: sem candidato, nao e um par.
                row(request_id="e", requested_model="claude-opus-5", candidate_model=None),
                # Listagem: sem modelo pedido, tambem nao.
                row(request_id="f", requested_model="", candidate_model=None, route="/v1/models"),
            ]
        )
        session.commit()
    pares = queries.pairs(engine)
    assert pares[0] == {
        "asked": "claude-opus-5",
        "served": "qwen3.8-27b",
        "requests": 2,
        "tokens": 30,
    }
    assert len(pares) == 3
    assert all(par["served"] for par in pares)
