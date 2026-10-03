"""As consultas do painel, contra uma tabela semeada com numeros conhecidos.

Cada asserticao e um numero exato. "Maior que zero" passaria com a consulta
somando a coluna errada.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.config.settings import ModelCaps, ModelConfig, ProviderConfig, Settings
from app.core import dispatcher
from app.core.dispatcher import ShuntRequest
from app.core.resolver import Candidate, Resolution
from app.stats import queries
from app.stats.models import RequestEvent

# Congela relógio do módulo queries para evitar flake de janela temporal.
# _since() usa datetime.now(UTC) inline; os offsets dos testes partem de NOW,
# então o instante congelado tem de ser o MESMO que NOW, senão as linhas
# semeadas saem da janela e a consulta conta zero.
FROZEN_NOW = datetime.now(UTC)


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FROZEN_NOW


@pytest.fixture(autouse=True)
def _freeze_clock(monkeypatch):
    monkeypatch.setattr(queries, "datetime", _FrozenDatetime)


# O candidato que a rota resolveu para a linha de pulo do teste cruzado.
NEMOTRON_CHAIN = [
    Candidate(
        alias="nemotron",
        provider="openrouter",
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        protocol="openai",
    )
]

NOW = FROZEN_NOW


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
        "input_tokens": None,
        "output_tokens": None,
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


def relay_row(**over):
    """Linha de relay, com valores extremos de proposito: se vazar em um
    agregado, ela mexe em erro, p95, cadeia, ferramentas e facetas ao mesmo
    tempo."""
    return row(
        kind="relay",
        requested_model="",
        rule="none",
        matched=None,
        provider=None,
        candidate_model=None,
        route="/api/oauth/usage",
        dialect="anthropic",
        status=502,
        error_type="upstream_unreachable",
        duration_ms=99999,
        attempts=["x: 502 (attempt 1)"],
        tools_offered=["read"],
        tools_called=["read"],
        **over,
    )


def _base_rows():
    return [
        row(request_id="a", duration_ms=100, input_tokens=10, output_tokens=5),
        row(request_id="b", duration_ms=200, input_tokens=20, output_tokens=10),
        row(
            request_id="c",
            duration_ms=300,
            input_tokens=10,
            output_tokens=5,
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
            input_tokens=10,
            output_tokens=5,
            attempts=["free: 400 (attempt 1)"],
            fell_back=True,
            tools_offered=["read", "write"],
            tools_called=["read"],
            thinking_blocks=2,
        ),
        # Fora da janela de 1 hora, e dentro da de 24.
        row(request_id="antigo", started_at=NOW - timedelta(hours=5), duration_ms=999),
    ]


@pytest.fixture
def seeded(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(_base_rows())
        session.commit()
    return engine


@pytest.fixture
def seeded_with_relay(make_engine):
    """A janela de modelo com uma linha de relay atravesada.

    O relay entra DENTRO da janela de 1 hora de proposito: fora dela nenhum
    agregado o veria, e o teste nao provaria nada.
    """
    engine = make_engine()
    with Session(engine) as session:
        session.add_all([*_base_rows(), relay_row(request_id="relay")])
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


def test_an_empty_window_answers_zeros_and_no_percentile(make_engine):
    engine = make_engine()
    totals = queries.totals(engine)
    assert totals["requests"] == 0
    assert totals["error_rate"] == 0.0
    assert totals["p50_duration_ms"] is None


def test_the_series_buckets_follow_the_window(seeded):
    """Cinco minutos agrupados por HORA sao um balde so: foi o que o painel
    mostrou, uma barra solitaria no meio do nada."""
    assert queries.bucket_minutes(0.0833) == 1
    assert queries.bucket_minutes(1) == 1
    assert queries.bucket_minutes(6) == 5
    assert queries.bucket_minutes(24) == 60
    assert queries.bucket_minutes(24 * 30) == 60 * 24

    # A janela e o teto; o balde segue o intervalo que o trafego ocupa. As
    # linhas semeadas cabem em cinco horas, entao 24 h nao vira balde de hora.
    dia = queries.series(seeded, hours=24)
    assert dia["bucket_minutes"] == 5
    assert sum(point["requests"] for point in dia["points"]) == 5
    assert dia["points"][0]["at"] < dia["points"][-1]["at"]


def test_a_short_window_gets_a_bucket_per_minute(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="a", started_at=NOW - timedelta(minutes=1)),
                row(request_id="b", started_at=NOW - timedelta(minutes=1)),
                row(request_id="c", started_at=NOW - timedelta(minutes=4)),
                row(request_id="d", started_at=NOW - timedelta(minutes=9)),
            ]
        )
        session.commit()
    curta = queries.series(engine, hours=0.25)
    assert curta["bucket_minutes"] == 1
    assert [p["requests"] for p in curta["points"]] == [1, 1, 2]
    assert all(len(p["at"]) == len("2026-09-19T12:34") for p in curta["points"])


def test_a_medium_window_groups_by_five_minutes(make_engine):
    engine = make_engine()
    base = (NOW - timedelta(hours=3)).replace(minute=32, second=0, microsecond=0)
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="a", started_at=base),
                row(request_id="b", started_at=base + timedelta(minutes=1)),
                row(request_id="c", started_at=base + timedelta(minutes=6)),
                # Estica o intervalo para tres horas, que e o que escolhe o balde.
                row(request_id="d", started_at=NOW - timedelta(minutes=2)),
            ]
        )
        session.commit()
    media = queries.series(engine, hours=6)
    assert media["bucket_minutes"] == 5
    assert [p["requests"] for p in media["points"]] == [2, 1, 1]
    assert media["points"][0]["at"].endswith(":30")
    assert media["points"][1]["at"].endswith(":35")


def test_a_long_window_groups_in_blocks_of_hours(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="a", started_at=NOW - timedelta(days=1)),
                row(request_id="b", started_at=NOW - timedelta(days=9)),
            ]
        )
        session.commit()
    longa = queries.series(engine, hours=24 * 14)
    assert longa["bucket_minutes"] == 60 * 6
    assert len(longa["points"]) == 2
    assert all(p["at"].endswith(":00") for p in longa["points"])


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


def test_routes_are_listed_with_their_own_traffic(make_engine):
    engine = make_engine()
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
    assert chain["skips"] == [{"candidate": "free", "reason": "falhou na chamada", "count": 1}]


def test_tools_are_listed_by_what_was_actually_called(make_engine):
    """Um harness oferece o catalogo inteiro toda vez.

    Ordenar por oferta enche a tabela com as ferramentas que ninguem usou --
    medido no painel ao vivo: onze linhas, todas com 0%. Quem foi chamada vem
    primeiro.
    """
    engine = make_engine()
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


def test_a_tool_called_without_ever_being_offered_still_shows_up(make_engine):
    """Modelo que inventa ferramenta e coisa que se quer ver, nao esconder."""
    engine = make_engine()
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
        "series",
        "by_model",
        "by_provider",
        "by_route",
        "by_requested_model",
        "by_project",
        "pairs",
        "errors",
        "chain",
        "tools",
        "recent",
    }
    assert snapshot["window_hours"] == 1


def test_the_percentile_sorts_before_it_picks(make_engine):
    """Mutante M11: sem `sorted`, a suite passava.

    As linhas semeadas estavam em ordem crescente de duracao por acidente, e o
    percentil por indice acertava sem ordenar nada. Aqui a ordem de insercao e
    embaralhada de proposito: a mediana de 10,20,...,90 e 50, venha na ordem
    que vier.
    """
    engine = make_engine()
    durations = [90, 10, 50, 70, 30, 80, 20, 60, 40]
    with Session(engine) as session:
        session.add_all(
            [row(request_id=f"d{value}", duration_ms=value) for value in durations]
        )
        session.commit()
    totals = queries.totals(engine, hours=1)
    assert totals["p50_duration_ms"] == 50
    assert totals["p95_duration_ms"] == 90


def test_a_route_that_asks_for_no_model_is_not_a_requested_model(make_engine):
    """Medido com trafego real: `/v1/models` grava modelo vazio.

    Vazio nao e NULL, entao o filtro de nulos deixava passar uma linha sem nome
    no diagrama do painel.
    """
    engine = make_engine()
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


def test_a_stored_instant_comes_back_saying_it_is_utc(make_engine):
    """O SQLite devolve `datetime` ingenuo mesmo em coluna com `timezone=True`.

    Medido no painel: a fita misturava o ISO sem offset do banco com o ISO em
    UTC do SSE, lia o primeiro como hora local e saia tres horas fora de ordem.
    """
    engine = make_engine()
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


def test_clearing_takes_the_conversation_with_it(make_engine):
    """Texto sem o evento dele fica invisivel para sempre na tela de auditoria."""
    from sqlalchemy import func, select

    from app.stats.models import RequestBody

    engine = make_engine()
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


def test_the_tool_table_stops_at_eight_rows(make_engine):
    """Um harness oferece o catalogo inteiro; a nona nunca foi chamada.

    Este teste existe porque o limite se perdeu uma vez ao restaurar arquivos
    depois de uma varredura de mutacao, e so o painel ao vivo mostrou.
    """
    engine = make_engine()
    catalogo = [f"Ferramenta{i:02d}" for i in range(15)]
    with Session(engine) as session:
        session.add_all(
            [row(request_id=f"r{i}", tools_offered=catalogo, tools_called=[]) for i in range(3)]
        )
        session.commit()
    assert len(queries.tool_usage(engine)) == 8
    assert len(queries.snapshot(engine)["tools"]) == 8


def test_the_pair_says_which_request_landed_on_which_model(make_engine):
    """As duas listas separadas nao dizem se foi o opus que caiu no local."""
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="a", requested_model="claude-opus-5", candidate_model="qwen3.8-27b", input_tokens=10, output_tokens=5),
                row(request_id="b", requested_model="claude-opus-5", candidate_model="qwen3.8-27b", input_tokens=10, output_tokens=5),
                row(request_id="c", requested_model="claude-opus-5", candidate_model="gpt-oss-120b", input_tokens=15, output_tokens=5),
                row(request_id="d", requested_model="claude-haiku-4-5", candidate_model="gpt-oss-120b", input_tokens=15, output_tokens=5),
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


def test_the_bucket_follows_the_traffic_and_not_only_the_window(make_engine):
    """Medido no painel real: 72 requisicoes em 44 minutos dentro de uma janela
    de 24 horas davam DUAS barras."""
    engine = make_engine()
    base = NOW - timedelta(minutes=44)
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id=f"r{i}", started_at=base + timedelta(minutes=i * 2))
                for i in range(22)
            ]
        )
        session.commit()
    largo = queries.series(engine, hours=24)
    assert largo["bucket_minutes"] == 1, "o balde ignorou o intervalo real do trafego"
    assert len(largo["points"]) == 22


def test_traffic_spread_over_days_still_gets_a_coarse_bucket(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id=f"d{i}", started_at=NOW - timedelta(hours=i * 20))
                for i in range(1, 8)
            ]
        )
        session.commit()
    largo = queries.series(engine, hours=24 * 14)
    assert largo["bucket_minutes"] >= 60
    assert len(largo["points"]) <= 14


# -- Historia E: o painel separa o descarte por motivo -------------------------


def test_skips_are_grouped_by_reason(make_engine):
    """"Pulado por nao caber" e "pulado por estar fora do ar" pedem decisoes
    opostas do operador: um mexe no catalogo, o outro no provedor."""
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="a", attempts=["groq-free: context window too small (175k > 131k)"]),
                row(request_id="b", attempts=["groq-free: context window too small (180k > 131k)"]),
                row(request_id="c", attempts=["qwen-local: ConnectError"]),
                row(request_id="d", attempts=["free: credential OPENROUTER_API_KEY is not set"]),
                row(request_id="e", attempts=["free: no tool support"]),
            ]
        )
        session.commit()
    chain = queries.chain_health(engine, hours=1)
    por_motivo = {(s["candidate"], s["reason"]): s["count"] for s in chain["skips"]}
    assert por_motivo[("groq-free", "não coube")] == 2
    assert por_motivo[("qwen-local", "falhou na chamada")] == 1
    assert por_motivo[("free", "sem credencial")] == 1
    assert por_motivo[("free", "sem suporte")] == 1


def test_the_last_resort_note_is_not_counted_as_a_skip(make_engine):
    """A linha que diz "fui chamado assim mesmo" e o contrario de um pulo."""
    engine = make_engine()
    with Session(engine) as session:
        session.add(
            row(
                request_id="a",
                attempts=[
                    "curto: context window too small (2000 > 50)",
                    "curto: taken anyway, nothing in the chain fits",
                ],
            )
        )
        session.commit()
    chain = queries.chain_health(engine, hours=1)
    assert [(s["candidate"], s["reason"], s["count"]) for s in chain["skips"]] == [
        ("curto", "não coube", 1)
    ]


def test_a_model_name_with_a_colon_is_not_cut_and_still_classifies(make_engine):
    """A linha e escrita como `<modelo>: <motivo>`, e nome de modelo tem dois-pontos.

    `nvidia/nemotron-3-ultra-550b-a55b:free` (catalogo real) cortado no primeiro
    dois-pontos vira `nvidia/nemotron-3-ultra-550b-a55b` -- nome truncado -- e o
    motivo que sobra nao casa com nenhuma regra, entao "sem credencial" e
    "nao coube" caiam os dois na classe "falhou na chamada". Sao decisoes
    opostas do operador colapsadas numa contagem so.
    """
    modelo = "nvidia/nemotron-3-ultra-550b-a55b:free"
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(
                    request_id="a",
                    attempts=[f"{modelo}: credential OPENROUTER_API_KEY is not set"],
                ),
                row(
                    request_id="b",
                    attempts=[f"{modelo}: context window too small (300000 > 262144)"],
                ),
            ]
        )
        session.commit()

    chain = queries.chain_health(engine, hours=1)
    por_motivo = {(s["candidate"], s["reason"]): s["count"] for s in chain["skips"]}

    assert por_motivo[(modelo, "sem credencial")] == 1
    assert por_motivo[(modelo, "não coube")] == 1


def test_the_colon_parsing_survives_a_skip_row_written_by_the_dispatcher(make_engine):
    """RED->GREEN cruzado: a linha de pulo REAL, escrita pelo dispatcher.

    Sem o cruzamento, os dois lados passariam a discordar em silencio -- o
    dispatcher nomeando pelo modelo e a consulta cortando o nome no primeiro
    dois-pontos.
    """
    settings = Settings(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://api.test/v1", protocol="openai", api_key=None
            )
        },
        models={
            "nemotron": ModelConfig(
                provider="openrouter",
                model="nvidia/nemotron-3-ultra-550b-a55b:free",
                supports=ModelCaps(vision=False),
                context_window=262144,
                max_output_tokens=8192,
            )
        },
        routes=[("opus", ["nemotron"])],
        default_model=None,
    )
    req = ShuntRequest(
        "anthropic",
        {
            "model": "claude-opus-4-5",
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}}
                    ],
                }
            ],
        },
        {},
    )
    _cadeia, trace = dispatcher._chain_for(req, settings, Resolution("opus", None, NEMOTRON_CHAIN), req.body)

    assert trace == ["nvidia/nemotron-3-ultra-550b-a55b:free: no vision support"]

    engine = make_engine()
    with Session(engine) as session:
        session.add(row(request_id="a", attempts=trace))
        session.commit()

    chain = queries.chain_health(engine, hours=1)
    assert [(s["candidate"], s["reason"]) for s in chain["skips"]] == [
        ("nvidia/nemotron-3-ultra-550b-a55b:free", "sem suporte")
    ]


# --- Taxa de geracao (tokens de saida por segundo) ---------------------------
#
# A taxa e AGRUPADA -- soma de tokens sobre soma de tempo -- e nao a mediana da
# taxa de cada requisicao: uma requisicao de 22 tokens tem de pesar 22 tokens, e
# nao um voto inteiro igual ao de uma de 8.000.


def test_a_taxa_soma_tokens_sobre_tempo_de_geracao(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                # Streaming: o tempo ate o primeiro token e espera, nao geracao.
                row(request_id="s", stream=True, ttft_ms=900, duration_ms=30000,
                    input_tokens=100, output_tokens=1200, provider="local"),
                # Sem streaming nao da para separar: a duracao inteira conta.
                row(request_id="n", stream=False, duration_ms=300, input_tokens=50,
                    output_tokens=50, provider="local"),
                # Sem saida nao entra na conta -- nem em cima nem embaixo.
                row(request_id="erro", status=500, duration_ms=80, output_tokens=0,
                    provider="local"),
            ]
        )
        session.commit()

    linha = queries.by_provider(engine)[0]

    assert linha["rated_requests"] == 2
    assert linha["generation_ms"] == 29400
    assert linha["tokens_per_second"] == 42.5


def test_stream_sem_primeiro_token_fica_fora_da_taxa(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        # Streaming sem `ttft_ms`: nao ha como separar espera de geracao, e
        # contar a duracao inteira chamaria a espera de geracao.
        session.add_all(
            [
                row(request_id="s", stream=True, ttft_ms=None, duration_ms=1000,
                    input_tokens=100, output_tokens=500, provider="local"),
                row(request_id="n", stream=False, duration_ms=1000, input_tokens=50,
                    output_tokens=100, provider="local"),
            ]
        )
        session.commit()

    linha = queries.by_provider(engine)[0]

    assert linha["rated_requests"] == 1
    assert linha["tokens_per_second"] == 100.0


def test_grupo_sem_nada_para_medir_responde_none(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        session.add(row(request_id="vazio", output_tokens=0, duration_ms=120))
        session.commit()

    linha = queries.by_model(engine)[0]

    # Zero e uma AFIRMACAO sobre velocidade; None diz que nao houve o que medir.
    assert linha["tokens_per_second"] is None
    assert linha["rated_requests"] == 0


def test_duracao_zero_nao_divide_por_zero(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="instantanea", duration_ms=0, input_tokens=10, output_tokens=40),
                row(request_id="normal", duration_ms=500, input_tokens=10, output_tokens=50),
            ]
        )
        session.commit()

    linha = queries.by_model(engine)[0]

    assert linha["rated_requests"] == 1
    assert linha["tokens_per_second"] == 100.0


def test_provedor_mudo_em_input_tokens_fica_fora_da_taxa(make_engine):
    """Silencio nao e medicao: `input_tokens` NULL exclui so a linha muda.

    O filtro do `_usage_reported` (queries.py) e o coracao da Task 2: sem ele,
    uma requisicao em que o provedor nao disse nada entra na taxa com a
    output_tokens que registramos e puxa a soma para baixo. A linha mutuamente
    exclusiva (provedor que fala) fica para provar que o NULL derruba a LINHA e
    nao o grupo.
    """
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                # Mudo: informou saida, mas nunca disse quanto entrou.
                row(request_id="mudo", provider="mudo", stream=False, duration_ms=400,
                    input_tokens=None, output_tokens=1200),
                # Fala: mesma janela e mesmo provedor, para o grupo existir.
                row(request_id="falante", provider="mudo", stream=False, duration_ms=400,
                    input_tokens=100, output_tokens=1200),
            ]
        )
        session.commit()

    linha = {r["provider"]: r for r in queries.by_provider(engine)}["mudo"]

    assert linha["rated_requests"] == 1
    assert linha["tokens_per_second"] == 3000.0


def test_a_janela_inteira_tem_a_propria_taxa(seeded):
    totais = queries.totals(seeded, hours=1)

    # a: 5 tokens / 100ms, b: 10 / 200ms, c: 5 / 300ms, d: 5 / (400-50)ms.
    # Somados: 25 tokens em 950ms.
    assert totais["rated_requests"] == 4
    assert totais["generation_ms"] == 950
    assert totais["tokens_per_second"] == 26.3


def test_o_modelo_carrega_o_proprio_provedor(make_engine):
    """Sem isso a tela pareava modelo e provedor por POSICAO nas duas listas."""
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="1", candidate_model="qwen", provider="local"),
                row(request_id="2", candidate_model="qwen", provider="local"),
                row(request_id="3", candidate_model="oss", provider="groq"),
            ]
        )
        session.commit()

    por_modelo = {linha["model"]: linha["provider"] for linha in queries.by_model(engine)}

    assert por_modelo == {"qwen": "local", "oss": "groq"}


def test_modelo_servido_por_dois_provedores_diz_varios(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="1", candidate_model="oss", provider="groq"),
                row(request_id="2", candidate_model="oss", provider="openrouter"),
            ]
        )
        session.commit()

    # Escolher um dos dois seria inventar; dizer "vários" e o que se sabe.
    assert queries.by_model(engine)[0]["provider"] == "vários"


def test_por_projeto_mostra_tambem_o_que_nao_tem_projeto(make_engine):
    """A linha "sem projeto" e a medida da cobertura da extracao."""
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="1", project="/home/x/agenda"),
                row(request_id="2", project="/home/x/agenda"),
                row(request_id="3", project="/home/x/shunt"),
                row(request_id="4", project=None),
                row(request_id="5", project=None),
            ]
        )
        session.commit()

    linhas = {linha["name"]: linha["requests"] for linha in queries.by_project(engine)}

    assert linhas == {"agenda": 2, "shunt": 1, "sem projeto": 2}
    # A soma fecha com a janela: esconder os sem projeto daria conta que nao bate.
    assert sum(linhas.values()) == queries.totals(engine)["requests"]


def test_o_projeto_entra_no_evento_e_nas_facetas(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        session.add(row(request_id="1", project="/home/x/agenda", session_id="s1"))
        session.commit()

    evento = queries.search_events(engine)["events"][0]
    facetas = queries.facets(engine)

    assert evento["project"] == "/home/x/agenda"
    assert evento["session_id"] == "s1"
    assert facetas["projects"] == [{"value": "/home/x/agenda", "count": 1}]


def test_a_busca_filtra_e_acha_pelo_projeto(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="um", project="/home/x/agenda"),
                row(request_id="dois", project="/home/x/proxy-shunt"),
            ]
        )
        session.commit()

    exato = queries.search_events(engine, project="/home/x/agenda")
    # O trecho existe SO no caminho do projeto: buscar por "dois" acharia pelo
    # id e nao provaria nada sobre o projeto.
    livre = queries.search_events(engine, text="proxy-shunt")

    assert [e["request_id"] for e in exato["events"]] == ["um"]
    assert [e["request_id"] for e in livre["events"]] == ["dois"]


# --- Cache -------------------------------------------------------------------
#
# Silencio e zero sao coisas diferentes. Medido: o Groq nao manda o campo, o
# OpenRouter manda zero, e o llama.cpp local manda 2814 de 2818. Uma tela que
# chama as tres coisas de "0%" leva a decisao errada -- foi assim que um modelo
# leu este painel e recomendou ligar um cache que ja estava ligado.


def test_a_taxa_de_cache_so_conta_quem_informou(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="c1", provider="local", input_tokens=1000,
                    cached_input_tokens=900),
                row(request_id="c2", provider="local", input_tokens=1000,
                    cached_input_tokens=100),
                # Sem sinal: fica fora das duas somas.
                row(request_id="c3", provider="local", input_tokens=5000,
                    cached_input_tokens=None),
            ]
        )
        session.commit()

    linha = queries.by_provider(engine)[0]

    assert linha["cache_reported_requests"] == 2
    assert linha["cached_input_tokens"] == 1000
    # 1000 de 2000 tokens informados, e nao de 7000.
    assert linha["cache_hit_rate"] == 0.5


def test_provedor_que_nunca_informa_nao_ganha_taxa(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        session.add(row(request_id="mudo", provider="groq", input_tokens=3678))
        session.commit()

    linha = queries.by_provider(engine)[0]

    assert linha["cache_reported_requests"] == 0
    assert linha["cache_hit_rate"] is None


def test_provedor_que_informa_zero_tem_taxa_zero(make_engine):
    """Zero E uma medicao: o provedor disse que nao reaproveitou nada."""
    engine = make_engine()
    with Session(engine) as session:
        session.add(
            row(request_id="frio", provider="openrouter", input_tokens=3620,
                cached_input_tokens=0)
        )
        session.commit()

    linha = queries.by_provider(engine)[0]

    assert linha["cache_reported_requests"] == 1
    assert linha["cache_hit_rate"] == 0.0


def test_a_janela_inteira_tem_a_propria_taxa_de_cache(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="a", input_tokens=2000, cached_input_tokens=1500),
                row(request_id="b", input_tokens=2000, cached_input_tokens=500),
                row(request_id="c", input_tokens=9000),
            ]
        )
        session.commit()

    totais = queries.totals(engine)

    assert totais["cache_reported_requests"] == 2
    assert totais["cached_input_tokens"] == 2000
    assert totais["cache_hit_rate"] == 0.5


# --- A linha de relay nao vaza em agregado -----------------------------------
#
# Cada teste semeia a mesma janela do vizinho ja existente MAIS uma linha de
# relay, e afirma o MESMO numero: se o relay vazar, o numero muda. Os valores
# extremos do `relay_row` fazem a linha aparecer em erro, p95, cadeia,
# ferramentas e facetas ao mesmo tempo.


def test_a_relay_line_does_not_enter_the_totals(seeded_with_relay):
    one_hour = queries.totals(seeded_with_relay, hours=1)
    assert one_hour["requests"] == 4
    assert one_hour["input_tokens"] == 10 + 20 + 10 + 10
    assert one_hour["output_tokens"] == 5 + 10 + 5 + 5
    assert one_hour["errors"] == 1
    assert one_hour["error_rate"] == 0.25
    assert one_hour["fallbacks"] == 1
    assert one_hour["streams"] == 1
    assert one_hour["p95_duration_ms"] == 400
    assert queries.totals(seeded_with_relay, hours=24)["requests"] == 5


def test_a_relay_line_does_not_enter_the_series(seeded_with_relay):
    hora = queries.series(seeded_with_relay, hours=1)
    assert sum(point["requests"] for point in hora["points"]) == 4


def test_a_relay_line_does_not_enter_by_model(seeded_with_relay):
    groups = {group["model"]: group for group in queries.by_model(seeded_with_relay, hours=1)}
    assert groups["gpt-oss-120b"]["tokens"] == (10 + 5) + (20 + 10) + (10 + 5)
    assert groups["gpt-oss-120b"]["requests"] == 3


def test_a_relay_line_does_not_enter_by_provider(seeded_with_relay):
    groups = {
        group["provider"]: group for group in queries.by_provider(seeded_with_relay, hours=1)
    }
    assert groups["groq"]["requests"] == 3
    assert groups["openrouter"]["requests"] == 1
    assert groups["openrouter"]["errors"] == 1
    assert groups["groq"]["errors"] == 0


def test_a_relay_line_does_not_enter_by_route(seeded_with_relay):
    rotas = {group["route"] for group in queries.by_route(seeded_with_relay, hours=1)}
    assert "/api/oauth/usage" not in rotas


def test_a_relay_line_does_not_enter_by_requested_model(seeded_with_relay):
    asked = queries.by_requested_model(seeded_with_relay, hours=1)
    assert asked[0]["requested_model"] == "claude-haiku-4-5"
    assert asked[0]["requests"] == 4


def test_a_relay_line_does_not_grow_the_no_project_line(seeded_with_relay):
    linhas = {
        linha["name"]: linha["requests"] for linha in queries.by_project(seeded_with_relay, hours=1)
    }
    assert linhas["sem projeto"] == 4


def test_a_relay_line_does_not_enter_the_pairs(seeded_with_relay):
    pares = queries.pairs(seeded_with_relay, hours=1)
    assert [
        (par["asked"], par["served"], par["requests"]) for par in pares
    ] == [
        ("claude-haiku-4-5", "gpt-oss-120b", 3),
        ("claude-haiku-4-5", "vendor/free", 1),
    ]


def test_a_relay_line_does_not_enter_errors_by_type(seeded_with_relay):
    errors = queries.errors_by_type(seeded_with_relay, hours=1)
    assert errors == [{"status": 429, "error_type": "rate_limit_error", "requests": 1}]


def test_a_relay_line_does_not_enter_the_chain(seeded_with_relay):
    chain = queries.chain_health(seeded_with_relay, hours=1)
    assert chain["requests"] == 4
    assert chain["first_candidate_answered"] == 3
    assert chain["first_candidate_rate"] == 0.75
    assert chain["skips"] == [{"candidate": "free", "reason": "falhou na chamada", "count": 1}]


def test_a_relay_line_does_not_enter_tool_usage(seeded_with_relay):
    tools = {tool["tool"]: tool for tool in queries.tool_usage(seeded_with_relay, hours=1)}
    assert tools["read"] == {"tool": "read", "offered": 1, "called": 1, "call_rate": 1.0}
    assert tools["write"]["called"] == 0


def test_a_relay_line_does_not_appear_in_recent(seeded_with_relay):
    recent = queries.recent(seeded_with_relay)
    assert [item["request_id"] for item in recent] == ["antigo", "d", "c", "b", "a"]


def test_a_relay_line_does_not_enter_the_facets(seeded_with_relay):
    facetas = queries.facets(seeded_with_relay)
    assert all(item["value"] != "/api/oauth/usage" for item in facetas["routes"])
    assert all(item["value"] != "upstream_unreachable" for item in facetas["error_types"])


def test_a_relay_line_does_not_enter_the_snapshot(seeded_with_relay):
    snapshot = queries.snapshot(seeded_with_relay, hours=1)
    assert snapshot["totals"]["requests"] == 4


def test_lines_without_kind_are_counted_as_model_lines(seeded):
    """Toda linha antiga tem `kind` NULL, e NULL significa modelo.

    Se o filtro fosse `kind != 'relay'` sozinho, a janela inteira -- que e
    historia sem coluna -- somaria zero: medido em SQLite, a comparacao exclui
    o NULL. O `or_` com `is_(None)` e o que mantem essa conta em 4.
    """
    assert queries.totals(seeded, hours=1)["requests"] == 4


# --- B3: a linha parcialmente muda nao pode sumir da soma -------------------
#
# `input_tokens + output_tokens` no SQL e NULL quando qualquer lado e NULL, e
# `sum` de NULL ignora a linha inteira. O caminho e real: `sse_to_openai.py` e
# `sse_to_anthropic.py` gravam so o campo que o provedor enviou, e o outro
# chega NULL. A soma tem que ser POR COLUNA, cada uma com o seu coalesce.


def test_a_partially_muted_line_still_counts_its_reported_side(make_engine):
    """Linha com input NULL e output informado entra com o output na soma.

    Antes do fix, `sum(input_tokens + output_tokens)` devolvia NULL para essa
    linha e ela desaparecia do todo; o total ficaria 150 em vez de 1350.
    """
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="cheia", input_tokens=100, output_tokens=100),
                row(request_id="semi", input_tokens=None, output_tokens=1200),
                row(request_id="calada", input_tokens=None, output_tokens=None),
            ]
        )
        session.commit()

    totais = queries.totals(engine)
    assert totais["input_tokens"] == 100
    assert totais["output_tokens"] == 1300

    for funcao in (queries.by_model, queries.by_provider, queries.by_requested_model):
        linha = funcao(engine)[0]
        assert linha["tokens"] == 1400, f"{funcao.__name__} perdeu a linha muda"
        assert linha["requests"] == 3, f"{funcao.__name__} perdeu a linha muda"


def test_a_partially_muted_line_still_counts_in_pairs(make_engine):
    """`pairs` tambem soma as duas colunas; a linha muda entra com o que informou."""
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(
                    request_id="semi",
                    requested_model="claude-opus-5",
                    candidate_model="qwen3.8-27b",
                    input_tokens=None,
                    output_tokens=1200,
                ),
                row(
                    request_id="cheia",
                    requested_model="claude-opus-5",
                    candidate_model="qwen3.8-27b",
                    input_tokens=100,
                    output_tokens=100,
                ),
            ]
        )
        session.commit()

    par = queries.pairs(engine)[0]
    assert par["requests"] == 2
    assert par["tokens"] == 1400


def test_min_tokens_ignores_a_linha_sem_tokens_informados(make_engine):
    """`min_tokens` so faz sentido sobre o que foi MEDIDO.

    A linha em silencio total (input NULL e output NULL) nao tem como satisfazer
    "mostrar so quem passou de N tokens": nao e que ela tenha poucos tokens, e
    que ela nao informou nenhum. A linha parcial (input NULL, output 1200) entra
    -- o lado informado e a soma do que ela de fato mediu.
    """
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="calada", input_tokens=None, output_tokens=None),
                row(request_id="semi", input_tokens=None, output_tokens=1200),
                row(request_id="cheia", input_tokens=50, output_tokens=50),
            ]
        )
        session.commit()

    pagina = queries.search_events(engine, min_tokens=101)
    ids = {linha["request_id"] for linha in pagina["events"]}

    assert pagina["total"] == 1, f"min_tokens selecionou errado: total={pagina['total']}"
    assert ids == {"semi"}, f"min_tokens selecionou errado: {ids}"
