"""A busca da tela de auditoria, um caso por filtro e um combinando tres.

O que se pergunta aqui nao e "como esta indo", e sim "o que aconteceu naquela
requisicao" -- entao cada filtro tem de estreitar de verdade, e a contagem tem
de dizer quantas existem, e nao quantas couberam na pagina.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.stats import queries
from tests.stats.test_queries import row

NOW = datetime.now(UTC)


@pytest.fixture
def store(make_engine, tmp_path):
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'busca.db'}")
    with Session(engine) as session:
        session.add_all(
            [
                row(
                    request_id="ok-groq",
                    started_at=NOW - timedelta(minutes=2),
                    route="/v1/messages",
                    provider="groq",
                    candidate_model="openai/gpt-oss-120b",
                    requested_model="claude-sonnet-4-5",
                    duration_ms=300,
                    input_tokens=100,
                    output_tokens=50,
                    tools_offered=["Read", "Bash"],
                    tools_called=["Bash"],
                ),
                row(
                    request_id="lento-local",
                    started_at=NOW - timedelta(minutes=10),
                    provider="local",
                    candidate_model="qwen3.8-27b",
                    requested_model="claude-opus-5",
                    stream=True,
                    ttft_ms=900,
                    duration_ms=30_000,
                    input_tokens=9000,
                    output_tokens=1200,
                ),
                row(
                    request_id="erro-429",
                    started_at=NOW - timedelta(minutes=20),
                    route="/v1/chat/completions",
                    dialect="openai",
                    provider="openrouter",
                    candidate_model="openrouter/free",
                    status=429,
                    error_type="rate_limit_error",
                    duration_ms=80,
                ),
                row(
                    request_id="recusada",
                    started_at=NOW - timedelta(minutes=30),
                    provider=None,
                    candidate_model=None,
                    requested_model="modelo-sem-dono",
                    status=400,
                    error_type="invalid_request_error",
                    attempts=["free: credential OPENROUTER_API_KEY is not set"],
                    duration_ms=1,
                ),
                row(
                    request_id="com-fallback",
                    started_at=NOW - timedelta(hours=3),
                    provider="groq",
                    candidate_model="openai/gpt-oss-120b",
                    attempts=["qwen-local: 502 (attempt 1)"],
                    fell_back=True,
                    duration_ms=1500,
                ),
            ]
        )
        session.commit()
    return engine


def ids(page):
    return [event["request_id"] for event in page["events"]]


def test_with_no_filter_everything_comes_back_newest_first(store):
    page = queries.search_events(store)
    assert ids(page) == ["ok-groq", "lento-local", "erro-429", "recusada", "com-fallback"]
    assert page["total"] == 5
    assert page["next_cursor"] is None


def test_the_window_is_exact_and_not_just_the_last_n_hours(store):
    page = queries.search_events(
        store,
        since=NOW - timedelta(minutes=25),
        until=NOW - timedelta(minutes=5),
    )
    assert ids(page) == ["lento-local", "erro-429"]


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"route": "/v1/chat/completions"}, ["erro-429"]),
        ({"dialect": "openai"}, ["erro-429"]),
        ({"provider": "groq"}, ["ok-groq", "com-fallback"]),
        ({"candidate_model": "qwen3.8-27b"}, ["lento-local"]),
        ({"requested_model": "claude-opus-5"}, ["lento-local"]),
        ({"error_type": "rate_limit_error"}, ["erro-429"]),
        ({"status_min": 400}, ["erro-429", "recusada"]),
        ({"status_min": 400, "status_max": 428}, ["recusada"]),
        ({"stream": True}, ["lento-local"]),
        ({"fell_back": True}, ["com-fallback"]),
        ({"has_tools": True}, ["ok-groq"]),
        ({"min_duration_ms": 1500}, ["lento-local", "com-fallback"]),
        ({"min_tokens": 5000}, ["lento-local"]),
    ],
)
def test_each_filter_narrows_on_its_own(store, filters, expected):
    assert ids(queries.search_events(store, **filters)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("gpt-oss", ["ok-groq", "com-fallback"]),
        ("opus", ["lento-local"]),
        ("erro-4", ["erro-429"]),
        ("Bash", ["ok-groq"]),
        ("OPENROUTER_API_KEY", ["recusada"]),
        ("nao-existe-nada-assim", []),
    ],
)
def test_the_free_text_reaches_id_models_tools_and_the_chain(store, text, expected):
    assert ids(queries.search_events(store, text=text)) == expected


def test_three_filters_at_once_are_combined_with_and(store):
    page = queries.search_events(store, provider="groq", min_duration_ms=1000, fell_back=True)
    assert ids(page) == ["com-fallback"]


def test_ordering_by_duration_puts_the_slowest_first(store):
    page = queries.search_events(store, order_by="duration")
    assert ids(page)[:2] == ["lento-local", "com-fallback"]


def test_paging_walks_the_whole_result_without_repeating(store):
    first = queries.search_events(store, limit=2)
    assert len(first["events"]) == 2
    assert first["total"] == 5, "o total conta a busca inteira, e nao a pagina"
    second = queries.search_events(store, limit=2, cursor=first["next_cursor"])
    third = queries.search_events(store, limit=2, cursor=second["next_cursor"])
    seen = ids(first) + ids(second) + ids(third)
    assert len(seen) == len(set(seen)) == 5
    assert third["next_cursor"] is None


def test_an_unknown_order_falls_back_to_time_instead_of_raising(store):
    assert ids(queries.search_events(store, order_by="abacaxi")) == ids(
        queries.search_events(store)
    )


def test_one_request_comes_back_whole(store):
    event = queries.event_detail(store, "com-fallback")
    assert event is not None
    assert event["provider"] == "groq"
    assert event["attempts"] == ["qwen-local: 502 (attempt 1)"]
    assert event["fell_back"] is True
    assert event["started_at"].endswith("+00:00")


def test_an_unknown_request_id_answers_nothing(store):
    assert queries.event_detail(store, "nunca-existiu") is None


def test_a_broken_cursor_starts_over_instead_of_failing(store):
    """Cursor copiado pela metade de uma URL nao pode virar erro na cara de quem investiga."""
    whole = queries.search_events(store)
    for lixo in ("", "%%%", "abc", "eyJ4Ijox"):
        assert ids(queries.search_events(store, cursor=lixo)) == ids(whole)


def test_paging_by_duration_also_walks_without_repeating(store):
    first = queries.search_events(store, order_by="duration", limit=2)
    second = queries.search_events(
        store, order_by="duration", limit=2, cursor=first["next_cursor"]
    )
    third = queries.search_events(
        store, order_by="duration", limit=2, cursor=second["next_cursor"]
    )
    seen = ids(first) + ids(second) + ids(third)
    assert len(seen) == len(set(seen)) == 5
    durations = [
        event["duration_ms"]
        for event in first["events"] + second["events"] + third["events"]
    ]
    assert durations == sorted(durations, reverse=True)


def test_the_page_size_is_clamped(store):
    assert len(queries.search_events(store, limit=0)["events"]) == 1
    assert queries.search_events(store, limit=10_000)["total"] == 5


def test_rows_out_of_insertion_order_still_come_back_newest_first(make_engine, tmp_path):
    """O gravador entrega em LOTE: duas requisicoes da mesma rajada podem entrar
    no banco fora da ordem em que aconteceram, e o `id` nao pode mandar."""
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'ordem.db'}")
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="antiga", started_at=NOW - timedelta(hours=2)),
                row(request_id="recente", started_at=NOW - timedelta(minutes=1)),
                row(request_id="meio", started_at=NOW - timedelta(minutes=30)),
            ]
        )
        session.commit()
    assert ids(queries.search_events(engine)) == ["recente", "meio", "antiga"]


def test_a_cursor_that_decodes_to_the_wrong_shape_starts_over():
    """base64 valido com conteudo de outra coisa: comeca do inicio, sem estourar."""
    from base64 import urlsafe_b64encode

    for payload in (b'["lista"]', b'{"sem": "id"}', b'"texto"'):
        cursor = urlsafe_b64encode(payload).decode().rstrip("=")
        assert queries.decode_cursor(cursor) is None


def test_a_cursor_without_its_instant_falls_back_to_now(store):
    """Cursor de uma versao antiga, sem o campo de tempo: nao pode levantar."""
    from base64 import urlsafe_b64encode

    cursor = urlsafe_b64encode(b'{"id": 99, "duration_ms": 0}').decode().rstrip("=")
    page = queries.search_events(store, cursor=cursor)
    assert isinstance(page["events"], list)


def test_two_requests_at_the_very_same_instant_do_not_repeat_across_pages(make_engine, tmp_path):
    """Mutante A4: o cursor sem desempate por id sobreviveu a varredura.

    Duas requisicoes do mesmo lote podem ter o mesmo instante ate o
    microssegundo. Sem o desempate, a pagina seguinte comeca em "tudo que e
    mais antigo que este instante" e as gemeas somem -- ou voltam duas vezes.
    """
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'gemeas.db'}")
    instante = NOW - timedelta(minutes=1)
    with Session(engine) as session:
        session.add_all(
            [row(request_id=f"gemea-{i}", started_at=instante, duration_ms=500) for i in range(4)]
        )
        session.commit()
    vistos = []
    cursor = None
    for _ in range(4):
        page = queries.search_events(engine, limit=2, cursor=cursor)
        vistos += ids(page)
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert sorted(vistos) == ["gemea-0", "gemea-1", "gemea-2", "gemea-3"]
    assert len(vistos) == len(set(vistos)), "a paginacao repetiu uma gemea"


def test_the_same_holds_when_the_order_is_by_duration(make_engine, tmp_path):
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'iguais.db'}")
    with Session(engine) as session:
        session.add_all(
            [row(request_id=f"igual-{i}", duration_ms=1234) for i in range(4)]
        )
        session.commit()
    vistos = []
    cursor = None
    for _ in range(4):
        page = queries.search_events(engine, order_by="duration", limit=2, cursor=cursor)
        vistos += ids(page)
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert len(vistos) == len(set(vistos)) == 4


def test_min_tokens_counts_both_directions(make_engine, tmp_path):
    """Mutante A6: contar so a entrada sobreviveu.

    Uma resposta longa a um prompt curto e exatamente a requisicao cara que se
    quer achar, e ela tem entrada pequena.
    """
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'tokens.db'}")
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="entrada-grande", input_tokens=900, output_tokens=10),
                row(request_id="saida-grande", input_tokens=10, output_tokens=900),
                row(request_id="pequena", input_tokens=10, output_tokens=10),
            ]
        )
        session.commit()
    achadas = set(ids(queries.search_events(engine, min_tokens=500)))
    assert achadas == {"entrada-grande", "saida-grande"}


def test_ordering_by_duration_never_starts_with_the_fastest(store):
    """Mutante A1: `asc` no lugar de `desc` devolveria a mais rapida primeiro."""
    page = queries.search_events(store, order_by="duration")
    durations = [event["duration_ms"] for event in page["events"]]
    assert durations == sorted(durations, reverse=True)
    assert durations[0] == 30_000
