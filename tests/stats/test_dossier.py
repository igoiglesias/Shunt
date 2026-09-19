"""O dossie que vai para o analista, contra uma tabela semeada.

Um dossie e uma FONTE: cada recomendacao da analise cita um numero daqui. Um
numero errado aqui vira conselho errado com cara de medicao, entao toda
asserticao e um valor exato.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.stats import dossier
from app.stats.models import RequestBody, RequestEvent

NOW = datetime.now(UTC)


def row(**over):
    base = {
        "request_id": "r",
        "started_at": NOW - timedelta(minutes=5),
        "route": "/v1/messages",
        "dialect": "anthropic",
        "stream": False,
        "requested_model": "claude-opus-5",
        "rule": "family",
        "matched": "opus",
        "provider": "groq",
        "candidate_model": "gpt-oss-120b",
        "status": 200,
        "error_type": None,
        "input_tokens": 100,
        "output_tokens": 10,
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
def engine(make_engine, tmp_path):
    return make_engine(f"sqlite+pysqlite:///{tmp_path / 'stats.db'}")


@pytest.fixture
def seeded(engine):
    with Session(engine) as session:
        session.add_all(
            [
                row(
                    request_id="a",
                    duration_ms=100,
                    input_tokens=100,
                    output_tokens=10,
                    tools_offered=["read", "write"],
                    tools_called=["read"],
                ),
                row(
                    request_id="b",
                    duration_ms=300,
                    input_tokens=200,
                    output_tokens=20,
                    stream=True,
                    tools_offered=["read", "write"],
                    tools_called=[],
                    attempts=["groq-free:context window too small (1500 > 1000)"],
                    fell_back=True,
                ),
                row(
                    request_id="c",
                    duration_ms=900,
                    input_tokens=50,
                    output_tokens=5,
                    status=429,
                    error_type="rate_limit_error",
                    provider="openrouter",
                    candidate_model="vendor/free",
                    requested_model="claude-haiku-4-5",
                    attempts=[
                        "groq-free:context window too small (9000 > 1000)",
                        "local:taken anyway, nothing in the chain fits",
                    ],
                ),
            ]
        )
        session.commit()
    return engine


def test_periodo_ecoa_a_janela_e_so_os_filtros_usados(seeded):
    since = NOW - timedelta(hours=2)
    dossie = dossier.build(seeded, filters={"since": since, "provider": "groq", "text": None})

    assert dossie["period"]["since"] == since.isoformat()
    assert dossie["period"]["until"] is None
    # Filtro vazio nao entra: quem le a analise precisa saber o que foi pedido,
    # e uma lista com dezesseis "None" esconde justamente isso.
    assert dossie["period"]["filters"] == {"provider": "groq"}


def test_volume_conta_erro_stream_e_fallback_separados(seeded):
    volume = dossier.build(seeded)["volume"]

    assert volume["requests"] == 3
    assert volume["errors"] == 1
    # A fixture tem uma requisicao em stream e uma com fallback -- e sao linhas
    # DIFERENTES, para que trocar as duas colunas nao passe.
    assert volume["streamed"] == 1
    assert volume["fell_back"] == 1
    assert volume["input_tokens"] == 350
    assert volume["output_tokens"] == 35
    assert volume["sampled"] is False


def test_latencia_e_percentil_do_observado(seeded):
    volume = dossier.build(seeded)["volume"]

    assert volume["duration_ms"] == {"p50": 300, "p95": 900, "p99": 900}


def test_por_modelo_soma_tokens_e_erros_de_cada_candidato(seeded):
    linhas = {linha["candidate_model"]: linha for linha in dossier.build(seeded)["by_model"]}

    assert linhas["gpt-oss-120b"] == {
        "candidate_model": "gpt-oss-120b",
        "provider": "groq",
        "requests": 2,
        "errors": 0,
        "input_tokens": 300,
        "output_tokens": 30,
        "p50_ms": 100,
        "p95_ms": 300,
    }
    assert linhas["vendor/free"]["errors"] == 1


def test_por_modelo_pedido_mostra_o_que_o_harness_pediu(seeded):
    pedidos = {linha["requested_model"]: linha["requests"] for linha in
               dossier.build(seeded)["by_requested_model"]}

    assert pedidos == {"claude-opus-5": 2, "claude-haiku-4-5": 1}


def test_cadeia_conta_o_pulo_por_classe_e_ignora_o_ultimo_recurso(seeded):
    cadeia = dossier.build(seeded)["chain"]

    assert cadeia["skips"] == [
        {"candidate": "groq-free", "reason": "não coube", "count": 2},
    ]
    # Uma requisicao (a) respondeu no primeiro candidato; duas tentaram antes.
    assert cadeia["first_candidate_answered"] == 1
    assert cadeia["first_candidate_rate"] == 0.3333


def test_ferramenta_oferecida_e_nunca_chamada_aparece_com_a_razao(seeded):
    ferramentas = {linha["tool"]: linha for linha in dossier.build(seeded)["tools"]}

    assert ferramentas["write"] == {
        "tool": "write",
        "offered": 2,
        "called": 0,
        "call_rate": 0.0,
    }
    assert ferramentas["read"]["call_rate"] == 0.5


def test_mais_lentas_vem_ordenadas_e_carregam_a_cadeia(seeded):
    lentas = dossier.build(seeded, top=2)["slowest"]

    assert [linha["request_id"] for linha in lentas] == ["c", "b"]
    assert lentas[0]["duration_ms"] == 900
    assert lentas[0]["attempts"] == [
        "groq-free:context window too small (9000 > 1000)",
        "local:taken anyway, nothing in the chain fits",
    ]


def test_mais_caras_sao_por_token_e_nao_por_tempo(seeded):
    caras = dossier.build(seeded, top=2)["costliest"]

    # `b` tem 220 tokens e `a` tem 110; `c`, a mais LENTA, e a mais barata.
    assert [linha["request_id"] for linha in caras] == ["b", "a"]
    assert caras[0]["total_tokens"] == 220


def test_erros_agrupados_por_status_e_tipo(seeded):
    assert dossier.build(seeded)["errors"] == [
        {"status": 429, "error_type": "rate_limit_error", "requests": 1},
    ]


def test_filtro_do_periodo_e_obedecido(seeded):
    dossie = dossier.build(seeded, filters={"provider": "openrouter"})

    assert dossie["volume"]["requests"] == 1
    assert [linha["request_id"] for linha in dossie["slowest"]] == ["c"]


def test_conversa_entra_so_quando_foi_gravada(seeded):
    with Session(seeded) as session:
        session.add(
            RequestBody(
                request_id="a",
                prompt="p" * 50,
                answer="r" * 50,
                request_json="{}",
                prompt_bytes=50,
                answer_bytes=50,
                truncated=False,
            )
        )
        session.commit()

    conversas = dossier.build(seeded, sample_chars=10)["conversations"]

    assert [conversa["request_id"] for conversa in conversas] == ["a"]
    # Teto de tamanho: o dossie vai inteiro para dentro de um prompt, e uma
    # conversa de 400 KB custaria mais que a analise que ela sustenta.
    assert conversas[0]["prompt"] == "p" * 10
    assert conversas[0]["answer"] == "r" * 10
    assert conversas[0]["truncated"] is True


def test_sem_conversa_gravada_a_lista_vem_vazia(seeded):
    assert dossier.build(seeded)["conversations"] == []


def test_conversa_pode_ser_desligada_mesmo_com_gravacao_ligada(seeded):
    with Session(seeded) as session:
        session.add(
            RequestBody(
                request_id="a",
                prompt="segredo",
                answer="segredo",
                request_json="{}",
                prompt_bytes=7,
                answer_bytes=7,
                truncated=False,
            )
        )
        session.commit()

    assert dossier.build(seeded, include_conversations=False)["conversations"] == []


def test_periodo_grande_demais_se_diz_amostrado(seeded):
    dossie = dossier.build(seeded, row_limit=2)

    # O total continua exato -- ele vem de um COUNT, nao das linhas lidas.
    assert dossie["volume"]["requests"] == 3
    assert dossie["volume"]["sampled"] is True
    assert dossie["volume"]["rows_read"] == 2


def test_periodo_vazio_nao_estoura(engine):
    dossie = dossier.build(engine)

    assert dossie["volume"]["requests"] == 0
    assert dossie["volume"]["duration_ms"] == {"p50": None, "p95": None, "p99": None}
    assert dossie["by_model"] == []
    assert dossie["chain"]["first_candidate_rate"] == 0.0


def test_status_400_exato_conta_como_erro(engine):
    with Session(engine) as session:
        session.add(row(request_id="e", status=400, error_type="invalid_request_error"))
        session.commit()

    dossie = dossier.build(engine)

    assert dossie["volume"]["errors"] == 1
    assert dossie["by_model"][0]["errors"] == 1


def test_stream_e_fallback_sao_contagens_diferentes(engine):
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="s", stream=True, fell_back=False),
                row(request_id="s2", stream=True, fell_back=False),
                row(request_id="f", stream=False, fell_back=True),
                row(request_id="n", stream=False, fell_back=False),
            ]
        )
        session.commit()

    volume = dossier.build(engine)["volume"]

    # Contagens ASSIMETRICAS de proposito: com 1 e 1, trocar as duas colunas
    # passaria despercebido.
    assert volume["streamed"] == 2
    assert volume["fell_back"] == 1


def test_mesmo_modelo_em_dois_provedores_sao_duas_linhas(engine):
    # O mesmo modelo aberto e servido por varios provedores, com latencia e
    # preco diferentes: junta-los apaga exatamente a comparacao que o operador
    # quer fazer.
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="g", provider="groq", candidate_model="gpt-oss-120b"),
                row(request_id="o", provider="openrouter", candidate_model="gpt-oss-120b"),
            ]
        )
        session.commit()

    linhas = dossier.build(engine)["by_model"]

    assert sorted(linha["provider"] for linha in linhas) == ["groq", "openrouter"]
    assert [linha["requests"] for linha in linhas] == [1, 1]


def test_modelo_pedido_vazio_nao_vira_linha_sem_nome(engine):
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="m", route="/v1/models", requested_model=""),
                row(request_id="x", requested_model="claude-opus-5"),
            ]
        )
        session.commit()

    pedidos = dossier.build(engine)["by_requested_model"]

    assert [linha["requested_model"] for linha in pedidos] == ["claude-opus-5"]


def test_requisicao_recusada_nao_conta_como_primeiro_candidato(engine):
    # Recusada no boot da cadeia: nenhuma tentativa E nenhum candidato. Conta-la
    # como "o primeiro candidato bastou" inflaria a saude da cadeia justamente
    # quando ela falhou.
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="ok", attempts=[], candidate_model="gpt-oss-120b"),
                row(request_id="no", attempts=[], candidate_model=None, status=400),
            ]
        )
        session.commit()

    cadeia = dossier.build(engine)["chain"]

    assert cadeia["first_candidate_answered"] == 1
    assert cadeia["first_candidate_rate"] == 0.5


def test_amostra_de_conversa_pega_a_mais_cara(engine):
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="barata", input_tokens=10, output_tokens=1),
                row(request_id="cara", input_tokens=9000, output_tokens=900),
            ]
        )
        session.add_all(
            [
                RequestBody(
                    request_id="barata",
                    prompt="pouco",
                    answer="pouco",
                    request_json="{}",
                    prompt_bytes=5,
                    answer_bytes=5,
                    truncated=False,
                ),
                RequestBody(
                    request_id="cara",
                    prompt="muito",
                    answer="muito",
                    request_json="{}",
                    prompt_bytes=5,
                    answer_bytes=5,
                    truncated=False,
                ),
            ]
        )
        session.commit()

    conversas = dossier.build(engine, conversations=1)["conversations"]

    assert [conversa["request_id"] for conversa in conversas] == ["cara"]


def test_filtro_desconhecido_e_recusado(engine):
    # Ignorar em silencio daria um dossie de um recorte diferente do que a tela
    # mostra, e ninguem teria como notar.
    with pytest.raises(ValueError, match="order_by"):
        dossier.build(engine, filters={"order_by": "duration"})


def test_o_dossie_diz_de_quais_projetos_veio_o_periodo(engine):
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="a", project="/home/x/agenda"),
                row(request_id="b", project="/home/x/agenda"),
                row(request_id="c", project=None),
            ]
        )
        session.commit()

    projetos = {linha["project"]: linha["requests"] for linha in
                dossier.build(engine)["by_project"]}

    # "sem projeto" aparece: escondido, o analista concluiria sobre a parte
    # achando que via o todo.
    assert projetos == {"/home/x/agenda": 2, "sem projeto": 1}


def test_a_analise_pode_ser_de_um_projeto_so(engine):
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="a", project="/home/x/agenda"),
                row(request_id="b", project="/home/x/shunt"),
            ]
        )
        session.commit()

    dossie = dossier.build(engine, filters={"project": "/home/x/agenda"})

    assert dossie["volume"]["requests"] == 1
    assert dossie["period"]["filters"] == {"project": "/home/x/agenda"}
