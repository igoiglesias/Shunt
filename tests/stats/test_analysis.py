"""A analise do periodo: o prompt, a chamada pelo proprio proxy, e o guardado.

O modelo aqui e sempre dublê. O que se testa e o que o Shunt monta e o que ele
faz com a resposta -- e nao a qualidade do texto, que nenhum teste pode afirmar.
"""

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.core.dispatcher import ShuntResult
from app.stats import analysis
from app.stats.models import Analysis
from tests.core.test_dispatcher import SETTINGS
from tests.stats.test_dossier import row

NOW = datetime.now(UTC)


@pytest.fixture
def store(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        session.add_all(
            [
                row(request_id="a", duration_ms=120),
                row(request_id="b", duration_ms=800, status=500, error_type="api_error"),
            ]
        )
        session.commit()
    return engine


def answer(text="1. Tire a ferramenta `write` do catalogo (oferecida 2, chamada 0).",
           input_tokens=1200, output_tokens=90):
    return ShuntResult(
        status=200,
        body={
            "id": "msg_1",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
        real_model="gpt-oss-120b",
        real_provider="groq",
    )


@pytest.fixture
def dublê(monkeypatch):
    """Substitui o despacho e guarda o que o Shunt mandou para o modelo."""
    enviados = []

    async def fake_dispatch(req, settings, pool):
        enviados.append(req)
        return answer()

    monkeypatch.setattr(analysis, "dispatch", fake_dispatch)
    return enviados


def test_prompt_carrega_o_dossie_em_json(store, dublê):
    import asyncio

    asyncio.run(analysis.analyse(store, SETTINGS, None, filters={}, model="claude-opus-5"))

    pedido = dublê[0]
    assert pedido.protocol == "anthropic"
    assert pedido.body["model"] == "claude-opus-5"
    texto = pedido.body["messages"][0]["content"]
    dossie = json.loads(texto[texto.index("{"):texto.rindex("}") + 1])
    # O dossie vai INTEIRO e em JSON: numero em prosa e numero que o modelo
    # reescreve, e a analise tem de citar o valor exato.
    assert dossie["volume"]["requests"] == 2
    assert dossie["volume"]["errors"] == 1


def test_prompt_exige_recomendacao_com_o_numero_que_a_sustenta(store, dublê):
    import asyncio

    asyncio.run(analysis.analyse(store, SETTINGS, None, filters={}, model="claude-opus-5"))

    system = dublê[0].body["system"]
    assert "cita o número do dossiê que a sustenta" in system
    assert "ordenadas por impacto" in system


def test_analise_e_guardada_com_o_dossie_que_a_gerou(store, dublê):
    import asyncio

    resultado = asyncio.run(
        analysis.analyse(store, SETTINGS, None, filters={"provider": "groq"}, model="m")
    )

    with Session(store) as session:
        linha = session.get(Analysis, resultado["id"])
        assert linha is not None
        assert linha.text.startswith("1. Tire a ferramenta")
        assert linha.candidate_model == "gpt-oss-120b"
        assert linha.provider == "groq"
        assert linha.input_tokens == 1200
        assert linha.output_tokens == 90
        assert linha.status == 200
        # O dossie fica junto: quem le uma recomendacao precisa poder conferir
        # o numero por tras dela, e o periodo ja passou.
        assert json.loads(linha.dossier_json)["volume"]["requests"] == 2
        assert json.loads(linha.filters_json) == {"provider": "groq"}


def test_mesma_janela_reaproveita_a_analise_ja_paga(store, dublê):
    import asyncio

    primeira = asyncio.run(analysis.analyse(store, SETTINGS, None, filters={}, model="m"))
    segunda = asyncio.run(analysis.analyse(store, SETTINGS, None, filters={}, model="m"))

    assert segunda["id"] == primeira["id"]
    assert segunda["cached"] is True
    # Uma chamada ao modelo, duas leituras: analise custa token.
    assert len(dublê) == 1


def test_refresh_paga_de_novo_de_proposito(store, dublê):
    import asyncio

    primeira = asyncio.run(analysis.analyse(store, SETTINGS, None, filters={}, model="m"))
    segunda = asyncio.run(
        analysis.analyse(store, SETTINGS, None, filters={}, model="m", refresh=True)
    )

    assert segunda["id"] != primeira["id"]
    assert len(dublê) == 2


def test_janela_diferente_nao_reaproveita(store, dublê):
    import asyncio

    asyncio.run(analysis.analyse(store, SETTINGS, None, filters={}, model="m"))
    outra = asyncio.run(
        analysis.analyse(
            store, SETTINGS, None, filters={"since": NOW - timedelta(hours=3)}, model="m"
        )
    )

    assert outra["cached"] is False
    assert len(dublê) == 2


def test_erro_do_modelo_volta_como_erro_e_nao_como_analise(store, monkeypatch):
    import asyncio

    async def falha(req, settings, pool):
        return ShuntResult(
            status=502,
            body={"error": {"type": "api_error", "message": "upstream caiu"}},
            trace=["groq: 502 (attempt 1)"],
        )

    monkeypatch.setattr(analysis, "dispatch", falha)

    resultado = asyncio.run(analysis.analyse(store, SETTINGS, None, filters={}, model="m"))

    assert resultado["status"] == 502
    assert "upstream caiu" in resultado["error"]
    assert resultado["text"] == ""
    # Analise que falhou NAO entra no cache: a proxima tentativa tem de chamar
    # o modelo de novo, e nao devolver o erro para sempre.
    with Session(store) as session:
        assert session.query(Analysis).count() == 0


def test_resposta_sem_texto_e_erro_e_nao_analise_vazia(store, monkeypatch):
    import asyncio

    async def vazio(req, settings, pool):
        return ShuntResult(status=200, body={"content": [], "usage": {}}, real_model="m")

    monkeypatch.setattr(analysis, "dispatch", vazio)

    resultado = asyncio.run(analysis.analyse(store, SETTINGS, None, filters={}, model="m"))

    assert resultado["status"] == 502
    assert resultado["text"] == ""


def test_o_dict_de_retorno_concorda_com_o_que_foi_persistido_em_chaves_openai(store, monkeypatch):
    """Divergencia API vs banco quando o usage chega em chaves OpenAI.

    O `_store` (analysis.py) grava `input_tokens` com o mesmo `if/else` do
    dispatcher (`usage.get("input_tokens") if ... is not None else
    usage.get("prompt_tokens")`), mas o dict de retorno da analise devolve
    `usage.get("input_tokens")` puro. Com um provedor que envia so
    `prompt_tokens`/`completion_tokens` (formato OpenAI, sem traducao de
    usage), a tabela `analyses` fica com os numeros e a API devolve None --
    quem le a auditoria via `_cached` ve uma coisa, quem le o banco outra.
    """
    import asyncio

    async def resposta_em_chaves_openai(req, settings, pool):
        return ShuntResult(
            status=200,
            body={
                "id": "msg-1",
                "content": [{"type": "text", "text": "1. Recomendacao."}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 42},
            },
            real_model="gpt-oss-120b",
            real_provider="groq",
        )

    monkeypatch.setattr(analysis, "dispatch", resposta_em_chaves_openai)

    resultado = asyncio.run(analysis.analyse(store, SETTINGS, None, filters={}, model="m"))

    with Session(store) as session:
        linha = session.get(Analysis, resultado["id"])

    # O banco ja usa o if/else: os numeros chegam persistidos.
    assert linha.input_tokens == 7
    assert linha.output_tokens == 42
    # A API tem de devolver o mesmo que o banco guardou -- nao None.
    assert resultado["input_tokens"] == linha.input_tokens
    assert resultado["output_tokens"] == linha.output_tokens


def test_sem_modelo_declarado_nao_chama_ninguem(store, dublê):
    import asyncio

    # `SETTINGS.default_model` e None: nao ha para quem mandar a analise, e
    # inventar um alias aqui mandaria o pedido a um modelo que o operador nao
    # escolheu -- e ele paga a conta.
    resultado = asyncio.run(analysis.analyse(store, SETTINGS, None, filters={}, model=None))

    assert resultado["status"] == 409
    assert "modelo" in resultado["error"]
    assert dublê == []


def test_modelo_ausente_cai_no_default_do_proxy(store, dublê):
    import asyncio

    com_default = SETTINGS.model_copy(update={"default_model": "cheap"})
    asyncio.run(analysis.analyse(store, com_default, None, filters={}, model=None))

    assert dublê[0].body["model"] == "cheap"


def test_lista_traz_as_analises_mais_recentes_primeiro(store, dublê):
    import asyncio

    asyncio.run(analysis.analyse(store, SETTINGS, None, filters={}, model="m"))
    asyncio.run(analysis.analyse(store, SETTINGS, None, filters={"provider": "groq"}, model="m"))

    recentes = analysis.recent(store)

    assert [item["filters"] for item in recentes] == [{"provider": "groq"}, {}]
    # A listagem NAO carrega o dossie: ela vira um menu na tela, e o dossie
    # inteiro em cada item seria megabytes para desenhar uma lista.
    assert "dossier" not in recentes[0]


def test_uma_analise_pelo_id_traz_o_dossie(store, dublê):
    import asyncio

    salva = asyncio.run(analysis.analyse(store, SETTINGS, None, filters={}, model="m"))

    completa = analysis.one(store, salva["id"])

    assert completa["dossier"]["volume"]["requests"] == 2
    assert analysis.one(store, 999) is None


def test_bloco_que_nao_e_texto_fica_fora_da_analise(store, monkeypatch):
    import asyncio

    async def com_raciocinio(req, settings, pool):
        return ShuntResult(
            status=200,
            body={
                "content": [
                    {"type": "thinking", "thinking": "deixa eu ver os numeros"},
                    {"type": "text", "text": "1. Faça X."},
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            real_model="m",
        )

    monkeypatch.setattr(analysis, "dispatch", com_raciocinio)

    resultado = asyncio.run(analysis.analyse(store, SETTINGS, None, filters={}, model="m"))

    # Raciocinio nao e analise: entregar os dois juntos poria o rascunho do
    # modelo na tela como se fosse recomendacao.
    assert resultado["text"] == "1. Faça X."


def test_o_prompt_avisa_que_silencio_nao_e_ausencia_de_cache(store, dublê):
    """A recomendacao errada que motivou esta historia: um modelo leu o periodo
    sem a secao de cache e propos ligar um cache que ja estava ligado."""
    import asyncio

    asyncio.run(analysis.analyse(store, SETTINGS, None, filters={}, model="m"))

    system = dublê[0].body["system"]
    assert "silent_requests" in system
    assert "ausência de dado" in system
