"""De qual projeto veio a requisicao, lido do corpo que o harness manda.

Medido no banco real antes de escrever qualquer linha disto: 117 de 261 corpos
gravados trazem o bloco `Primary working directory:`, e ele aparece em posicoes
DIFERENTES -- ora numa mensagem `system` no meio de `messages`, ora dentro de um
`<system-reminder>` da primeira mensagem do usuario. Por isso a varredura passa
por tudo em vez de olhar um indice fixo.

Medido tambem: 249 dos 261 `request_json` gravados estao cortados em 64.000
chars e nao parseiam como JSON. Extrair depois, do que ficou guardado, nao
funciona -- a extracao roda na INGESTAO, com o corpo inteiro em memoria.
"""

import pytest

from app.core import project as mod

CAMINHO = "/home/iglesias/Documents/Projetos_Pessoais/AgendaFacil"
BLOCO = f"""# Environment
You have been invoked in the following environment:
 - Primary working directory: {CAMINHO}
 - Is a git repository: true
 - Platform: linux
"""


@pytest.mark.parametrize(
    ("corpo", "esperado"),
    [
        # Mensagem `system` no meio de `messages` -- a posicao mais comum.
        ({"messages": [{"role": "user", "content": "oi"},
                       {"role": "system", "content": BLOCO}]}, CAMINHO),
        # Dentro de um bloco de texto da primeira mensagem do usuario.
        ({"messages": [{"role": "user", "content": [{"type": "text", "text": BLOCO}]}]}, CAMINHO),
        # No campo `system` de topo, como string.
        ({"system": BLOCO, "messages": []}, CAMINHO),
        # No campo `system` como lista de blocos.
        ({"system": [{"type": "text", "text": BLOCO}], "messages": []}, CAMINHO),
        # Caminho curto, com a linha seguinte colada.
        ({"system": "Primary working directory: /tmp\n - Is a git repository: false"}, "/tmp"),
        # Caminho com espaco: o corte e na quebra de linha, nao no espaco.
        ({"system": "Primary working directory: /home/x/Meus Projetos\n"}, "/home/x/Meus Projetos"),
    ],
)
def test_acha_o_diretorio_onde_ele_estiver(corpo, esperado):
    assert mod.project_of(corpo) == esperado


@pytest.mark.parametrize(
    "corpo",
    [
        {"messages": [{"role": "user", "content": "sem bloco nenhum"}]},
        {},
        {"messages": "lixo"},
        {"system": [{"type": "image", "source": {}}]},
        {"messages": [{"role": "user", "content": [{"type": "image"}]}]},
        {"system": 42, "messages": None},
    ],
)
def test_corpo_sem_o_bloco_ou_torto_devolve_none(corpo):
    # Requisicao que nao e do Claude Code, ou de uma versao que mudou o rotulo:
    # o proxy continua servindo, e a coluna fica vazia.
    assert mod.project_of(corpo) is None


def test_le_a_sessao_do_metadata():
    corpo = {"metadata": {"user_id": '{"device_id":"d","session_id":"abc-123"}'}}

    assert mod.session_of(corpo) == "abc-123"


@pytest.mark.parametrize(
    "corpo",
    [
        {"metadata": {"user_id": "nao é json"}},
        {"metadata": {}},
        {"metadata": "texto"},
        {},
    ],
)
def test_sessao_ausente_ou_torta_devolve_none(corpo):
    assert mod.session_of(corpo) is None


def test_a_sessao_carrega_o_projeto_para_as_requisicoes_seguintes():
    """O bloco so vem na primeira requisicao de uma sessao do harness."""
    cache = mod.SessionProjects()
    cache.remember("s1", CAMINHO)

    assert cache.recall("s1") == CAMINHO
    assert cache.recall("desconhecida") is None


def test_o_cache_de_sessao_tem_teto_e_descarta_a_mais_antiga():
    # Sem teto, um proxy ligado por semanas guarda uma entrada por sessao para
    # sempre -- vazamento lento, do tipo que so aparece em producao.
    cache = mod.SessionProjects(limit=2)
    cache.remember("a", "/a")
    cache.remember("b", "/b")
    cache.remember("c", "/c")

    assert cache.recall("a") is None
    assert cache.recall("b") == "/b"
    assert cache.recall("c") == "/c"


def test_lembrar_de_novo_renova_a_sessao():
    cache = mod.SessionProjects(limit=2)
    cache.remember("a", "/a")
    cache.remember("b", "/b")
    cache.remember("a", "/a")
    cache.remember("c", "/c")

    # `a` foi usada por ultimo, entao quem sai e `b`.
    assert cache.recall("a") == "/a"
    assert cache.recall("b") is None


def test_caminho_absurdamente_longo_e_cortado():
    longo = "/" + "x" * 900
    assert len(mod.project_of({"system": f"Primary working directory: {longo}\n"})) == mod.MAX_PATH


def test_o_par_herda_o_projeto_da_sessao_e_nao_vaza_entre_sessoes():
    mod.SESSIONS = mod.SessionProjects()
    primeiro = {"metadata": {"user_id": '{"session_id":"s1"}'}, "system": BLOCO}
    seguinte = {"metadata": {"user_id": '{"session_id":"s1"}'}, "messages": [{"role": "user", "content": "e agora?"}]}
    outra = {"metadata": {"user_id": '{"session_id":"s2"}'}, "messages": []}

    assert mod.project_and_session(primeiro) == (CAMINHO, "s1")
    assert mod.project_and_session(seguinte) == (CAMINHO, "s1")
    # Sessao diferente nao herda: seria atribuir trabalho ao projeto errado.
    assert mod.project_and_session(outra) == (None, "s2")


def test_sem_sessao_nenhuma_nao_ha_heranca():
    mod.SESSIONS = mod.SessionProjects()
    mod.project_and_session({"system": BLOCO})

    assert mod.project_and_session({"messages": []}) == (None, None)
