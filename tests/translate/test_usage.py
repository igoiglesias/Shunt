"""Os sinais de cache que os provedores mandam, atravessando a traducao.

MEDIDO nos tres provedores desta instalacao, com duas requisicoes identicas:

  llama.cpp local     1a: cached_tokens 0    2a: cached_tokens 2814 de 2818
  groq gpt-oss-120b   nenhum `prompt_tokens_details` em nenhuma das duas
  openrouter/free     cached_tokens 0 nas duas, e `cache_write_tokens` tambem

Dai a regra que este modulo existe para sustentar: AUSENTE e ZERO sao coisas
diferentes. Zero e uma medicao ("o cache errou"); ausente e "este provedor nao
diz nada". Transformar ausente em zero faria o painel afirmar que o Groq tem 0%
de cache, quando o que se sabe e que ele nao informa.
"""

from app.translate.usage import (
    anthropic_usage_to_openai,
    openai_usage_to_anthropic,
    tokens_of,
)


def test_cache_do_dialeto_openai_vira_o_nome_anthropic():
    traduzido = openai_usage_to_anthropic(
        {
            "prompt_tokens": 2818,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 2814},
        }
    )

    assert traduzido == {
        "input_tokens": 2818,
        "output_tokens": 10,
        "cache_read_input_tokens": 2814,
    }


def test_escrita_de_cache_tambem_atravessa():
    traduzido = openai_usage_to_anthropic(
        {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 90},
        }
    )

    assert traduzido["cache_read_input_tokens"] == 0
    assert traduzido["cache_creation_input_tokens"] == 90


def test_provedor_que_nao_fala_de_cache_nao_ganha_campo():
    """O Groq nao manda `prompt_tokens_details`. Inventar um zero ali seria
    afirmar "o cache errou" onde nao houve medicao nenhuma."""
    traduzido = openai_usage_to_anthropic({"prompt_tokens": 3678, "completion_tokens": 10})

    assert traduzido == {"input_tokens": 3678, "output_tokens": 10}


def test_detalhe_torto_nao_estoura():
    assert openai_usage_to_anthropic({"prompt_tokens": 1, "prompt_tokens_details": None}) == {
        "input_tokens": 1,
        "output_tokens": None,
    }
    assert openai_usage_to_anthropic({"prompt_tokens_details": "lixo"}) == {
        "input_tokens": None,
        "output_tokens": None,
    }
    assert openai_usage_to_anthropic(None) == {"input_tokens": None, "output_tokens": None}


def test_o_caminho_de_volta_leva_o_cache_para_o_dialeto_openai():
    traduzido = anthropic_usage_to_openai(
        {
            "input_tokens": 500,
            "output_tokens": 20,
            "cache_read_input_tokens": 480,
            "cache_creation_input_tokens": 12,
        }
    )

    assert traduzido == {
        "prompt_tokens": 500,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 480, "cache_write_tokens": 12},
    }


def test_sem_cache_o_caminho_de_volta_nao_inventa_detalhe():
    assert anthropic_usage_to_openai({"input_tokens": 7, "output_tokens": 3}) == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
    }


def test_o_gravador_le_o_cache_nos_dois_dialetos():
    from app.translate.usage import cache_of

    assert cache_of({"input_tokens": 9, "cache_read_input_tokens": 8}) == (8, None)
    assert cache_of({"prompt_tokens_details": {"cached_tokens": 5, "cache_write_tokens": 2}}) == (5, 2)
    # Silencio continua silencio: None, e nao zero.
    assert cache_of({"prompt_tokens": 100}) == (None, None)
    assert cache_of(None) == (None, None)


def test_tokens_do_dialeto_anthropic_sao_lidos_direto():
    assert tokens_of({"input_tokens": 10, "output_tokens": 5}) == (10, 5)


def test_tokens_do_dialeto_openai_caem_no_segundo_nome():
    """Um provedor OpenAI respondendo a um caller OpenAI nao tem traducao: o
    usage chega em `prompt_tokens`. Esse era o ramo que nao tinha teste
    (revisao do plano bypass-stream-bugs) e devolvia None onde havia numero."""
    assert tokens_of({"prompt_tokens": 31, "completion_tokens": 17}) == (31, 17)


def test_zero_de_um_dialeto_nao_e_sobreposto_pelo_outro():
    """Zero e medicao: `input_tokens: 0` tem que vencer `prompt_tokens`. O
    `or` aqui trocaria um zero verdadeiro por um numero do outro dialeto."""
    assert tokens_of({"input_tokens": 0, "prompt_tokens": 31}) == (0, None)


def test_usage_nao_inteiro_e_truncado_nao_descartado():
    """Alguns provedores enviam numerico nao-inteiro; truncar e a leitura
    honesta, descartar faria o painel afirmar que o provedor nao informou."""
    assert tokens_of({"prompt_tokens": 5.0, "completion_tokens": 2.0}) == (5, 2)


def test_bool_e_string_nao_sao_tokens():
    assert tokens_of({"input_tokens": True}) == (None, None)
    assert tokens_of({"input_tokens": "7"}) == (None, None)


def test_silencio_continua_silencio_no_painel():
    assert tokens_of({}) == (None, None)
    assert tokens_of(None) == (None, None)
