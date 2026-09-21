from app.config.settings import ModelCaps, ModelConfig, ProviderConfig, Settings
from app.core.capabilities import Requirements, estimate_tokens, filter_chain, requirements_of
from app.core.resolver import Candidate

SETTINGS = Settings(
    providers={
        "openrouter": ProviderConfig(base_url="https://x/v1", protocol="openai", api_key_env=None)
    },
    models={
        "sem_tools": ModelConfig(
            provider="openrouter",
            model="a",
            supports=ModelCaps(tools=False),
            context_window=64000,
            max_output_tokens=8192,
        ),
        "com_tools": ModelConfig(
            provider="openrouter",
            model="b",
            supports=ModelCaps(tools=True),
            context_window=64000,
            max_output_tokens=8192,
        ),
        "curto": ModelConfig(
            provider="openrouter",
            model="c",
            supports=ModelCaps(tools=True),
            context_window=1000,
            max_output_tokens=256,
        ),
        "sem_vision": ModelConfig(
            provider="openrouter",
            model="d",
            supports=ModelCaps(tools=True, vision=False),
            context_window=64000,
            max_output_tokens=8192,
        ),
        "com_vision": ModelConfig(
            provider="openrouter",
            model="e",
            supports=ModelCaps(tools=True, vision=True),
            context_window=64000,
            max_output_tokens=8192,
        ),
        "sem_streaming": ModelConfig(
            provider="openrouter",
            model="f",
            supports=ModelCaps(tools=True, streaming=False),
            context_window=64000,
            max_output_tokens=8192,
        ),
        "com_streaming": ModelConfig(
            provider="openrouter",
            model="g",
            supports=ModelCaps(tools=True, streaming=True),
            context_window=64000,
            max_output_tokens=8192,
        ),
    },
    routes=[],
    default_model=None,
)


def cand(alias):
    return Candidate(
        alias=alias, provider="openrouter", model=SETTINGS.models[alias].model, protocol="openai"
    )


def test_request_with_tools_requires_tool_support():
    req = requirements_of({"messages": [], "tools": [{"type": "function"}]})
    assert req.tools is True
    assert req.vision is False
    assert req.streaming is False
    assert req.input_tokens == estimate_tokens({"messages": [], "tools": [{"type": "function"}]})


def test_candidate_without_tool_support_is_dropped_with_reason():
    req = requirements_of({"messages": [], "tools": [{"type": "function"}]})
    kept, dropped = filter_chain([cand("sem_tools"), cand("com_tools")], req, SETTINGS)
    assert [c.alias for c in kept] == ["com_tools"]
    assert dropped == [("a", "no tool support")]


def test_candidate_with_smaller_context_window_is_dropped():
    req = requirements_of({"messages": [{"role": "user", "content": "x" * 40000}]})
    kept, dropped = filter_chain([cand("curto"), cand("com_tools")], req, SETTINGS)
    assert [c.alias for c in kept] == ["com_tools"]
    assert dropped[0][1].startswith("context window too small")


def test_transparent_candidate_is_never_filtered():
    req = requirements_of({"messages": [], "tools": [{"type": "function"}]})
    passthrough = Candidate(
        alias=None, provider="openrouter", model="x", protocol="openai", transparent=True
    )
    kept, dropped = filter_chain([passthrough], req, SETTINGS)
    assert kept == [passthrough]
    assert dropped == []


def test_request_with_image_requires_vision_support():
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://x"}}]}
        ]
    }
    req = requirements_of(payload)
    assert req.vision is True
    assert req.tools is False
    assert req.streaming is False
    assert req.input_tokens == estimate_tokens(payload)


def test_candidate_without_vision_support_is_dropped_with_reason():
    req = requirements_of(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "http://x"}}],
                }
            ]
        }
    )
    kept, dropped = filter_chain([cand("sem_vision"), cand("com_vision")], req, SETTINGS)
    assert [c.alias for c in kept] == ["com_vision"]
    assert dropped == [("d", "no vision support")]


def test_request_with_stream_requires_streaming_support():
    payload = {"messages": [], "stream": True}
    req = requirements_of(payload)
    assert req.streaming is True
    assert req.tools is False
    assert req.vision is False
    assert req.input_tokens == estimate_tokens(payload)


def test_candidate_without_streaming_support_is_dropped_with_reason():
    req = requirements_of({"messages": [], "stream": True})
    kept, dropped = filter_chain([cand("sem_streaming"), cand("com_streaming")], req, SETTINGS)
    assert [c.alias for c in kept] == ["com_streaming"]
    assert dropped == [("f", "no streaming support")]


def test_transparent_candidate_with_none_alias_never_filtered_even_when_unfit():
    req = requirements_of({"messages": [], "tools": [{"type": "function"}], "stream": True})
    passthrough = Candidate(
        alias=None, provider="openrouter", model="x", protocol="openai", transparent=False
    )
    kept, dropped = filter_chain([passthrough], req, SETTINGS)
    assert kept == [passthrough]
    assert dropped == []


def test_estimate_tokens_counts_tools_not_just_messages():
    payload = {
        "messages": [{"role": "user", "content": "short"}],
        "tools": [{"type": "function", "function": {"name": "f", "description": "x" * 5000}}],
    }
    req = requirements_of(payload)
    kept, dropped = filter_chain([cand("curto"), cand("com_tools")], req, SETTINGS)
    assert [c.alias for c in kept] == ["com_tools"]
    assert dropped[0][0] == "c"
    assert dropped[0][1].startswith("context window too small")


def test_kept_is_empty_when_the_only_candidate_is_unfit():
    req = requirements_of({"messages": [], "tools": [{"type": "function"}]})
    kept, dropped = filter_chain([cand("sem_tools")], req, SETTINGS)
    assert kept == []
    assert dropped == [("a", "no tool support")]


def test_context_window_boundary_exact_fit_is_kept():
    # Control the estimate precisely by constructing Requirements directly
    # (bypassing requirements_of/estimate_tokens) instead of hand-tuning a
    # payload to hit an exact character count.
    req = Requirements(tools=False, vision=False, streaming=False, input_tokens=1000)
    kept, dropped = filter_chain([cand("curto")], req, SETTINGS)
    assert [c.alias for c in kept] == ["curto"]
    assert dropped == []


def test_context_window_boundary_one_over_is_dropped():
    req = Requirements(tools=False, vision=False, streaming=False, input_tokens=1001)
    kept, dropped = filter_chain([cand("curto")], req, SETTINGS)
    assert kept == []
    assert dropped[0][0] == "c"
    assert dropped[0][1].startswith("context window too small")


def test_transparent_flag_bypasses_filter_even_with_an_unfit_real_alias():
    req = requirements_of({"messages": [], "tools": [{"type": "function"}]})
    unfit_but_transparent = Candidate(
        alias="sem_tools", provider="openrouter", model="a", protocol="openai", transparent=True
    )
    kept, dropped = filter_chain([unfit_but_transparent], req, SETTINGS)
    assert kept == [unfit_but_transparent]
    assert dropped == []


def test_anthropic_image_block_also_requires_vision_support():
    """Uma requisicao Anthropic carrega a imagem como `{"type": "image"}` com
    `source`, nao como o `image_url` da OpenAI. Reconhecer so a grafia OpenAI
    fazia a exigencia de visao sumir quando o corpo chegava sem traducao, e um
    modelo cego passava no filtro."""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgo=",
                        },
                    }
                ],
            }
        ]
    }
    req = requirements_of(payload)
    assert req.vision is True


def test_candidate_without_vision_is_dropped_for_an_anthropic_image_block():
    req = requirements_of(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo=",
                            },
                        }
                    ],
                }
            ]
        }
    )
    kept, dropped = filter_chain([cand("sem_vision"), cand("com_vision")], req, SETTINGS)
    assert [c.alias for c in kept] == ["com_vision"]
    assert dropped == [("d", "no vision support")]


# -- Historia E: o tamanho do pedido escolhe o candidato -----------------------


def test_the_requested_max_tokens_counts_against_the_window():
    """A resposta ocupa a mesma janela do prompt.

    Medido no trafego real: `groq-free` recusava com 413 pedidos que a conta
    so-do-prompt dizia caber. A saida pedida faz parte do orcamento.
    """
    req = Requirements(
        tools=False, vision=False, streaming=False, input_tokens=900, output_tokens=200
    )
    kept, dropped = filter_chain([cand("curto")], req, SETTINGS)
    assert kept == []
    assert dropped[0][0] == "c"


def test_the_drop_reason_carries_the_two_numbers():
    """Sem os numeros, o rastro nao diz por quanto o pedido passou do teto."""
    req = Requirements(
        tools=False, vision=False, streaming=False, input_tokens=1200, output_tokens=300
    )
    _, dropped = filter_chain([cand("curto")], req, SETTINGS)
    assert dropped == [("c", "context window too small (1500 > 1000)")]


def test_requirements_reads_the_max_tokens_of_the_payload():
    req = requirements_of({"messages": [], "max_tokens": 4096})
    assert req.output_tokens == 4096


def test_a_payload_without_max_tokens_budgets_nothing_for_the_answer():
    """Ausente ou nao numerico nao pode virar um teto inventado."""
    assert requirements_of({"messages": []}).output_tokens == 0
    assert requirements_of({"messages": [], "max_tokens": "muitos"}).output_tokens == 0
