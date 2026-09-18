from app.translate.sse_to_anthropic import OpenAIStreamToAnthropic


def names(events):
    return [name for name, _ in events]


def test_first_text_chunk_opens_message_and_block():
    tr = OpenAIStreamToAnthropic("claude-opus-4-5", "msg_1")
    events = tr.feed(
        {"id": "chatcmpl-1", "model": "vendor/free", "choices": [{"delta": {"content": "Oi"}}]}
    )
    assert names(events) == ["message_start", "content_block_start", "content_block_delta"]
    message = events[0][1]["message"]
    assert message["model"] == "claude-opus-4-5"
    assert message["role"] == "assistant"
    assert message["content"] == []
    assert message["stop_reason"] is None
    assert message["stop_sequence"] is None
    assert message["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert events[1][1]["content_block"] == {"type": "text", "text": ""}
    assert events[2][1]["delta"] == {"type": "text_delta", "text": "Oi"}


def test_second_text_chunk_only_emits_a_delta():
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    tr.feed({"choices": [{"delta": {"content": "Oi"}}]})
    assert names(tr.feed({"choices": [{"delta": {"content": " mundo"}}]})) == [
        "content_block_delta"
    ]


def test_tool_call_name_arrives_only_in_the_first_chunk():
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    first = tr.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "read", "arguments": ""},
                            }
                        ]
                    }
                }
            ]
        }
    )
    start = next(data for name, data in first if name == "content_block_start")
    assert start["content_block"]["name"] == "read"
    assert start["content_block"]["id"].startswith("toolu_")


def test_tool_arguments_arrive_fragmented_as_json_deltas():
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    tr.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "read", "arguments": '{"pa'},
                            }
                        ]
                    }
                }
            ]
        }
    )
    events = tr.feed(
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "a"}'}}]}}
            ]
        }
    )
    assert names(events) == ["content_block_delta"]
    assert events[0][1]["delta"]["type"] == "input_json_delta"
    assert events[0][1]["delta"]["partial_json"] == 'th": "a"}'


def test_switching_tool_index_closes_the_previous_block():
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    tr.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "a", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        }
    )
    events = tr.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call_2",
                                "function": {"name": "b", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert names(events)[:2] == ["content_block_stop", "content_block_start"]


def test_finish_emits_stop_reason_and_usage():
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    tr.feed({"choices": [{"delta": {"content": "oi"}}]})
    tr.feed(
        {
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9},
        }
    )
    events = tr.finish()
    assert names(events) == ["content_block_stop", "message_delta", "message_stop"]
    delta = next(d for n, d in events if n == "message_delta")
    assert delta["delta"]["stop_reason"] == "tool_use"
    assert delta["usage"]["output_tokens"] == 9


def test_stream_with_no_content_still_closes_cleanly():
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    events = tr.finish()
    assert names(events) == ["message_start", "message_delta", "message_stop"]
    delta = next(d for n, d in events if n == "message_delta")
    assert delta["delta"]["stop_reason"] == "end_turn"


# --- Beyond the brief ---------------------------------------------------


def test_text_block_start_and_delta_carry_index_zero():
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    events = tr.feed({"choices": [{"delta": {"content": "Oi"}}]})
    block_start = next(d for n, d in events if n == "content_block_start")
    block_delta = next(d for n, d in events if n == "content_block_delta")
    assert block_start["index"] == 0
    assert block_delta["index"] == 0


def test_chunk_with_text_and_tool_call_together():
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    events = tr.feed(
        {
            "choices": [
                {
                    "delta": {
                        "content": "Oi",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "read", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        }
    )
    assert names(events) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
    ]
    text_delta = events[2][1]
    assert text_delta["delta"] == {"type": "text_delta", "text": "Oi"}
    tool_start = events[4][1]
    assert tool_start["content_block"]["name"] == "read"
    tool_delta = events[5][1]
    assert tool_delta["delta"]["type"] == "input_json_delta"


def test_second_tool_call_arrives_before_first_is_finished():
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    events = tr.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "a", "arguments": "{"},
                            },
                            {
                                "index": 1,
                                "id": "call_2",
                                "function": {"name": "b", "arguments": "{"},
                            },
                        ]
                    }
                }
            ]
        }
    )
    assert names(events) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
    ]
    first_start = events[1][1]
    assert first_start["content_block"]["name"] == "a"
    second_start = events[4][1]
    assert second_start["content_block"]["name"] == "b"
    assert second_start["index"] == first_start["index"] + 1


def test_unknown_finish_reason_maps_to_end_turn():
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    tr.feed({"choices": [{"delta": {"content": "oi"}}]})
    tr.feed({"choices": [{"delta": {}, "finish_reason": "some_new_vendor_reason"}]})
    events = tr.finish()
    delta = next(d for n, d in events if n == "message_delta")
    assert delta["delta"]["stop_reason"] == "end_turn"


def test_chunk_with_empty_choices_list_produces_no_events_after_start():
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    tr.feed({"choices": [{"delta": {"content": "oi"}}]})
    events = tr.feed({"choices": []})
    assert events == []


def test_usage_arriving_in_its_own_final_chunk_with_no_choices():
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    tr.feed({"choices": [{"delta": {"content": "oi"}}]})
    events = tr.feed({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 7}})
    assert events == []
    finish_events = tr.finish()
    delta = next(d for n, d in finish_events if n == "message_delta")
    assert delta["usage"]["output_tokens"] == 7


def test_finish_is_idempotent_second_call_emits_nothing():
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    tr.feed({"choices": [{"delta": {"content": "oi"}}]})
    first = tr.finish()
    assert names(first) == ["content_block_stop", "message_delta", "message_stop"]
    second = tr.finish()
    assert second == []


def test_finish_after_partial_stream_that_opened_a_block_but_never_closed_it():
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    tr.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "read", "arguments": '{"p'},
                            }
                        ]
                    }
                }
            ]
        }
    )
    events = tr.finish()
    assert names(events) == ["content_block_stop", "message_delta", "message_stop"]
    stop_event = events[0][1]
    assert stop_event["index"] == 0


# --- Task 20 gate: mutation-sweep survivors -----------------------------


def test_a_tool_block_switches_back_to_text_and_closes_it_first():
    """The reverse of test_chunk_with_text_and_tool_call_together: a tool
    call opens first, then a later chunk carries plain text. The tool block
    must be closed (content_block_stop) before the text block opens --
    _open_text must call _close_block, not skip it."""
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    tr.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "a", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        }
    )
    events = tr.feed({"choices": [{"delta": {"content": "depois"}}]})
    assert names(events) == ["content_block_stop", "content_block_start", "content_block_delta"]


def test_a_second_fragment_of_a_nonzero_indexed_tool_call_stays_on_its_block():
    """_open_tool must remember the REAL call index it was given, not a
    constant -- otherwise a fragmented tool call whose OpenAI `index` isn't
    0 gets mistaken for a different tool on its second chunk and reopened."""
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    tr.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 2,
                                "id": "call_1",
                                "function": {"name": "a", "arguments": '{"p'},
                            }
                        ]
                    }
                }
            ]
        }
    )
    events = tr.feed(
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 2, "function": {"arguments": 'ath": 1}'}}]}}
            ]
        }
    )
    assert names(events) == ["content_block_delta"]
    assert events[0][1]["delta"]["partial_json"] == 'ath": 1}'


def test_a_tool_call_chunk_with_no_function_key_opens_a_blank_tool_block():
    """`call.get("function") or {}` must fall back to an empty mapping, not
    blow up, when a provider sends a tool_calls entry with no `function`
    key at all (a malformed or minimal first fragment)."""
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    events = tr.feed({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1"}]}}]})
    start = next(d for n, d in events if n == "content_block_start")
    assert start["content_block"]["name"] == ""
    assert start["content_block"]["input"] == {}
    # No `function.arguments` means no delta -- only the block opened.
    assert names(events)[-1] == "content_block_start"


def test_a_tool_call_chunk_missing_id_and_name_gets_empty_defaults():
    from app.translate.ids import to_anthropic_id

    tr = OpenAIStreamToAnthropic("m", "msg_1")
    events = tr.feed({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {}}]}}]})
    start = next(d for n, d in events if n == "content_block_start")
    assert start["content_block"]["name"] == ""
    assert start["content_block"]["id"] == to_anthropic_id("")


def test_a_choice_with_no_delta_key_at_all_produces_no_content_block():
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    events = tr.feed({"choices": [{"index": 0}]})
    assert names(events) == ["message_start"]


def test_usage_missing_completion_tokens_defaults_to_zero():
    """`usage` present and truthy (has `prompt_tokens`), but missing
    `completion_tokens` -- the only key of `_usage` that is ever surfaced
    (in `finish()`'s message_delta), so its default must be exercised with
    a non-empty `usage` dict, not an empty one (`{}` is falsy and would
    skip the update entirely)."""
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    tr.feed({"choices": [{"delta": {"content": "oi"}}], "usage": {"prompt_tokens": 5}})
    events = tr.finish()
    delta = next(d for n, d in events if n == "message_delta")
    assert delta["usage"]["output_tokens"] == 0


def test_a_stream_with_no_choices_key_at_all_still_opens_the_message():
    """`choices` missing outright (not just an empty list) must still hit
    the `.get("choices") or []` fallback and return the events `_start()`
    already produced, not an empty list that would silently drop
    `message_start`."""
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    events = tr.feed({"usage": {"prompt_tokens": 1, "completion_tokens": 0}})
    assert names(events) == ["message_start"]


def test_a_tool_calls_index_missing_key_defaults_to_zero():
    """A tool_calls entry with no `index` key at all must default to 0, the
    same as an explicit `"index": 0` -- mixed across two chunks so a wrong
    default (e.g. 1) shows up as a spurious block switch instead of a
    continued fragment, which a test using the same (wrong) default on
    both sides would never catch."""
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    tr.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "a", "arguments": "{"},
                            }
                        ]
                    }
                }
            ]
        }
    )
    events = tr.feed({"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "}"}}]}}]})
    assert names(events) == ["content_block_delta"]


def test_only_the_first_choice_of_a_multi_choice_chunk_is_translated():
    """Real OpenAI streams always send exactly one choice, but nothing in
    this translator enforces that -- it must read choices[0], the first
    one, not choices[-1] or any other."""
    tr = OpenAIStreamToAnthropic("m", "msg_1")
    events = tr.feed(
        {
            "choices": [
                {"index": 0, "delta": {"content": "primeiro"}},
                {"index": 1, "delta": {"content": "segundo"}},
            ]
        }
    )
    delta = next(d for n, d in events if n == "content_block_delta")
    assert delta["delta"]["text"] == "primeiro"


# --------------------------------------------------------------------------
# Raciocinio em streaming
# --------------------------------------------------------------------------


def test_a_reasoning_delta_opens_a_thinking_block():
    t = OpenAIStreamToAnthropic("m", "msg_1")
    events = t.feed({"choices": [{"delta": {"reasoning_content": "pen"}}]})
    assert names(events) == ["message_start", "content_block_start", "content_block_delta"]
    assert events[1][1]["content_block"] == {"type": "thinking", "thinking": ""}
    assert events[2][1]["delta"] == {"type": "thinking_delta", "thinking": "pen"}


def test_reasoning_then_text_closes_the_thinking_block_first():
    """Os dois nao convivem no mesmo bloco: a Anthropic exige um
    `content_block_stop` antes de abrir o proximo, e um cliente que receba
    `text_delta` dentro de um bloco `thinking` renderiza a resposta como se
    fosse pensamento."""
    t = OpenAIStreamToAnthropic("m", "msg_1")
    t.feed({"choices": [{"delta": {"reasoning_content": "pen"}}]})
    events = t.feed({"choices": [{"delta": {"content": "Paris"}}]})
    assert names(events) == ["content_block_stop", "content_block_start", "content_block_delta"]
    assert events[1][1]["index"] == 1
    assert events[1][1]["content_block"]["type"] == "text"


def test_consecutive_reasoning_deltas_stay_in_one_block():
    t = OpenAIStreamToAnthropic("m", "msg_1")
    t.feed({"choices": [{"delta": {"reasoning_content": "pen"}}]})
    events = t.feed({"choices": [{"delta": {"reasoning_content": "sando"}}]})
    assert names(events) == ["content_block_delta"]
    assert events[0][1]["index"] == 0
    assert events[0][1]["delta"] == {"type": "thinking_delta", "thinking": "sando"}


def test_the_openrouter_spelling_streams_the_same_way():
    t = OpenAIStreamToAnthropic("m", "msg_1")
    events = t.feed({"choices": [{"delta": {"reasoning": "pen"}}]})
    assert events[-1][1]["delta"] == {"type": "thinking_delta", "thinking": "pen"}


def test_an_open_thinking_block_is_closed_by_finish():
    """`finish()` fecha o que estiver aberto, e o bloco de pensamento nao e
    excecao: um stream que morre raciocinando ainda tem de sair bem formado."""
    t = OpenAIStreamToAnthropic("m", "msg_1")
    t.feed({"choices": [{"delta": {"reasoning_content": "pen"}}]})
    assert names(t.finish()) == ["content_block_stop", "message_delta", "message_stop"]


def test_reasoning_after_text_closes_the_text_block_first():
    """A ordem inversa da usual, e ela acontece: ha provedor que volta a
    raciocinar depois de ja ter escrito. Abrir o bloco de pensamento sem
    fechar o de texto deixa dois blocos abertos ao mesmo tempo, e o
    `content_block_stop` seguinte fecha o indice errado."""
    t = OpenAIStreamToAnthropic("m", "msg_1")
    t.feed({"choices": [{"delta": {"content": "Paris"}}]})
    events = t.feed({"choices": [{"delta": {"reasoning_content": "repensando"}}]})
    assert names(events) == ["content_block_stop", "content_block_start", "content_block_delta"]
    assert events[0][1]["index"] == 0
    assert events[1][1]["index"] == 1
    assert events[1][1]["content_block"]["type"] == "thinking"


def test_reasoning_after_a_tool_call_closes_the_tool_block_first():
    t = OpenAIStreamToAnthropic("m", "msg_1")
    t.feed(
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "f"}}]}}
            ]
        }
    )
    events = t.feed({"choices": [{"delta": {"reasoning_content": "pen"}}]})
    assert names(events) == ["content_block_stop", "content_block_start", "content_block_delta"]
    assert events[1][1]["content_block"]["type"] == "thinking"
