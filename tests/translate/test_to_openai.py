import json

import pytest

from app.translate.ids import to_openai_id
from app.translate.to_openai import TooManyStopSequencesError, anthropic_request_to_openai


def convert(body, target="vendor/free", cap=8192):
    return anthropic_request_to_openai(body, target, cap)


def test_system_blocks_are_joined_with_blank_line():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [],
            "system": [{"type": "text", "text": "um"}, {"type": "text", "text": "dois"}],
        }
    )
    assert out["messages"][0] == {"role": "system", "content": "um\n\ndois"}


def test_tool_use_becomes_tool_calls_with_string_arguments():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "read",
                            "input": {"path": "a"},
                        }
                    ],
                }
            ],
        }
    )
    call = out["messages"][0]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "read"
    assert json.loads(call["function"]["arguments"]) == {"path": "a"}


def test_tool_use_without_input_sends_empty_object_not_empty_string():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {}}],
                }
            ],
        }
    )
    assert out["messages"][0]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_tool_results_become_tool_messages_in_order():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "primeiro"},
                        {"type": "tool_result", "tool_use_id": "toolu_2", "content": "segundo"},
                    ],
                }
            ],
        }
    )
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["tool", "tool"]
    assert [m["content"] for m in out["messages"]] == ["primeiro", "segundo"]


def test_error_tool_result_is_prefixed():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "falhou",
                            "is_error": True,
                        }
                    ],
                }
            ],
        }
    )
    assert out["messages"][0]["content"] == "Error: falhou"


def test_text_next_to_tool_result_becomes_a_user_message_after_it():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
                        {"type": "text", "text": "e agora?"},
                    ],
                }
            ],
        }
    )
    assert [m["role"] for m in out["messages"]] == ["tool", "user"]
    assert out["messages"][1]["content"] == "e agora?"


def test_tool_without_parameters_keeps_an_empty_object_schema():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [],
            "tools": [{"name": "now", "description": "hora", "input_schema": {}}],
        }
    )
    fn = out["tools"][0]["function"]
    assert fn["parameters"] == {"type": "object", "properties": {}}
    assert "strict" not in fn


def test_tool_choice_any_becomes_required_and_disables_parallel_calls():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
        }
    )
    assert out["tool_choice"] == "required"
    assert out["parallel_tool_calls"] is False


def test_image_block_becomes_data_url():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
                        }
                    ],
                }
            ],
        }
    )
    part = out["messages"][0]["content"][0]
    assert part["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_cache_control_and_thinking_are_stripped():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "oi", "cache_control": {"type": "ephemeral"}}
                    ],
                }
            ],
        }
    )
    assert "thinking" not in out and "budget_tokens" not in json.dumps(out)
    assert "cache_control" not in json.dumps(out)


def test_max_tokens_is_clamped_to_the_candidate_limit():
    assert (
        convert({"model": "m", "max_tokens": 64000, "messages": []}, cap=4096)["max_tokens"] == 4096
    )


def test_max_tokens_under_the_cap_is_left_alone():
    assert convert({"model": "m", "max_tokens": 100, "messages": []}, cap=4096)["max_tokens"] == 100


def test_streaming_request_asks_for_usage():
    out = convert({"model": "m", "max_tokens": 10, "messages": [], "stream": True})
    assert out["stream"] is True
    assert out["stream_options"] == {"include_usage": True}


def test_non_streaming_request_has_no_stream_fields():
    out = convert({"model": "m", "max_tokens": 10, "messages": []})
    assert "stream" not in out
    assert "stream_options" not in out


def test_more_than_four_stop_sequences_fails_loudly():
    with pytest.raises(TooManyStopSequencesError):
        convert(
            {
                "model": "m",
                "max_tokens": 10,
                "messages": [],
                "stop_sequences": ["a", "b", "c", "d", "e"],
            }
        )


def test_exactly_four_stop_sequences_is_allowed():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [],
            "stop_sequences": ["a", "b", "c", "d"],
        }
    )
    assert out["stop"] == ["a", "b", "c", "d"]


def test_temperature_and_top_p_pass_through_when_present():
    out = convert(
        {"model": "m", "max_tokens": 10, "messages": [], "temperature": 0.7, "top_p": 0.8}
    )
    assert out["temperature"] == 0.7
    assert out["top_p"] == 0.8


def test_temperature_and_top_p_absent_when_not_given():
    out = convert({"model": "m", "max_tokens": 10, "messages": []})
    assert "temperature" not in out
    assert "top_p" not in out


def test_non_dict_content_block_is_skipped():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": ["not a dict", {"type": "text", "text": "oi"}]}],
        }
    )
    assert out["messages"][0]["content"] == "oi"


def test_target_model_replaces_the_requested_one():
    assert (
        convert({"model": "claude-opus-4-5", "max_tokens": 10, "messages": []}, target="qwen3-8b")[
            "model"
        ]
        == "qwen3-8b"
    )


# --- Extra coverage for branches the brief leaves open ---


def test_message_with_plain_string_content_passes_through():
    out = convert(
        {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "oi tudo bem?"}]}
    )
    assert out["messages"] == [{"role": "user", "content": "oi tudo bem?"}]


def test_assistant_message_with_text_and_tool_use_keeps_both_on_one_message():
    # OpenAI carries `content` and `tool_calls` as sibling fields on the SAME
    # assistant message object -- splitting them into two adjacent messages
    # is non-standard and a strict provider may reject or misorder it.
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "vou checar"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "read",
                            "input": {"path": "a"},
                        },
                    ],
                }
            ],
        }
    )
    msgs = out["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"] == "vou checar"
    assert msgs[0]["tool_calls"][0]["function"]["name"] == "read"


def test_tool_result_content_as_list_of_blocks_is_joined_to_text():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [
                                {"type": "text", "text": "linha 1"},
                                {"type": "text", "text": "linha 2"},
                            ],
                        }
                    ],
                }
            ],
        }
    )
    assert out["messages"][0]["content"] == "linha 1\nlinha 2"


def test_image_block_with_url_source_passes_the_url_through():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.com/a.png"},
                        }
                    ],
                }
            ],
        }
    )
    part = out["messages"][0]["content"][0]
    assert part["image_url"]["url"] == "https://example.com/a.png"


# --- Fix round 1: close gaps the review found (correct behaviour, no test) ---


def test_tool_call_id_and_matching_tool_result_id_use_to_openai_id():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "read",
                            "input": {"path": "a"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                },
            ],
        }
    )
    tool_call_id = out["messages"][0]["tool_calls"][0]["id"]
    tool_result_id = out["messages"][1]["tool_call_id"]
    assert tool_call_id == to_openai_id("toolu_1")
    assert tool_result_id == to_openai_id("toolu_1")
    # This is the property the second turn of a conversation depends on: the
    # provider correlates a tool result to its call by matching these ids.
    assert tool_call_id == tool_result_id


def test_cache_control_and_thinking_are_stripped_from_every_location():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "system": [{"type": "text", "text": "oi", "cache_control": {"type": "ephemeral"}}],
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "segredo"}],
                }
            ],
            "tools": [
                {
                    "name": "read",
                    "description": "le arquivo",
                    "input_schema": {"type": "object", "properties": {}},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    )
    dumped = json.dumps(out)
    assert "segredo" not in dumped
    assert "cache_control" not in dumped


def test_tool_choice_forced_tool_becomes_function_choice():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [],
            "tool_choice": {"type": "tool", "name": "read"},
        }
    )
    assert out["tool_choice"] == {"type": "function", "function": {"name": "read"}}


def test_tool_choice_none_is_passed_through():
    out = convert({"model": "m", "max_tokens": 10, "messages": [], "tool_choice": {"type": "none"}})
    assert out["tool_choice"] == "none"


def test_tool_choice_auto_is_passed_through():
    out = convert({"model": "m", "max_tokens": 10, "messages": [], "tool_choice": {"type": "auto"}})
    assert out["tool_choice"] == "auto"


def test_tool_use_with_no_input_key_at_all_sends_empty_object():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "ls"}],
                }
            ],
        }
    )
    assert out["messages"][0]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_system_as_a_plain_string_is_passed_through():
    out = convert({"model": "m", "max_tokens": 10, "messages": [], "system": "seja breve"})
    assert out["messages"][0] == {"role": "system", "content": "seja breve"}


def test_several_tool_results_and_text_in_one_message_keep_tools_before_text():
    out = convert(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "primeiro"},
                        {"type": "tool_result", "tool_use_id": "toolu_2", "content": "segundo"},
                        {"type": "tool_result", "tool_use_id": "toolu_3", "content": "terceiro"},
                        {"type": "text", "text": "e agora?"},
                    ],
                }
            ],
        }
    )
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["tool", "tool", "tool", "user"]
    assert [m["content"] for m in out["messages"][:3]] == ["primeiro", "segundo", "terceiro"]
    assert out["messages"][3]["content"] == "e agora?"
