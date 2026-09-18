import json

import pytest

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


def test_streaming_request_asks_for_usage():
    out = convert({"model": "m", "max_tokens": 10, "messages": [], "stream": True})
    assert out["stream_options"] == {"include_usage": True}


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
