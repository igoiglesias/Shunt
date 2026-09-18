import json
import time

from app.translate.to_anthropic_request import openai_request_to_anthropic
from app.translate.to_openai import anthropic_response_to_openai


def test_system_message_moves_to_the_root_field():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "seja breve"},
                {"role": "user", "content": "oi"},
            ],
        },
        "claude-fable-5-1",
        4096,
    )
    assert out["system"] == "seja breve"
    assert [m["role"] for m in out["messages"]] == ["user"]


def test_max_tokens_is_required_by_anthropic_and_always_present():
    out = openai_request_to_anthropic(
        {"model": "m", "messages": [{"role": "user", "content": "oi"}]}, "claude-fable-5-1", 4096
    )
    assert out["max_tokens"] == 4096


def test_tool_calls_become_tool_use_blocks():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path": "a"}'},
                        }
                    ],
                }
            ],
        },
        "claude-fable-5-1",
        4096,
    )
    block = out["messages"][0]["content"][0]
    assert block["type"] == "tool_use"
    assert block["input"] == {"path": "a"}
    assert block["id"].startswith("toolu_")


def test_tool_messages_become_tool_result_blocks_in_a_user_message():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [{"role": "tool", "tool_call_id": "call_1", "content": "conteudo"}],
        },
        "claude-fable-5-1",
        4096,
    )
    message = out["messages"][0]
    assert message["role"] == "user"
    assert message["content"][0]["type"] == "tool_result"
    assert message["content"][0]["content"] == "conteudo"
    assert message["content"][0]["tool_use_id"].startswith("toolu_")


def test_tools_move_to_the_anthropic_schema():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "le",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "required",
        },
        "claude-fable-5-1",
        4096,
    )
    assert out["tools"][0]["name"] == "read"
    assert out["tools"][0]["input_schema"] == {"type": "object", "properties": {}}
    assert out["tool_choice"] == {"type": "any"}


def test_anthropic_answer_becomes_an_openai_completion():
    out = anthropic_response_to_openai(
        {
            "id": "msg_1",
            "model": "claude-fable-5-1",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "vou ler"},
                {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a"}},
            ],
            "usage": {"input_tokens": 5, "output_tokens": 9},
        },
        "gpt-4o",
    )
    message = out["choices"][0]["message"]
    assert out["object"] == "chat.completion"
    assert out["model"] == "gpt-4o"
    assert message["content"] == "vou ler"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"path": "a"}
    assert message["tool_calls"][0]["id"].startswith("call_")
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert out["usage"]["prompt_tokens"] == 5


def test_anthropic_answer_created_uses_a_real_timestamp_by_default():
    before = int(time.time())
    out = anthropic_response_to_openai({"id": "msg_1", "content": []}, "gpt-4o")
    after = int(time.time())
    assert before <= out["created"] <= after


def test_anthropic_answer_created_is_injectable_for_determinism():
    out = anthropic_response_to_openai(
        {"id": "msg_1", "content": []}, "gpt-4o", clock=lambda: 1700000000.0
    )
    assert out["created"] == 1700000000


# --- Additional coverage beyond the brief ---


def test_consecutive_tool_messages_become_one_user_message_with_several_results():
    # Decision: consecutive `role: "tool"` messages collapse into a SINGLE
    # user message carrying multiple tool_result blocks, matching how the
    # assistant that requested them saw parallel tool calls answered
    # together, and avoiding a burst of near-empty user turns that a strict
    # provider could read as odd alternation.
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {"role": "tool", "tool_call_id": "call_1", "content": "um"},
                {"role": "tool", "tool_call_id": "call_2", "content": "dois"},
            ],
        },
        "claude-fable-5-1",
        4096,
    )
    assert len(out["messages"]) == 1
    message = out["messages"][0]
    assert message["role"] == "user"
    assert [b["type"] for b in message["content"]] == ["tool_result", "tool_result"]
    assert message["content"][0]["content"] == "um"
    assert message["content"][1]["content"] == "dois"


def test_assistant_message_with_text_and_tool_calls_keeps_both():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": "vou checar",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                }
            ],
        },
        "claude-fable-5-1",
        4096,
    )
    blocks = out["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "vou checar"}
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["name"] == "read"


def test_malformed_tool_call_arguments_become_empty_input():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{not json"},
                        }
                    ],
                }
            ],
        },
        "claude-fable-5-1",
        4096,
    )
    block = out["messages"][0]["content"][0]
    assert block["input"] == {}


def test_bare_string_stop_becomes_a_single_element_stop_sequences_list():
    out = openai_request_to_anthropic(
        {"model": "m", "messages": [{"role": "user", "content": "oi"}], "stop": "FIM"},
        "claude-fable-5-1",
        4096,
    )
    assert out["stop_sequences"] == ["FIM"]


def test_system_message_in_the_middle_still_moves_to_the_root_field():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "oi"},
                {"role": "system", "content": "seja breve"},
                {"role": "user", "content": "tudo bem?"},
            ],
        },
        "claude-fable-5-1",
        4096,
    )
    assert out["system"] == "seja breve"
    assert [m["role"] for m in out["messages"]] == ["user", "user"]


def test_tool_arguments_valid_json_but_not_an_object_becomes_empty_input():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "[1, 2, 3]"},
                        }
                    ],
                }
            ],
        },
        "claude-fable-5-1",
        4096,
    )
    block = out["messages"][0]["content"][0]
    assert block["input"] == {}


def test_every_known_stop_reason_maps_to_its_own_finish_reason():
    # Only "tool_use" -> "tool_calls" was pinned elsewhere in this file --
    # "end_turn", "max_tokens" and "stop_sequence" all default to the same
    # "stop" fallback an unrecognized value produces, so swapping any one
    # of their FINISH_REASONS entries for "stop" left the suite green
    # before this test existed.
    cases = {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop"}
    for stop_reason, expected in cases.items():
        out = anthropic_response_to_openai(
            {
                "id": "msg_1",
                "model": "m",
                "stop_reason": stop_reason,
                "content": [{"type": "text", "text": "oi"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            "gpt-4o",
        )
        assert out["choices"][0]["finish_reason"] == expected, stop_reason


def test_msg_id_becomes_a_chatcmpl_id_with_hyphen():
    # Real OpenAI completion ids are hyphenated -- chatcmpl-01ABC. A prior
    # version of this transform did a bare raw_id.replace("msg", "chatcmpl",
    # 1), which left the original underscore in place and produced
    # "chatcmpl_01ABC" -- a shape no real OpenAI id has. Reverting the
    # hyphen fix would turn this green again.
    out = anthropic_response_to_openai(
        {
            "id": "msg_01ABC",
            "model": "m",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "oi"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "gpt-4o",
    )
    assert out["id"] == "chatcmpl-01ABC"


def test_id_not_starting_with_msg_still_gets_a_sane_chatcmpl_form():
    # An id that never had the "msg" prefix must not be passed through
    # unchanged (that would leak a raw Anthropic-shaped id into an
    # OpenAI-shaped field) -- it still gets rewritten into "chatcmpl-" form.
    out = anthropic_response_to_openai(
        {
            "id": "weird-vendor-id-123",
            "model": "m",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "oi"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "gpt-4o",
    )
    assert out["id"] == "chatcmpl-weird-vendor-id-123"
    assert not out["id"].startswith("weird")


def test_anthropic_response_missing_id_falls_back_to_chatcmpl_shunt_shape():
    out = anthropic_response_to_openai(
        {
            "model": "m",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "oi"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "gpt-4o",
    )
    assert out["id"] == "chatcmpl_shunt"


def test_tool_choice_string_auto_is_passed_through():
    out = openai_request_to_anthropic(
        {"model": "m", "messages": [], "tool_choice": "auto"}, "claude-fable-5-1", 4096
    )
    assert out["tool_choice"] == {"type": "auto"}


def test_tool_choice_string_none_becomes_none_type():
    out = openai_request_to_anthropic(
        {"model": "m", "messages": [], "tool_choice": "none"}, "claude-fable-5-1", 4096
    )
    assert out["tool_choice"] == {"type": "none"}


def test_tool_choice_unrecognized_string_falls_back_to_auto():
    out = openai_request_to_anthropic(
        {"model": "m", "messages": [], "tool_choice": "something_new"}, "claude-fable-5-1", 4096
    )
    assert out["tool_choice"] == {"type": "auto"}


def test_tool_choice_function_dict_becomes_a_forced_tool_choice():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [],
            "tool_choice": {"type": "function", "function": {"name": "read"}},
        },
        "claude-fable-5-1",
        4096,
    )
    assert out["tool_choice"] == {"type": "tool", "name": "read"}


def test_parallel_tool_calls_false_sets_disable_parallel_tool_use():
    out = openai_request_to_anthropic(
        {"model": "m", "messages": [], "parallel_tool_calls": False}, "claude-fable-5-1", 4096
    )
    assert out["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


def test_data_url_image_becomes_a_base64_image_block():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
                    ],
                }
            ],
        },
        "claude-fable-5-1",
        4096,
    )
    block = out["messages"][0]["content"][0]
    assert block == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }


def test_plain_url_image_becomes_a_url_image_block():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
                    ],
                }
            ],
        },
        "claude-fable-5-1",
        4096,
    )
    block = out["messages"][0]["content"][0]
    assert block == {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}


def test_two_system_messages_are_joined_with_a_blank_line():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "um"},
                {"role": "developer", "content": "dois"},
                {"role": "user", "content": "oi"},
            ],
        },
        "claude-fable-5-1",
        4096,
    )
    assert out["system"] == "um\n\ndois"


def test_stop_as_a_list_becomes_stop_sequences_directly():
    out = openai_request_to_anthropic(
        {"model": "m", "messages": [{"role": "user", "content": "oi"}], "stop": ["FIM", "PARA"]},
        "claude-fable-5-1",
        4096,
    )
    assert out["stop_sequences"] == ["FIM", "PARA"]


def test_tool_message_with_no_content_key_becomes_empty_string_result():
    out = openai_request_to_anthropic(
        {"model": "m", "messages": [{"role": "tool", "tool_call_id": "call_1"}]},
        "claude-fable-5-1",
        4096,
    )
    assert out["messages"][0]["content"][0]["content"] == ""


def test_tool_use_block_with_no_input_key_becomes_empty_arguments_object():
    out = anthropic_response_to_openai(
        {
            "id": "msg_1",
            "model": "m",
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "read"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "gpt-4o",
    )
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_temperature_and_top_p_pass_through_when_present():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "oi"}],
            "temperature": 0.5,
            "top_p": 0.9,
        },
        "claude-fable-5-1",
        4096,
    )
    assert out["temperature"] == 0.5
    assert out["top_p"] == 0.9


def test_stream_flag_is_forwarded_when_true():
    out = openai_request_to_anthropic(
        {"model": "m", "messages": [{"role": "user", "content": "oi"}], "stream": True},
        "claude-fable-5-1",
        4096,
    )
    assert out["stream"] is True


def test_tool_call_with_no_arguments_key_becomes_empty_input():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "read"}}
                    ],
                }
            ],
        },
        "claude-fable-5-1",
        4096,
    )
    block = out["messages"][0]["content"][0]
    assert block["input"] == {}


def test_non_dict_content_list_entries_are_skipped():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": ["not a dict", {"type": "text", "text": "oi"}]}
            ],
        },
        "claude-fable-5-1",
        4096,
    )
    assert out["messages"][0]["content"] == [{"type": "text", "text": "oi"}]


def test_tool_missing_parameters_key_gets_the_empty_object_schema():
    out = openai_request_to_anthropic(
        {
            "model": "m",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "now", "description": "hora"}}],
        },
        "claude-fable-5-1",
        4096,
    )
    assert out["tools"][0]["input_schema"] == {"type": "object", "properties": {}}


def test_response_with_only_a_tool_use_block_has_no_text_content():
    out = anthropic_response_to_openai(
        {
            "id": "msg_1",
            "model": "claude-fable-5-1",
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a"}}
            ],
            "usage": {"input_tokens": 5, "output_tokens": 9},
        },
        "gpt-4o",
    )
    message = out["choices"][0]["message"]
    assert message["content"] is None
