from app.translate.to_anthropic import openai_error_to_anthropic, openai_response_to_anthropic


def test_text_answer_becomes_a_single_text_block():
    out = openai_response_to_anthropic({
        "id": "chatcmpl-1", "model": "vendor/free",
        "choices": [{"message": {"role": "assistant", "content": "oi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }, "claude-opus-4-5")
    assert out["content"] == [{"type": "text", "text": "oi"}]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"] == {"input_tokens": 7, "output_tokens": 3}


def test_response_echoes_the_requested_model_not_the_real_one():
    out = openai_response_to_anthropic({
        "id": "chatcmpl-1", "model": "vendor/free",
        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
    }, "claude-opus-4-5")
    assert out["model"] == "claude-opus-4-5"


def test_tool_only_answer_has_no_empty_text_block():
    out = openai_response_to_anthropic({
        "id": "chatcmpl-2", "model": "m",
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "read", "arguments": '{"path": "a"}'}}]},
            "finish_reason": "tool_calls"}],
    }, "claude-opus-4-5")
    assert [b["type"] for b in out["content"]] == ["tool_use"]
    assert out["content"][0]["input"] == {"path": "a"}
    assert out["content"][0]["id"].startswith("toolu_")
    assert out["stop_reason"] == "tool_use"


def test_text_and_tool_produce_two_blocks_in_order():
    out = openai_response_to_anthropic({
        "id": "c", "model": "m",
        "choices": [{"message": {"content": "vou ler", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "read", "arguments": "{}"}}]},
            "finish_reason": "tool_calls"}],
    }, "m")
    assert [b["type"] for b in out["content"]] == ["text", "tool_use"]


def test_missing_arguments_string_becomes_empty_input():
    out = openai_response_to_anthropic({
        "id": "c", "model": "m",
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "read"}}]},
            "finish_reason": "tool_calls"}],
    }, "m")
    assert out["content"][0]["input"] == {}


def test_invalid_tool_arguments_do_not_raise():
    out = openai_response_to_anthropic({
        "id": "c", "model": "m",
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "read", "arguments": "{path: broken"}}]},
            "finish_reason": "tool_calls"}],
    }, "m")
    assert out["content"][0]["input"] == {}


def test_error_envelope_maps_status_to_anthropic_type():
    out = openai_error_to_anthropic(429, "slow down")
    assert out == {"type": "error",
                   "error": {"type": "rate_limit_error", "message": "slow down"}}
    assert openai_error_to_anthropic(529, "busy")["error"]["type"] == "overloaded_error"
    assert openai_error_to_anthropic(418, "?")["error"]["type"] == "api_error"


def test_chatcmpl_id_becomes_a_msg_id_with_underscore():
    # Real Anthropic ids use an underscore (msg_01ABC...), never a hyphen.
    # A prior version of this transform did a bare
    # "chatcmpl" -> "msg" string replace, which left the original hyphen
    # in place and produced "msg-abc123" -- a shape no real Anthropic id
    # has. Reverting the underscore fix would turn this green again.
    out = openai_response_to_anthropic({
        "id": "chatcmpl-abc123", "model": "m",
        "choices": [{"message": {"content": "oi"}, "finish_reason": "stop"}],
    }, "m")
    assert out["id"] == "msg_abc123"


def test_id_not_starting_with_chatcmpl_still_gets_a_sane_msg_form():
    # An id that never had the "chatcmpl" prefix must not be passed through
    # unchanged (that would leak a raw OpenAI-shaped id into an
    # Anthropic-shaped field) -- it still gets rewritten into "msg_" form.
    out = openai_response_to_anthropic({
        "id": "weird-vendor-id-123", "model": "m",
        "choices": [{"message": {"content": "oi"}, "finish_reason": "stop"}],
    }, "m")
    assert out["id"] == "msg_weird-vendor-id-123"
    assert not out["id"].startswith("weird")


def test_missing_id_falls_back_to_msg_shunt():
    out = openai_response_to_anthropic({
        "model": "m",
        "choices": [{"message": {"content": "oi"}, "finish_reason": "stop"}],
    }, "m")
    assert out["id"] == "msg_shunt"


def test_empty_choices_list_falls_back_to_defaults():
    out = openai_response_to_anthropic({
        "id": "chatcmpl-3", "model": "m", "choices": [],
    }, "claude-opus-4-5")
    assert out["content"] == []
    assert out["stop_reason"] == "end_turn"


def test_missing_usage_defaults_to_zero_tokens():
    out = openai_response_to_anthropic({
        "id": "chatcmpl-4", "model": "m",
        "choices": [{"message": {"content": "oi"}, "finish_reason": "stop"}],
    }, "m")
    assert out["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_every_known_finish_reason_maps_to_its_own_stop_reason():
    # Only "stop" and "tool_calls" were pinned elsewhere in this file --
    # "length", "function_call" and "content_filter" all default to the
    # same "end_turn" fallback an unrecognized value produces, so swapping
    # any one of their STOP_REASONS entries for "end_turn" left the suite
    # green before this test existed.
    cases = {
        "length": "max_tokens",
        "function_call": "tool_use",
        "content_filter": "end_turn",
    }
    for finish_reason, expected in cases.items():
        out = openai_response_to_anthropic({
            "id": "chatcmpl-1", "model": "m",
            "choices": [{"message": {"content": "oi"}, "finish_reason": finish_reason}],
        }, "m")
        assert out["stop_reason"] == expected, finish_reason


def test_unknown_finish_reason_falls_back_to_end_turn():
    out = openai_response_to_anthropic({
        "id": "chatcmpl-5", "model": "m",
        "choices": [{"message": {"content": "oi"}, "finish_reason": "something_new"}],
    }, "m")
    assert out["stop_reason"] == "end_turn"


def test_tool_arguments_valid_json_but_not_an_object_becomes_empty_input():
    out = openai_response_to_anthropic({
        "id": "chatcmpl-6", "model": "m",
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "read", "arguments": "[1, 2, 3]"}}]},
            "finish_reason": "tool_calls"}],
    }, "m")
    assert out["content"][0]["input"] == {}
