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


def test_chatcmpl_id_becomes_a_msg_id():
    out = openai_response_to_anthropic({
        "id": "chatcmpl-abc123", "model": "m",
        "choices": [{"message": {"content": "oi"}, "finish_reason": "stop"}],
    }, "m")
    assert out["id"] == "msg-abc123"


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
