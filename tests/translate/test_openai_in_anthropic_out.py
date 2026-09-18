import json

from app.translate.to_anthropic_request import openai_request_to_anthropic
from app.translate.to_openai import anthropic_response_to_openai


def test_system_message_moves_to_the_root_field():
    out = openai_request_to_anthropic({"model": "m", "messages": [
        {"role": "system", "content": "seja breve"},
        {"role": "user", "content": "oi"}]}, "claude-fable-5-1", 4096)
    assert out["system"] == "seja breve"
    assert [m["role"] for m in out["messages"]] == ["user"]


def test_max_tokens_is_required_by_anthropic_and_always_present():
    out = openai_request_to_anthropic(
        {"model": "m", "messages": [{"role": "user", "content": "oi"}]},
        "claude-fable-5-1", 4096)
    assert out["max_tokens"] == 4096


def test_tool_calls_become_tool_use_blocks():
    out = openai_request_to_anthropic({"model": "m", "messages": [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "read", "arguments": '{"path": "a"}'}}]}]},
        "claude-fable-5-1", 4096)
    block = out["messages"][0]["content"][0]
    assert block["type"] == "tool_use"
    assert block["input"] == {"path": "a"}
    assert block["id"].startswith("toolu_")


def test_tool_messages_become_tool_result_blocks_in_a_user_message():
    out = openai_request_to_anthropic({"model": "m", "messages": [
        {"role": "tool", "tool_call_id": "call_1", "content": "conteudo"}]},
        "claude-fable-5-1", 4096)
    message = out["messages"][0]
    assert message["role"] == "user"
    assert message["content"][0]["type"] == "tool_result"
    assert message["content"][0]["content"] == "conteudo"


def test_tools_move_to_the_anthropic_schema():
    out = openai_request_to_anthropic({"model": "m", "messages": [], "tools": [
        {"type": "function", "function": {"name": "read", "description": "le",
                                          "parameters": {"type": "object",
                                                         "properties": {}}}}],
        "tool_choice": "required"}, "claude-fable-5-1", 4096)
    assert out["tools"][0]["name"] == "read"
    assert out["tools"][0]["input_schema"] == {"type": "object", "properties": {}}
    assert out["tool_choice"] == {"type": "any"}


def test_anthropic_answer_becomes_an_openai_completion():
    out = anthropic_response_to_openai({
        "id": "msg_1", "model": "claude-fable-5-1", "stop_reason": "tool_use",
        "content": [{"type": "text", "text": "vou ler"},
                    {"type": "tool_use", "id": "toolu_1", "name": "read",
                     "input": {"path": "a"}}],
        "usage": {"input_tokens": 5, "output_tokens": 9}}, "gpt-4o")
    message = out["choices"][0]["message"]
    assert out["object"] == "chat.completion"
    assert out["model"] == "gpt-4o"
    assert message["content"] == "vou ler"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"path": "a"}
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert out["usage"]["prompt_tokens"] == 5


# --- Additional coverage beyond the brief ---


def test_consecutive_tool_messages_become_one_user_message_with_several_results():
    # Decision: consecutive `role: "tool"` messages collapse into a SINGLE
    # user message carrying multiple tool_result blocks, matching how the
    # assistant that requested them saw parallel tool calls answered
    # together, and avoiding a burst of near-empty user turns that a strict
    # provider could read as odd alternation.
    out = openai_request_to_anthropic({"model": "m", "messages": [
        {"role": "tool", "tool_call_id": "call_1", "content": "um"},
        {"role": "tool", "tool_call_id": "call_2", "content": "dois"},
    ]}, "claude-fable-5-1", 4096)
    assert len(out["messages"]) == 1
    message = out["messages"][0]
    assert message["role"] == "user"
    assert [b["type"] for b in message["content"]] == ["tool_result", "tool_result"]
    assert message["content"][0]["content"] == "um"
    assert message["content"][1]["content"] == "dois"


def test_assistant_message_with_text_and_tool_calls_keeps_both():
    out = openai_request_to_anthropic({"model": "m", "messages": [
        {"role": "assistant", "content": "vou checar",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "read", "arguments": "{}"}}]}]},
        "claude-fable-5-1", 4096)
    blocks = out["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "vou checar"}
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["name"] == "read"


def test_malformed_tool_call_arguments_become_empty_input():
    out = openai_request_to_anthropic({"model": "m", "messages": [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "read", "arguments": "{not json"}}]}]},
        "claude-fable-5-1", 4096)
    block = out["messages"][0]["content"][0]
    assert block["input"] == {}


def test_bare_string_stop_becomes_a_single_element_stop_sequences_list():
    out = openai_request_to_anthropic({"model": "m", "messages": [
        {"role": "user", "content": "oi"}], "stop": "FIM"},
        "claude-fable-5-1", 4096)
    assert out["stop_sequences"] == ["FIM"]


def test_system_message_in_the_middle_still_moves_to_the_root_field():
    out = openai_request_to_anthropic({"model": "m", "messages": [
        {"role": "user", "content": "oi"},
        {"role": "system", "content": "seja breve"},
        {"role": "user", "content": "tudo bem?"}]}, "claude-fable-5-1", 4096)
    assert out["system"] == "seja breve"
    assert [m["role"] for m in out["messages"]] == ["user", "user"]
