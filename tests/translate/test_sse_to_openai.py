from unittest.mock import patch

from app.translate.sse_to_openai import AnthropicStreamToOpenAI


def test_text_delta_becomes_a_content_chunk():
    tr = AnthropicStreamToOpenAI("gpt-4o", "chatcmpl-1")
    tr.feed("content_block_start", {"index": 0, "content_block": {"type": "text"}})
    chunks = tr.feed(
        "content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "oi"}}
    )
    assert chunks[0]["choices"][0]["delta"] == {"content": "oi"}
    assert chunks[0]["model"] == "gpt-4o"


def test_tool_use_block_opens_a_tool_call_with_name_and_id():
    tr = AnthropicStreamToOpenAI("m", "chatcmpl-1")
    chunks = tr.feed(
        "content_block_start",
        {"index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read"}},
    )
    call = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert call["index"] == 0
    assert call["function"]["name"] == "read"
    assert call["id"].startswith("call_")


def test_input_json_delta_becomes_argument_fragments():
    tr = AnthropicStreamToOpenAI("m", "chatcmpl-1")
    tr.feed(
        "content_block_start",
        {"index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read"}},
    )
    chunks = tr.feed(
        "content_block_delta",
        {"index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"path"'}},
    )
    assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"path"'


def test_message_delta_carries_finish_reason_and_usage():
    tr = AnthropicStreamToOpenAI("m", "chatcmpl-1")
    chunks = tr.feed(
        "message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}}
    )
    assert chunks[0]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[0]["usage"]["completion_tokens"] == 12


def test_ping_produces_nothing():
    assert AnthropicStreamToOpenAI("m", "c").feed("ping", {}) == []


def test_two_tool_blocks_get_positional_tool_call_index():
    tr = AnthropicStreamToOpenAI("m", "chatcmpl-1")
    first = tr.feed(
        "content_block_start",
        {"index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read"}},
    )
    tr.feed("content_block_stop", {"index": 0})
    second = tr.feed(
        "content_block_start",
        {"index": 1, "content_block": {"type": "tool_use", "id": "toolu_2", "name": "write"}},
    )
    assert first[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert second[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 1


def test_tool_index_counts_tool_calls_not_content_block_index():
    tr = AnthropicStreamToOpenAI("m", "chatcmpl-1")
    tr.feed("content_block_start", {"index": 0, "content_block": {"type": "text"}})
    tr.feed("content_block_stop", {"index": 0})
    chunks = tr.feed(
        "content_block_start",
        {"index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read"}},
    )
    call = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert call["index"] == 0


def test_input_json_delta_maps_through_block_index_to_tool_index():
    tr = AnthropicStreamToOpenAI("m", "chatcmpl-1")
    tr.feed("content_block_start", {"index": 0, "content_block": {"type": "text"}})
    tr.feed("content_block_stop", {"index": 0})
    tr.feed(
        "content_block_start",
        {"index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read"}},
    )
    chunks = tr.feed(
        "content_block_delta",
        {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"path"'}},
    )
    call = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert call["index"] == 0
    assert call["function"]["arguments"] == '{"path"'


def test_text_block_after_tool_block_still_emits_content():
    tr = AnthropicStreamToOpenAI("m", "chatcmpl-1")
    tr.feed(
        "content_block_start",
        {"index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read"}},
    )
    tr.feed("content_block_stop", {"index": 0})
    tr.feed("content_block_start", {"index": 1, "content_block": {"type": "text"}})
    chunks = tr.feed(
        "content_block_delta", {"index": 1, "delta": {"type": "text_delta", "text": "done"}}
    )
    assert chunks[0]["choices"][0]["delta"] == {"content": "done"}


def test_content_block_stop_and_message_stop_produce_nothing():
    tr = AnthropicStreamToOpenAI("m", "chatcmpl-1")
    assert tr.feed("content_block_stop", {"index": 0}) == []
    assert tr.feed("message_stop", {}) == []


def test_unknown_stop_reason_falls_back_to_stop():
    tr = AnthropicStreamToOpenAI("m", "chatcmpl-1")
    chunks = tr.feed("message_delta", {"delta": {"stop_reason": "something_new"}, "usage": {}})
    assert chunks[0]["choices"][0]["finish_reason"] == "stop"


def test_message_delta_without_usage_defaults_completion_tokens_to_zero():
    tr = AnthropicStreamToOpenAI("m", "chatcmpl-1")
    chunks = tr.feed("message_delta", {"delta": {"stop_reason": "end_turn"}})
    assert chunks[0]["usage"]["completion_tokens"] == 0


def test_unknown_event_name_produces_nothing():
    assert AnthropicStreamToOpenAI("m", "c").feed("some_future_event", {"whatever": 1}) == []


def test_content_block_delta_with_unrecognized_delta_type_produces_nothing():
    tr = AnthropicStreamToOpenAI("m", "chatcmpl-1")
    assert (
        tr.feed(
            "content_block_delta",
            {"index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
        )
        == []
    )


def test_every_chunk_carries_the_same_nonzero_created_timestamp():
    tr = AnthropicStreamToOpenAI("m", "chatcmpl-1")
    chunks = []
    chunks += tr.feed("content_block_start", {"index": 0, "content_block": {"type": "text"}})
    chunks += tr.feed(
        "content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "hi"}}
    )
    chunks += tr.feed(
        "content_block_start",
        {"index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read"}},
    )
    chunks += tr.feed("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {}})
    created_values = {c["created"] for c in chunks}
    assert len(created_values) == 1
    assert created_values.pop() > 0


def test_created_is_captured_once_at_construction_not_per_chunk():
    # A real clock advancing between calls would mask a "captured once"
    # regression at test speed (the timestamps still match by luck). Force
    # the clock to move between feed() calls so a per-chunk capture is
    # caught deterministically.
    with patch("app.translate.sse_to_openai.time.time", side_effect=[100.0, 200.0, 300.0]):
        tr = AnthropicStreamToOpenAI("m", "chatcmpl-1")
        first = tr.feed(
            "content_block_start",
            {"index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read"}},
        )
        second = tr.feed(
            "content_block_delta",
            {"index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        )
    assert first[0]["created"] == 100
    assert second[0]["created"] == 100


def test_input_json_delta_for_a_block_that_was_never_opened_is_dropped():
    tr = AnthropicStreamToOpenAI("m", "chatcmpl-1")
    chunks = tr.feed(
        "content_block_delta",
        {"index": 7, "delta": {"type": "input_json_delta", "partial_json": '{"path"'}},
    )
    assert chunks == []


def test_finish_is_idempotent():
    tr = AnthropicStreamToOpenAI("m", "chatcmpl-1")
    tr.feed("content_block_start", {"index": 0, "content_block": {"type": "text"}})
    assert tr.finish() == []
    assert tr.finish() == []
