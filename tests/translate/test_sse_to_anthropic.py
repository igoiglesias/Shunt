from app.translate.sse_to_anthropic import OpenAIStreamToAnthropic


def names(events):
    return [name for name, _ in events]


def test_first_text_chunk_opens_message_and_block():
    tr = OpenAIStreamToAnthropic("claude-opus-4-5", "msg_1")
    events = tr.feed(
        {"id": "chatcmpl-1", "model": "vendor/free", "choices": [{"delta": {"content": "Oi"}}]}
    )
    assert names(events) == ["message_start", "content_block_start", "content_block_delta"]
    assert events[0][1]["message"]["model"] == "claude-opus-4-5"
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
    assert names(tr.finish()) == ["message_start", "message_delta", "message_stop"]


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
