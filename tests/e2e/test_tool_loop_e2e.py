"""O loop agentico inteiro: tool_use ida e volta, com e sem streaming.

O que nenhuma tarefa anterior cobriu: o id da tool precisa sobreviver a duas
traducoes consecutivas, e o stream precisa remontar argumentos que chegam
partidos no meio de um JSON.
"""

import json

from tests.e2e.fake_provider import sse_chunks


def events_of(text):
    return [
        line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")
    ]


def datas_of(text):
    return [
        json.loads(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def test_two_turn_tool_loop_keeps_the_tool_id_linked(shunt, provider):
    """Turno 1 devolve tool_use; o cliente responde com tool_result no turno 2.

    O id precisa voltar ao provedor como o mesmo `tool_call_id` que ele emitiu.
    """
    provider.queue(
        json_body={
            "id": "chatcmpl-1",
            "model": "qwen3-8b",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_xyz",
                                "type": "function",
                                "function": {"name": "read", "arguments": '{"path": "a.txt"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )
    first = shunt.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "leia a.txt"}],
            "tools": [
                {
                    "name": "read",
                    "description": "le",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            ],
        },
    ).json()
    assert first["stop_reason"] == "tool_use"
    tool_use = first["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["input"] == {"path": "a.txt"}

    provider.queue(
        json_body={
            "id": "chatcmpl-2",
            "model": "qwen3-8b",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "arquivo lido"},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    shunt.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-5",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "leia a.txt"},
                {"role": "assistant", "content": [tool_use]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": "conteudo do arquivo",
                        }
                    ],
                },
            ],
        },
    )
    sent = provider.bodies[1]["messages"]
    assistant = [m for m in sent if m.get("tool_calls")][0]
    tool_message = [m for m in sent if m["role"] == "tool"][0]
    assert assistant["tool_calls"][0]["id"] == "call_xyz"
    assert tool_message["tool_call_id"] == "call_xyz"


def test_streaming_tool_call_arrives_fragmented_and_is_reassembled(shunt, provider):
    provider.queue(
        sse=sse_chunks(
            {"id": "chatcmpl-1", "choices": [{"delta": {"content": "vou ler"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "read", "arguments": '{"pa'},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": 'th": "a.txt"}'}}
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
        )
    )
    with shunt.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "claude-opus-4-5",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "leia"}],
            "tools": [{"name": "read", "input_schema": {"type": "object"}}],
        },
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert events_of(body) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    fragments = [
        d["delta"]["partial_json"]
        for d in datas_of(body)
        if d.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert json.loads("".join(fragments)) == {"path": "a.txt"}
    closing = [d for d in datas_of(body) if d["type"] == "message_delta"][0]
    assert closing["delta"]["stop_reason"] == "tool_use"
    assert closing["usage"]["output_tokens"] == 7


def test_streaming_request_asks_the_provider_for_usage(shunt, provider):
    provider.queue(sse=sse_chunks({"choices": [{"delta": {"content": "oi"}}]}))
    with shunt.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "claude-opus-4-5",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "oi"}],
        },
    ) as response:
        "".join(response.iter_text())
    assert provider.bodies[0]["stream_options"] == {"include_usage": True}


def test_max_tokens_is_clamped_to_the_candidate_limit(shunt, provider):
    provider.queue(
        json_body={
            "id": "c",
            "model": "qwen3-8b",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
    )
    shunt.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-5",
            "max_tokens": 64000,
            "messages": [{"role": "user", "content": "oi"}],
        },
    )
    assert provider.bodies[0]["max_tokens"] == 4096
