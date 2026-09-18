"""Translate an Anthropic Messages request into the OpenAI pivot format.

The two dialects disagree in ways that produce silence rather than errors:
Anthropic puts a `tool_use` block inside the assistant message and the
matching `tool_result` inside a *user* message; OpenAI puts `tool_calls` on
the assistant message and each result in its own separate message with
`role: "tool"`. Anthropic's tool input is a JSON object; OpenAI's
`arguments` is a JSON string. Get any of that wrong and the provider either
rejects the request or answers in prose while the harness waits for a tool
call that never comes.

Deliberate drops -- fields or shapes this translator does not carry across,
and why:

- `top_k` is dropped, not forwarded. It has no place in the OpenAI request
  schema, and a strict target (`api.openai.com` included) rejects unknown
  parameters outright -- forwarding it would turn a harmless sampling hint
  into a failed request.
- `metadata` is dropped. Nothing in the OpenAI request shape corresponds to
  it, so there is nowhere to carry it and no receiver would read it.

Known limitations -- behaviour that is intentionally left as-is, with its
consequence stated so it is not mistaken for an oversight:

- An image inside a `tool_result` is lost: `_result_text` keeps only text
  blocks from a tool result's content list. A screenshot-returning tool
  therefore reports an empty result to the model.
- An assistant message made up only of `thinking` blocks (no text, no
  tool_use) produces no output message at all -- `_convert_message` returns
  an empty list for it. On a provider that enforces strict user/assistant
  alternation, this can collapse two turns together and break that
  alternation.

`anthropic_response_to_openai` is the return leg for the mirror direction
(OpenAI in, Anthropic-native provider out, from Task 9b): an Anthropic
Messages response comes back from the provider and an OpenAI-speaking
harness expects a `chat.completion` object. As on the outbound side, a
`tool_use` block's `input` object becomes a JSON *string* in
`function.arguments`, and text blocks are joined into the single
`message.content` string OpenAI expects -- there is no per-block content
list on that side.
"""

import json
from typing import Any

from app.translate.ids import to_openai_id

MAX_STOP_SEQUENCES = 4

FINISH_REASONS = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
}


class TooManyStopSequencesError(ValueError):
    """Raised when the request asks for more stop sequences than OpenAI accepts."""


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    return "\n\n".join(
        block.get("text", "")
        for block in system
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _content_part(block: dict) -> dict | None:
    if block.get("type") == "text":
        return {"type": "text", "text": block.get("text", "")}
    if block.get("type") == "image":
        source = block.get("source", {})
        if source.get("type") == "base64":
            url = f"data:{source.get('media_type')};base64,{source.get('data')}"
        else:
            url = source.get("url", "")
        return {"type": "image_url", "image_url": {"url": url}}
    return None


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _convert_message(message: dict) -> list[dict]:
    role = message.get("role")
    content = message.get("content")
    if isinstance(content, str):
        return [{"role": role, "content": content}]

    tool_messages: list[dict] = []
    parts: list[dict] = []
    tool_calls: list[dict] = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "tool_use":
            tool_calls.append(
                {
                    "id": to_openai_id(block.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )
        elif kind == "tool_result":
            text = _result_text(block.get("content", ""))
            if block.get("is_error"):
                text = f"Error: {text}"
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": to_openai_id(block.get("tool_use_id", "")),
                    "content": text,
                }
            )
        else:
            part = _content_part(block)
            if part is not None:
                parts.append(part)

    # Order matters: `tool_result` blocks become `role: "tool"` messages in
    # the order the tool_calls were made, and a text block sharing a message
    # with a tool_result becomes a separate user message AFTER those tool
    # messages -- a strict provider rejects any other order.
    out: list[dict] = list(tool_messages)
    if tool_calls:
        # OpenAI carries text and tool_calls on the SAME assistant message
        # object (content is a sibling field of tool_calls, not a separate
        # message) -- so any text emitted alongside a tool_use is folded
        # into this one message rather than sent as a second message.
        call_content: Any = None
        if parts:
            text_only = all(p["type"] == "text" for p in parts)
            call_content = "\n".join(p["text"] for p in parts) if text_only else parts
            parts = []
        out.append({"role": role, "content": call_content, "tool_calls": tool_calls})
    if parts:
        text_only = all(p["type"] == "text" for p in parts)
        payload = "\n".join(p["text"] for p in parts) if text_only else parts
        out.append({"role": role, "content": payload})
    return out


def _convert_tool(tool: dict) -> dict:
    schema = tool.get("input_schema") or {}
    if not schema:
        schema = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": schema,
        },
    }


def _convert_tool_choice(choice: dict) -> tuple[Any, bool | None]:
    mapping = {"auto": "auto", "any": "required", "none": "none"}
    kind = choice.get("type", "auto")
    parallel = None
    if choice.get("disable_parallel_tool_use") is True:
        parallel = False
    if kind == "tool":
        return {"type": "function", "function": {"name": choice.get("name", "")}}, parallel
    return mapping.get(kind, "auto"), parallel


def anthropic_request_to_openai(body: dict, target_model: str, max_output_tokens: int) -> dict:
    messages: list[dict] = []
    if body.get("system"):
        messages.append({"role": "system", "content": _system_text(body["system"])})
    for message in body.get("messages", []):
        messages.extend(_convert_message(message))

    out: dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "max_tokens": min(int(body.get("max_tokens", max_output_tokens)), max_output_tokens),
    }
    for key in ("temperature", "top_p"):
        if body.get(key) is not None:
            out[key] = body[key]
    if body.get("stream"):
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    if body.get("tools"):
        out["tools"] = [_convert_tool(t) for t in body["tools"]]
    if body.get("tool_choice"):
        choice, parallel = _convert_tool_choice(body["tool_choice"])
        out["tool_choice"] = choice
        if parallel is not None:
            out["parallel_tool_calls"] = parallel
    stops = body.get("stop_sequences")
    if stops:
        if len(stops) > MAX_STOP_SEQUENCES:
            raise TooManyStopSequencesError(
                f"OpenAI accepts at most {MAX_STOP_SEQUENCES} stop sequences, got {len(stops)}"
            )
        out["stop"] = stops
    return out


def _to_chatcmpl_id(raw_id: str | None) -> str:
    """Turn an Anthropic response id into a sane, hyphen-joined OpenAI one.

    Real OpenAI completion ids look like `chatcmpl-01ABC...` -- hyphen,
    never an underscore. A bare `raw_id.replace("msg", "chatcmpl", 1)` on
    `msg_01ABC` used to leave the original underscore in place and
    produce `chatcmpl_01ABC`, a shape no real OpenAI id has. This is the
    mirror of `_to_msg_id` above; see that function's docstring for why
    an id lacking the expected prefix is still rewritten rather than
    passed through unchanged.
    """
    if not raw_id:
        return "chatcmpl_shunt"
    suffix = raw_id.removeprefix("msg")
    suffix = suffix.lstrip("-_")
    return f"chatcmpl-{suffix}" if suffix else "chatcmpl_shunt"


def anthropic_response_to_openai(resp: dict, requested_model: str) -> dict:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in resp.get("content") or []:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": to_openai_id(block.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )

    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = resp.get("usage") or {}
    return {
        "id": _to_chatcmpl_id(resp.get("id")),
        "object": "chat.completion",
        "created": 0,
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": FINISH_REASONS.get(resp.get("stop_reason", "end_turn"), "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        },
    }
