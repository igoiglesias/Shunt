"""Translate an OpenAI chat-completion response into an Anthropic message.

This is the return leg of the pivot: a provider answered in OpenAI shape and
the client (Claude Code or any Anthropic-speaking harness) expects an
Anthropic Messages response back. Two shape mismatches matter most:

- OpenAI puts assistant text in `message.content` (a string or None) and
  tool calls in a separate `message.tool_calls` list with JSON-string
  `arguments`. Anthropic puts everything in one `content` list of typed
  blocks, `tool_use` blocks carrying a JSON object `input`. A response that
  is pure tool calls must not gain a stray `{"type": "text", "text": ...}`
  block -- emitting one with `text: None` is not valid Anthropic content at
  all, and even `text: ""` is a block the client did not ask for.
- The `model` field in the reply echoes the model the client requested, not
  whatever the provider actually ran, so a harness comparing that field
  against what it sent does not see a mismatch. The real model is exposed
  elsewhere (a response header), by a later task.

Deliberate leniency -- and why:

- A tool call's `arguments` string that fails to parse as JSON, or that
  parses to something other than a JSON object (e.g. a bare list), becomes
  an empty `input` dict rather than raising. Free/weak models emit malformed
  tool-call JSON often enough that treating it as fatal would take down the
  whole request instead of degrading one tool call.
- An empty `choices` list and a missing `usage` block are treated as "no
  data" rather than errors: content comes back empty, tokens come back
  zero, and the response is still shape-valid Anthropic JSON.
- An unrecognized `finish_reason` maps to `"end_turn"`, since that is the
  Anthropic stop reason least likely to make a harness treat an unfamiliar
  provider-specific reason as an error condition.

`STOP_REASONS` is part of this module's public interface -- Task 17's
streaming translator imports it directly -- not a private implementation
detail.
"""

import json
from typing import Any

from app.schemas.anthropic import AnthropicErrorResponse
from app.translate.ids import to_anthropic_id

STOP_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}

ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


def _to_msg_id(raw_id: str | None) -> str:
    """Turn an OpenAI response id into a sane, underscore-joined Anthropic one.

    Real Anthropic ids look like `msg_01ABC...` -- underscore, never a
    hyphen. A bare `raw_id.replace("chatcmpl", "msg", 1)` on
    `chatcmpl-abc123` used to leave the original hyphen in place and
    produce `msg-abc123`, a shape no real Anthropic id has. An id that
    never had the `chatcmpl` prefix is not passed through unchanged either
    -- that would leak a raw OpenAI-shaped id into an Anthropic-shaped
    field -- it is still rewritten into `msg_` form.
    """
    if not raw_id:
        return "msg_shunt"
    suffix = raw_id.removeprefix("chatcmpl")
    suffix = suffix.lstrip("-_")
    return f"msg_{suffix}" if suffix else "msg_shunt"


def _arguments(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def openai_response_to_anthropic(resp: dict, requested_model: str) -> dict:
    choices = resp.get("choices") or [{}]
    message = choices[0].get("message") or {}
    content: list[dict[str, Any]] = []

    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": to_anthropic_id(call.get("id", "")),
                "name": function.get("name", ""),
                "input": _arguments(function.get("arguments")),
            }
        )

    usage = resp.get("usage") or {}
    return {
        "id": _to_msg_id(resp.get("id")),
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": STOP_REASONS.get(choices[0].get("finish_reason", "stop"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def openai_error_to_anthropic(status: int, message: str) -> dict:
    return AnthropicErrorResponse.of(ERROR_TYPES.get(status, "api_error"), message).model_dump()
