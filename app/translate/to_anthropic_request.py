"""Translate an OpenAI chat-completion request into an Anthropic Messages request.

This is the mirror of `to_openai.py`'s `anthropic_request_to_openai`: a
harness that speaks OpenAI (opencode, hermes) points at a model whose
candidate is an Anthropic-native provider. The two request shapes disagree
in ways that are not merely cosmetic:

- Anthropic has no `system` role inside `messages` -- system content lives
  in a top-level `system` string, gathered here from every `system` (or
  `developer`) message regardless of where it appears in the conversation.
- Anthropic has no `tool` role at all. An OpenAI `role: "tool"` message
  becomes a `tool_result` content block inside a `role: "user"` message --
  the asymmetric mirror of Task 8/9's `tool_result` -> `role: "tool"`
  message. Consecutive `role: "tool"` messages are collapsed into ONE user
  message carrying several `tool_result` blocks, matching how the
  assistant that made several tool calls at once expects to see them
  answered together, and avoiding a burst of near-empty user turns that a
  strict provider could read as broken alternation.
- `max_tokens` is required by the Anthropic API and optional in OpenAI's.
  When the inbound request omits it, `max_output_tokens` (the candidate's
  configured ceiling) fills it in -- a request that reaches the provider
  without `max_tokens` is rejected outright, so this is not a nicety.
- OpenAI's `arguments` on a tool call is a JSON string; Anthropic's
  `input` on a `tool_use` block is a JSON object. A malformed or
  non-object `arguments` string degrades to an empty `input` dict rather
  than raising, matching the leniency `to_anthropic.py` documents for the
  same shape of problem on the return leg -- one bad tool call should not
  take down the whole request.

Deliberate omissions -- fields this translator does not add, and why:

- No `thinking` or `cache_control` field is emitted. Nothing inbound from
  an OpenAI-speaking harness carries the concepts either represents, so
  there is nothing to translate.
"""

import json
from typing import Any

from app.translate.effort import normalize_effort
from app.translate.ids import to_anthropic_id

TOOL_CHOICE = {
    "auto": {"type": "auto"},
    "required": {"type": "any"},
    "none": {"type": "none"},
}


def _arguments(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    header, _, data = url.partition(",")
                    media_type = header.removeprefix("data:").split(";")[0]
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": data},
                        }
                    )
                else:
                    blocks.append({"type": "image", "source": {"type": "url", "url": url}})
    return blocks


def _convert_messages(raw_messages: list[dict]) -> tuple[list[str], list[dict[str, Any]]]:
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] | None = None

    def flush_tool_results() -> None:
        nonlocal pending_tool_results
        if pending_tool_results:
            messages.append({"role": "user", "content": pending_tool_results})
        pending_tool_results = None

    for message in raw_messages:
        role = message.get("role")
        content = message.get("content")

        if role in ("system", "developer"):
            system_parts.append(content if isinstance(content, str) else "")
            continue

        if role == "tool":
            if pending_tool_results is None:
                pending_tool_results = []
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": to_anthropic_id(message.get("tool_call_id", "")),
                    "content": content or "",
                }
            )
            continue

        # Any non-tool message ends a run of consecutive tool messages.
        flush_tool_results()

        blocks = _content_blocks(content)
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": to_anthropic_id(call.get("id", "")),
                    "name": function.get("name", ""),
                    "input": _arguments(function.get("arguments")),
                }
            )
        if blocks:
            messages.append({"role": role, "content": blocks})

    flush_tool_results()
    return system_parts, messages


def openai_request_to_anthropic(body: dict, target_model: str, max_output_tokens: int) -> dict:
    system_parts, messages = _convert_messages(body.get("messages", []))

    out: dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "max_tokens": min(int(body.get("max_tokens") or max_output_tokens), max_output_tokens),
    }
    if system_parts:
        out["system"] = "\n\n".join(p for p in system_parts if p)
    for key in ("temperature", "top_p"):
        if body.get(key) is not None:
            out[key] = body[key]
    if body.get("stream"):
        out["stream"] = True
    if body.get("tools"):
        out["tools"] = [
            {
                "name": (t.get("function") or {}).get("name", ""),
                "description": (t.get("function") or {}).get("description", ""),
                "input_schema": (t.get("function") or {}).get("parameters")
                or {"type": "object", "properties": {}},
            }
            for t in body["tools"]
        ]
    choice = body.get("tool_choice")
    if isinstance(choice, str):
        out["tool_choice"] = TOOL_CHOICE.get(choice, {"type": "auto"})
    elif isinstance(choice, dict):
        out["tool_choice"] = {
            "type": "tool",
            "name": (choice.get("function") or {}).get("name", ""),
        }
    if body.get("parallel_tool_calls") is False:
        out.setdefault("tool_choice", {"type": "auto"})["disable_parallel_tool_use"] = True
    stop = body.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    # `reasoning_effort` e o campo oficial; `reasoning.effort` so vale quando ele
    # esta ausente (ou nulo). Presente e invalido e descartado, sem fallback.
    # O tradutor nunca cria `thinking` (spec R4).
    raw_effort = body.get("reasoning_effort")
    if raw_effort is None:
        reasoning = body.get("reasoning")
        if isinstance(reasoning, dict):
            raw_effort = reasoning.get("effort")
    effort = normalize_effort(raw_effort)
    if effort is not None:
        out["output_config"] = {"effort": effort}
    return out
