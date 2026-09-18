"""Stateless mapping of tool-call ids between the Anthropic and OpenAI dialects.

The proxy is stateless across requests: nothing persists between turn one
(the provider emits an id) and turn two (the client sends that same id back
inside a tool_result). So the mapping in both directions has to be a pure
function of the id string itself, never a lookup into memory.

- to_anthropic_id: always reversible. It base64-encodes the OpenAI id into
  the tail of a toolu_-prefixed string, so to_openai_id can recover the
  exact original.
- to_openai_id: if the given Anthropic id was produced by to_anthropic_id
  (marked and valid base64), decode it back to the original OpenAI id. For
  any other Anthropic id -- one the Shunt never encoded, e.g. a native
  Claude Code id -- there is no original to recover, so a deterministic
  hash-derived id is returned instead. Determinism is what keeps the second
  turn of a conversation matching the first.
"""

import base64
import hashlib

ANTHROPIC_PREFIX = "toolu_"
OPENAI_PREFIX = "call_"
ENCODED_MARK = "s"  # marks that what follows is a Shunt-encoded provider id


def to_anthropic_id(openai_id: str) -> str:
    raw = base64.urlsafe_b64encode(openai_id.encode()).decode().rstrip("=")
    return f"{ANTHROPIC_PREFIX}{ENCODED_MARK}{raw}"


def to_openai_id(anthropic_id: str) -> str:
    body = anthropic_id.removeprefix(ANTHROPIC_PREFIX)
    if body.startswith(ENCODED_MARK):
        encoded = body[len(ENCODED_MARK) :]
        padding = "=" * (-len(encoded) % 4)
        try:
            return base64.urlsafe_b64decode(encoded + padding).decode()
        except (ValueError, UnicodeDecodeError):
            pass
    digest = hashlib.sha256(anthropic_id.encode()).hexdigest()[:24]
    return f"{OPENAI_PREFIX}{digest}"
