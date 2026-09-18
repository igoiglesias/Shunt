"""Stateless mapping of tool-call ids between the Anthropic and OpenAI dialects.

The proxy is stateless across requests: nothing persists between turn one
(the provider emits an id) and turn two (the client sends that same id back
inside a tool_result). So the mapping in both directions has to be a pure
function of the id string itself, never a lookup into memory.

- to_anthropic_id: always reversible. It base64-encodes the OpenAI id,
  together with a short checksum of the payload, into the tail of a
  toolu_-prefixed string, so to_openai_id can recover the exact original.
- to_openai_id: if the given Anthropic id was produced by to_anthropic_id
  (marked, valid base64, and the embedded checksum verifies), decode it
  back to the original OpenAI id. For any other Anthropic id -- one the
  Shunt never encoded, e.g. a native Claude Code id -- there is no
  original to recover, so a deterministic hash-derived id is returned
  instead. Determinism is what keeps the second turn of a conversation
  matching the first.

  The checksum matters because a native Anthropic id (toolu_ + random
  alphanumerics) can, by accident, start with the marker and have its
  remainder decode as valid base64url *and* valid UTF-8 -- alphanumerics
  are a subset of the base64url alphabet, so "looks like our encoding" is
  not enough to prove it is our encoding. The checksum is a correctness
  discriminator, not a security boundary: a plain truncated sha256 over
  the payload is sufficient, and deliberately unkeyed so the whole scheme
  stays a pure function of the id (a keyed MAC would need the key to
  survive restarts to keep determinism, which reintroduces state).
"""

import base64
import hashlib

ANTHROPIC_PREFIX = "toolu_"
OPENAI_PREFIX = "call_"
ENCODED_MARK = "s"  # marks that what follows is a Shunt-encoded provider id
CHECKSUM_LEN = 4  # bytes of sha256(payload) appended before base64-encoding


def to_anthropic_id(openai_id: str) -> str:
    payload = openai_id.encode()
    checksum = hashlib.sha256(payload).digest()[:CHECKSUM_LEN]
    raw = base64.urlsafe_b64encode(payload + checksum).decode().rstrip("=")
    return f"{ANTHROPIC_PREFIX}{ENCODED_MARK}{raw}"


def to_openai_id(anthropic_id: str) -> str:
    body = anthropic_id.removeprefix(ANTHROPIC_PREFIX)
    if body.startswith(ENCODED_MARK):
        encoded = body[len(ENCODED_MARK) :]
        padding = "=" * (-len(encoded) % 4)
        try:
            combined = base64.urlsafe_b64decode(encoded + padding)
            if len(combined) >= CHECKSUM_LEN:
                payload, checksum = combined[:-CHECKSUM_LEN], combined[-CHECKSUM_LEN:]
                if hashlib.sha256(payload).digest()[:CHECKSUM_LEN] == checksum:
                    return payload.decode()
        except ValueError, UnicodeDecodeError:
            pass
    digest = hashlib.sha256(anthropic_id.encode()).hexdigest()[:24]
    return f"{OPENAI_PREFIX}{digest}"
