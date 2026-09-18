"""Convert an Anthropic-shaped SSE stream into OpenAI chat-completion chunks.

Mirror of `app.translate.sse_to_anthropic.OpenAIStreamToAnthropic`. The
asymmetry between the two directions is real: Task 17's translator had to
*invent* content-block boundaries that OpenAI's stream never states. Here
the input already states them explicitly (`content_block_start` /
`content_block_stop`), so the work is the opposite -- flattening Anthropic's
per-block numbering down into OpenAI's positional `tool_calls[].index`,
which counts only tool calls, not content blocks. A text block and a tool
block can both be Anthropic content-block index 0 while occupying different
(or no) OpenAI tool-call slots; a text block after a tool block is common
(e.g. a trailing sentence after a tool call) and must not disturb the tool
call's index.

So the translator remembers the mapping from an open Anthropic block index
to the OpenAI tool-call index assigned to it, and a running counter of how
many tool calls have been opened so far.

Deliberate leniency, matching the rest of `app.translate`:

- `input_json_delta.partial_json` fragments are passed through verbatim,
  never parsed or repaired here.
- An unrecognized `stop_reason` falls back to `FINISH_REASONS`'s implicit
  default of `"stop"`, imported from `app.translate.to_openai` rather than
  duplicated.
- `ping`, `content_block_stop`, `message_stop`, and any event name this
  translator does not recognise, produce no chunk -- OpenAI's chunk stream
  has no equivalent for a content-block boundary or a keep-alive.
- `finish()` is a no-op: unlike the Anthropic direction, OpenAI's stream
  needs no closing chunk this translator must synthesize (the dispatcher
  appends `[DONE]` on its own). It still exists, and is idempotent, to
  satisfy the same controller contract as Task 17's `finish()` and to leave
  room for a future need (e.g. flushing a trailing chunk) without changing
  the interface again.
"""

from typing import Any

from app.translate.ids import to_openai_id
from app.translate.to_openai import FINISH_REASONS


class AnthropicStreamToOpenAI:
    """Stateful translator: one instance per in-flight stream.

    `finish()` is safe to call more than once, for the same reason as
    Task 17's mirror: the dispatcher may call it both at the natural end
    of the loop and again from a `finally` guarding the crash path.
    """

    def __init__(self, requested_model: str, completion_id: str) -> None:
        self._model = requested_model
        self._id = completion_id
        self._tool_index_of_block: dict[int, int] = {}
        self._next_tool_index = 0
        self._finished = False

    def _chunk(
        self, delta: dict, finish: str | None = None, usage: dict | None = None
    ) -> dict:
        payload: dict[str, Any] = {
            "id": self._id,
            "object": "chat.completion.chunk",
            "model": self._model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage is not None:
            payload["usage"] = usage
        return payload

    def _content_block_start(self, data: dict) -> list[dict]:
        block = data.get("content_block") or {}
        if block.get("type") != "tool_use":
            return []
        index = self._next_tool_index
        self._next_tool_index += 1
        self._tool_index_of_block[data.get("index", 0)] = index
        return [
            self._chunk(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": to_openai_id(block.get("id", "")),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": "",
                            },
                        }
                    ]
                }
            )
        ]

    def _content_block_delta(self, data: dict) -> list[dict]:
        delta = data.get("delta") or {}
        if delta.get("type") == "text_delta":
            return [self._chunk({"content": delta.get("text", "")})]
        if delta.get("type") == "input_json_delta":
            index = self._tool_index_of_block.get(data.get("index", 0), 0)
            return [
                self._chunk(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "function": {
                                    "arguments": delta.get("partial_json", "")
                                },
                            }
                        ]
                    }
                )
            ]
        return []

    def _message_delta(self, data: dict) -> list[dict]:
        reason = (data.get("delta") or {}).get("stop_reason", "end_turn")
        usage = data.get("usage") or {}
        return [
            self._chunk(
                {},
                FINISH_REASONS.get(reason, "stop"),
                {"completion_tokens": usage.get("output_tokens", 0)},
            )
        ]

    def feed(self, event: str, data: dict) -> list[dict]:
        if event == "content_block_start":
            return self._content_block_start(data)
        if event == "content_block_delta":
            return self._content_block_delta(data)
        if event == "message_delta":
            return self._message_delta(data)
        return []

    def finish(self) -> list[dict]:
        if self._finished:
            return []
        self._finished = True
        return []
