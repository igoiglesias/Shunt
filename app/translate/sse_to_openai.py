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
- An `input_json_delta` addressed at a block index that was never opened
  (a dropped `content_block_start`, or a reordered stream) is dropped
  rather than attached to an unrelated tool call -- corrupting one call's
  arguments with another's fragments is worse than losing a fragment. The
  SSE decoder already makes this call for a partial event.
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

import time
from typing import Any

# Mesmo teto do outro tradutor: contar pedacos, nao bytes.
from app.config.config import ANSWER_PIECES
from app.stats import bodies
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
        # Captured once, at construction, and reused on every chunk: a real
        # provider stamps one `created` per completion, not one per chunk,
        # and a stable value keeps the stream testable.
        self._created = int(time.time())
        self._tool_index_of_block: dict[int, int] = {}
        self._next_tool_index = 0
        self._finished = False
        self._usage = {"input_tokens": 0, "output_tokens": 0}
        # So para o painel: nao muda um byte do que sai para o cliente.
        self._tools_called: list[str] = []
        self._thinking_blocks = 0
        # O texto da resposta so existe enquanto passa: em streaming nao ha
        # corpo final para ler depois. Acumula aqui, com teto, e so quando a
        # gravacao de conversa esta ligada.
        self._answer: list[str] = []

    def answer_text(self) -> str:
        return "".join(self._answer)

    def _remember(self, piece: str) -> None:
        if piece and bodies.enabled() and len(self._answer) < ANSWER_PIECES:
            self._answer.append(piece)

    def _chunk(self, delta: dict, finish: str | None = None, usage: dict | None = None) -> dict:
        payload: dict[str, Any] = {
            "id": self._id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage is not None:
            payload["usage"] = usage
        return payload

    def tools_called(self) -> list[str]:
        return list(self._tools_called)

    def thinking_blocks(self) -> int:
        return self._thinking_blocks

    def _content_block_start(self, data: dict) -> list[dict]:
        block = data.get("content_block") or {}
        if block.get("type") == "thinking":
            self._thinking_blocks += 1
            self._remember("[pensando] ")
        if block.get("type") != "tool_use":
            return []
        if isinstance(block.get("name"), str):
            self._tools_called.append(block["name"])
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
            texto = delta.get("text", "")
            self._remember(texto)
            return [self._chunk({"content": texto})]
        if delta.get("type") == "thinking_delta":
            self._remember(delta.get("thinking", ""))
        if delta.get("type") == "input_json_delta":
            block_index = data.get("index", 0)
            if block_index not in self._tool_index_of_block:
                # No content_block_start ever opened this block index (a
                # dropped event, or a reordered stream). Attaching the
                # fragment to whatever tool call happens to own index 0
                # would corrupt an unrelated call's arguments -- worse than
                # losing the fragment. Same call the SSE decoder already
                # makes: drop a partial event rather than emit a truncated
                # one.
                return []
            index = self._tool_index_of_block[block_index]
            return [
                self._chunk(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "function": {"arguments": delta.get("partial_json", "")},
                            }
                        ]
                    }
                )
            ]
        return []

    def usage(self) -> dict[str, int]:
        """Os tokens que este stream consumiu, para a linha de log. Mesma razao
        da contraparte em `sse_to_anthropic.py`."""
        return dict(self._usage)

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

    def _absorb_usage(self, data: dict) -> None:
        """A contagem chega espalhada: `input_tokens` dentro de
        `message.usage` no `message_start`, `output_tokens` no `usage` de topo
        do `message_delta`. Ler os dois lugares em toda entrada e mais barato
        do que tratar cada evento so por causa disso -- e o `message_start`
        nao tem tratamento nenhum aqui, porque nao produz chunk.
        """
        for source in (data.get("usage"), (data.get("message") or {}).get("usage")):
            if not isinstance(source, dict):
                continue
            for field_name in ("input_tokens", "output_tokens"):
                value = source.get(field_name)
                if isinstance(value, int):
                    self._usage[field_name] = value

    def feed(self, event: str, data: dict) -> list[dict]:
        self._absorb_usage(data)
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
