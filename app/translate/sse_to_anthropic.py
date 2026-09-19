"""Convert an OpenAI-shaped chat-completion stream into Anthropic SSE events.

The two protocols disagree about *when* information arrives, which is why
this is a stateful object rather than a pure function:

- OpenAI sends a tool call's `function.name` only in the first chunk of that
  call; Anthropic requires the name up front, in `content_block_start`,
  before any argument fragment.
- OpenAI has no explicit notion of a content block opening or closing;
  Anthropic demands `content_block_start` and `content_block_stop` around
  every one.
- `usage` can arrive attached to any chunk -- including a final chunk with
  an empty `choices` list -- and Anthropic only reports it once, in the
  closing `message_delta`.

So the translator remembers: whether `message_start` has gone out, which
block index is open and what kind it is, which `tool_calls[].index` maps to
the currently open block, the stop reason, and the last `usage` seen.

Deliberate leniency, matching `app.translate.to_anthropic`:

- Tool call argument fragments are passed through verbatim as
  `input_json_delta` without ever being parsed or repaired here -- a free
  model's JSON may never become valid, and reassembly is the client's job.
- An unrecognized `finish_reason` falls back to the `STOP_REASONS` default
  behaviour (`"end_turn"`), the same map `to_anthropic.py` uses, imported
  rather than duplicated.
- `finish()` is safe to call after any prefix of a stream, including an
  empty one: the dispatcher calls it in a `finally`, so a provider that
  dies mid-stream still produces a well-formed close (message_start emitted
  if it never was, any open block closed, then message_delta/message_stop).
"""

from typing import Any

from app.stats import bodies
from app.translate.ids import to_anthropic_id
from app.translate.to_anthropic import STOP_REASONS, reasoning_of

Event = tuple[str, dict[str, Any]]


# Quantos pedacos de texto a resposta guarda em streaming. Um teto em NUMERO de
# pedacos, e nao em bytes, porque cada um chega solto e contar bytes a cada
# chegada custaria mais do que o proprio teto economiza. O corte final em
# tamanho acontece no `bodies.capture`.
ANSWER_PIECES = 20_000


class OpenAIStreamToAnthropic:
    """Stateful translator: one instance per in-flight stream.

    `finish()` is safe to call more than once: only the first call emits
    events (the dispatcher may call it both at the natural end of the loop
    and again from a `finally` guarding the crash path; the second call is
    a no-op rather than a second message_delta/message_stop).
    """

    def __init__(self, requested_model: str, message_id: str) -> None:
        self._model = requested_model
        self._message_id = message_id
        self._started = False
        self._block_index = -1
        self._open_kind: str | None = None
        self._open_tool_index: int | None = None
        self._stop_reason = "end_turn"
        self._usage = {"input_tokens": 0, "output_tokens": 0}
        self._finished = False
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

    def tools_called(self) -> list[str]:
        return list(self._tools_called)

    def thinking_blocks(self) -> int:
        return self._thinking_blocks

    def usage(self) -> dict[str, int]:
        """Os tokens que este stream consumiu, para a linha de log.

        Quem observa a requisicao nao ve os chunks: o `usage` chega num deles,
        muitas vezes no ultimo, de `choices` vazio, e este objeto e o unico
        que passou por todos.
        """
        return dict(self._usage)

    def _start(self) -> list[Event]:
        if self._started:
            return []
        self._started = True
        return [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": self._message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self._model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": dict(self._usage),
                    },
                },
            )
        ]

    def _close_block(self) -> list[Event]:
        if self._open_kind is None:
            return []
        self._open_kind = None
        self._open_tool_index = None
        return [
            (
                "content_block_stop",
                {"type": "content_block_stop", "index": self._block_index},
            )
        ]

    def _open_text(self) -> list[Event]:
        events = self._close_block()
        self._block_index += 1
        self._open_kind = "text"
        events.append(
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        return events

    def _open_thinking(self) -> list[Event]:
        self._thinking_blocks += 1
        events = self._close_block()
        self._block_index += 1
        self._open_kind = "thinking"
        events.append(
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            )
        )
        return events

    def _open_tool(self, index: int, call: dict) -> list[Event]:
        events = self._close_block()
        self._block_index += 1
        self._open_kind = "tool"
        self._open_tool_index = index
        function = call.get("function") or {}
        if isinstance(function.get("name"), str):
            self._tools_called.append(function["name"])
        events.append(
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": to_anthropic_id(call.get("id", "")),
                        "name": function.get("name", ""),
                        "input": {},
                    },
                },
            )
        )
        return events

    def feed(self, chunk: dict) -> list[Event]:
        events: list[Event] = self._start()

        if chunk.get("usage"):
            self._usage = {
                "input_tokens": chunk["usage"].get("prompt_tokens", 0),
                "output_tokens": chunk["usage"].get("completion_tokens", 0),
            }

        choices = chunk.get("choices") or []
        if not choices:
            return events
        choice = choices[0]
        delta = choice.get("delta") or {}

        # O pensamento e tratado antes do texto porque e assim que chega: o
        # modelo raciocina e so depois responde. Se o texto abrisse o bloco
        # primeiro, um chunk que trouxesse os dois campos juntos poria o
        # `thinking_delta` dentro de um bloco de texto.
        reasoning = reasoning_of(delta)
        if reasoning:
            if self._open_kind != "thinking":
                self._remember("[pensando] ")
                events.extend(self._open_thinking())
            self._remember(reasoning)
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._block_index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    },
                )
            )

        text = delta.get("content")
        if text:
            self._remember(text)
            if self._open_kind != "text":
                events.extend(self._open_text())
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._block_index,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            )

        for call in delta.get("tool_calls") or []:
            index = call.get("index", 0)
            if self._open_kind != "tool" or self._open_tool_index != index:
                events.extend(self._open_tool(index, call))
            arguments = (call.get("function") or {}).get("arguments")
            if arguments:
                events.append(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self._block_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": arguments,
                            },
                        },
                    )
                )

        if choice.get("finish_reason"):
            self._stop_reason = STOP_REASONS.get(choice["finish_reason"], "end_turn")

        return events

    def finish(self) -> list[Event]:
        if self._finished:
            return []
        self._finished = True
        events: list[Event] = self._start()
        events.extend(self._close_block())
        events.append(
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": self._stop_reason,
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": self._usage["output_tokens"]},
                },
            )
        )
        events.append(("message_stop", {"type": "message_stop"}))
        return events
