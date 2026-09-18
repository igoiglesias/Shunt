"""Decode Server-Sent Events independently of TCP chunk boundaries.

The boundary of an SSE event has nothing to do with the boundary of a TCP
chunk. A single ``data:`` line can arrive split across three reads, two
complete events can arrive in one read, and a multi-byte UTF-8 character can
be cut in half between reads. This module buffers raw bytes/str across calls
to `feed` and only emits `SSEEvent`s once a full event ("\n\n"-terminated
block) has accumulated.

`SSEDecoder` and `SSEEvent` are consumed by the streaming dispatcher (a later
task): the class name, `SSEEvent` shape, and the `feed`/`flush` contract are
fixed interfaces other modules depend on.
"""

import codecs
from dataclasses import dataclass


@dataclass(frozen=True)
class SSEEvent:
    event: str | None
    data: str


class SSEDecoder:
    """Stateful incremental SSE decoder.

    Feed it bytes (or str, for tests) as they arrive from the network, in
    any chunking; it returns the complete events each call produced.
    """

    def __init__(self) -> None:
        self._buffer = ""
        # Incremental UTF-8 decoder: a multi-byte character (e.g. an emoji)
        # can arrive split across two TCP reads. `bytes.decode()` would
        # raise UnicodeDecodeError on the first half; this decoder holds
        # the partial sequence until the rest arrives.
        self._decoder = codecs.getincrementaldecoder("utf-8")()

    def feed(self, chunk: bytes | str) -> list[SSEEvent]:
        """Feed newly-arrived bytes (or str) and return whichever complete
        events they completed. Bytes are the production path; str exists so
        tests can be written without manually encoding every fixture."""
        text = self._decoder.decode(chunk) if isinstance(chunk, bytes) else chunk
        self._buffer += text.replace("\r\n", "\n").replace("\r", "\n")
        events: list[SSEEvent] = []
        while "\n\n" in self._buffer:
            block, self._buffer = self._buffer.split("\n\n", 1)
            parsed = self._parse(block)
            if parsed is not None:
                events.append(parsed)
        return events

    def flush(self) -> list[SSEEvent]:
        """Called once the underlying stream has ended, for the case where
        the provider closes the connection without sending a final blank
        line.

        The trailing buffer is emitted as one last event ONLY if its last
        field line is itself newline-terminated -- i.e. everything in the
        buffer is a complete line, and only the blank-line *separator*
        never arrived. That is the common, safe case this method exists
        for.

        If the last line has no trailing newline at all, its terminator
        never arrived either, which is indistinguishable from the value
        having been cut mid-write by the network. Emitting it would risk
        handing the translator downstream a truncated value it would treat
        as valid content -- worse than silently dropping it. So that case
        is dropped, and the whole partial buffer is discarded rather than
        reconstructed from only the complete lines, keeping the rule simple
        and the behaviour predictable.
        """
        rest, self._buffer = self._buffer, ""
        if not rest or not rest.endswith("\n"):
            return []
        parsed = self._parse(rest[:-1])
        return [parsed] if parsed is not None else []

    @staticmethod
    def _parse(block: str) -> SSEEvent | None:
        name: str | None = None
        data: list[str] = []
        has_data_field = False
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            field, sep, value = line.partition(":")
            if not sep:
                continue
            value = value.removeprefix(" ")
            if field == "event":
                name = value
            elif field == "data":
                has_data_field = True
                data.append(value)
            # Other fields (id, retry, ...) are recognized-and-ignored: the
            # decoder does not surface them, but must not treat them as
            # unknown/malformed input either.
        if not has_data_field and name is None:
            return None
        return SSEEvent(event=name, data="\n".join(data))
