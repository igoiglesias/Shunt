from app.translate.sse_parse import SSEDecoder


def test_single_event_in_one_chunk():
    events = SSEDecoder().feed(b'data: {"a": 1}\n\n')
    assert [(e.event, e.data) for e in events] == [(None, '{"a": 1}')]


def test_event_split_in_the_middle_of_the_data_prefix():
    decoder = SSEDecoder()
    assert decoder.feed(b'data') == []
    assert decoder.feed(b': {"a":') == []
    events = decoder.feed(b' 1}\n\n')
    assert events[0].data == '{"a": 1}'


def test_named_event_is_kept():
    events = SSEDecoder().feed(b'event: message_stop\ndata: {}\n\n')
    assert events[0].event == "message_stop"


def test_multiline_data_is_joined_with_newline():
    events = SSEDecoder().feed(b'data: linha1\ndata: linha2\n\n')
    assert events[0].data == "linha1\nlinha2"


def test_comment_lines_are_ignored():
    events = SSEDecoder().feed(b': OPENROUTER PROCESSING\n\ndata: {"a": 1}\n\n')
    assert len(events) == 1


def test_two_events_in_one_chunk_and_a_third_split_across_three():
    decoder = SSEDecoder()
    assert len(decoder.feed(b'data: 1\n\ndata: 2\n\ndata: 3')) == 2
    assert decoder.feed(b'\n') == []
    assert [e.data for e in decoder.feed(b'\n')] == ["3"]


def test_crlf_line_endings():
    events = SSEDecoder().feed(b'data: {"a": 1}\r\n\r\n')
    assert events[0].data == '{"a": 1}'


def test_multibyte_character_split_between_two_chunks():
    """Um emoji ocupa quatro bytes; o corte do TCP não respeita caractere."""
    payload = 'data: {"t": "✅ pronto"}\n\n'.encode()
    cut = payload.index(b"\xe2") + 2  # no meio do caractere
    decoder = SSEDecoder()
    assert decoder.feed(payload[:cut]) == []
    events = decoder.feed(payload[cut:])
    assert events[0].data == '{"t": "✅ pronto"}'


# --- Additional coverage required beyond the brief ---


def test_data_line_with_no_space_after_colon():
    events = SSEDecoder().feed(b'data:{"a": 1}\n\n')
    assert events[0].data == '{"a": 1}'


def test_id_and_retry_fields_are_ignored_without_breaking():
    events = SSEDecoder().feed(b'id: 42\nretry: 5000\ndata: hello\n\n')
    assert [(e.event, e.data) for e in events] == [(None, "hello")]


def test_lone_newline_inside_feed_is_not_an_event_boundary():
    decoder = SSEDecoder()
    # A single \n after a data line just separates fields, not events.
    assert decoder.feed(b'data: hello\n') == []
    events = decoder.feed(b'\n')
    assert events[0].data == "hello"


def test_event_with_empty_data():
    events = SSEDecoder().feed(b'event: ping\ndata:\n\n')
    assert events[0].event == "ping"
    assert events[0].data == ""


def test_chunk_boundary_falls_exactly_between_the_two_newlines():
    decoder = SSEDecoder()
    assert decoder.feed(b'data: hello\n') == []
    events = decoder.feed(b'\n')
    assert [e.data for e in events] == ["hello"]


def test_feed_accepts_str_directly():
    events = SSEDecoder().feed('data: {"a": 1}\n\n')
    assert events[0].data == '{"a": 1}'


def test_flush_drops_line_truncated_mid_transmission():
    """No trailing '\\n' at all means the last line's own terminator never
    arrived -- indistinguishable from a value cut mid-write. Emitting it
    would hand the translator downstream a value that might be missing its
    tail; dropping it is the safe choice."""
    decoder = SSEDecoder()
    decoder.feed(b'data: incomplete, no trailing newline')
    assert decoder.flush() == []


def test_flush_emits_final_event_whose_last_field_line_is_terminated():
    """The realistic 'stream ends without a trailing blank line' case: the
    provider closes the connection right after the data line's own '\\n',
    before sending the blank-line separator. The field line itself is
    complete, so this is safe to emit."""
    decoder = SSEDecoder()
    decoder.feed(b'data: last event\n')
    events = decoder.flush()
    assert [e.data for e in events] == ["last event"]


def test_flush_after_complete_event_returns_nothing_new():
    decoder = SSEDecoder()
    decoder.feed(b'data: done\n\n')
    assert decoder.flush() == []


def test_flush_on_empty_buffer_returns_nothing():
    assert SSEDecoder().flush() == []


def test_line_with_no_colon_at_all_is_ignored_without_breaking():
    events = SSEDecoder().feed(b'garbage_no_colon\ndata: hello\n\n')
    assert [(e.event, e.data) for e in events] == [(None, "hello")]


def test_bare_field_name_with_no_colon_is_not_mistaken_for_a_field():
    """A line that is exactly 'data' with no colon at all is malformed SSE,
    not a data field with an empty value -- it must not turn an otherwise
    fieldless block into a phantom event."""
    events = SSEDecoder().feed(b'data\n\n')
    assert events == []
