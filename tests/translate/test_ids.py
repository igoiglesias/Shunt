from app.translate.ids import to_anthropic_id, to_openai_id


def test_id_survives_two_full_turns():
    # Turn one: the provider emits an id, the proxy hands the client an
    # Anthropic-shaped id derived from it. Turn two: the client sends that
    # *exact same* id back (e.g. inside a tool_result) -- the proxy must
    # reproduce the original provider id from it, with no memory of turn
    # one. Reusing `shown_to_client` (rather than re-encoding the provider
    # id) is what actually models the client returning the value it was
    # given, instead of merely re-testing that to_anthropic_id is
    # deterministic.
    provider_id = "call_abc123XYZ"
    shown_to_client = to_anthropic_id(provider_id)
    assert shown_to_client.startswith("toolu_")
    assert to_openai_id(shown_to_client) == provider_id

    # A second, different provider id in the same conversation must round
    # trip independently -- proves the mapping doesn't cross-contaminate
    # between two ids handled in the same turn/session.
    other_provider_id = "call_def456"
    other_shown_to_client = to_anthropic_id(other_provider_id)
    assert other_shown_to_client != shown_to_client
    assert to_openai_id(other_shown_to_client) == other_provider_id
    # Turn two again, using the id the client actually holds for each.
    assert to_openai_id(shown_to_client) == provider_id
    assert to_openai_id(other_shown_to_client) == other_provider_id


def test_native_anthropic_id_is_derived_deterministically():
    native = "toolu_01ABCdefGHIjklMNO"
    first = to_openai_id(native)
    assert first.startswith("call_")
    assert first == to_openai_id(native)


def test_different_ids_do_not_collide():
    assert to_openai_id("toolu_aaa") != to_openai_id("toolu_bbb")


def test_non_base64_payload_falls_back_to_derived_id():
    # Starts with the encoded marker but the payload after it is not valid
    # base64 (contains characters outside the urlsafe-b64 alphabet). This
    # covers the except branch in to_openai_id: a corrupted/foreign id must
    # not raise, it must fall back to the deterministic hash derivation.
    bogus = "toolu_s!!!not-base64!!!"
    result = to_openai_id(bogus)
    assert result.startswith("call_")
    assert result == to_openai_id(bogus)


def test_id_without_anthropic_prefix_is_derived():
    # No "toolu_" prefix at all: removeprefix is a no-op, body does not start
    # with the ENCODED_MARK-prefixed pattern we produced ourselves, so this
    # must go straight to the hash-derived branch rather than crash or be
    # treated as an encoded id.
    result = to_openai_id("some_other_id")
    assert result.startswith("call_")
    assert result == to_openai_id("some_other_id")


def test_empty_string_is_derived_without_crashing():
    result = to_openai_id("")
    assert result.startswith("call_")
    assert result == to_openai_id("")


def test_round_trip_holds_for_empty_provider_id():
    # Explicit regression guard: to_openai_id(to_anthropic_id(x)) == x must
    # still hold for every provider id, empty string included, after the
    # checksum was added to the encoded tail.
    assert to_openai_id(to_anthropic_id("")) == ""


def test_native_looking_id_that_decodes_cleanly_is_not_misread_as_encoded():
    # Constructed by search (see task-7-report.md for the script): a body
    # that starts with the ENCODED_MARK ("s") and whose remainder IS valid
    # base64url AND decodes to valid UTF-8 -- exactly the accidental
    # collision the weak two-fact discriminator (starts-with-"s" + valid
    # base64/UTF-8) could not rule out. "toolu_s" + "CQlaERA5" decodes to
    # the bytes b'\t\tZ\x11\x109', which .decode() accepts as UTF-8.
    #
    # Before the checksum fix this id was misclassified as Shunt-encoded
    # and returned as that decoded garbage. It is NOT something
    # to_anthropic_id ever produced, so it must fall through to the
    # deterministic call_-derived branch, exactly like any other native
    # Anthropic id.
    native_lookalike = "toolu_sCQlaERA5"
    result = to_openai_id(native_lookalike)
    assert result.startswith("call_")
    assert result == to_openai_id(native_lookalike)
