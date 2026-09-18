from app.translate.ids import to_anthropic_id, to_openai_id


def test_id_survives_two_full_turns():
    provider_id = "call_abc123XYZ"
    shown_to_client = to_anthropic_id(provider_id)
    assert shown_to_client.startswith("toolu_")
    assert to_openai_id(shown_to_client) == provider_id
    # segunda volta: o cliente devolve o mesmo id no turno seguinte
    assert to_openai_id(to_anthropic_id(provider_id)) == provider_id


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
