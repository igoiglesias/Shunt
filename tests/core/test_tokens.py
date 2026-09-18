from app.core.tokens import estimate_input_tokens


def test_longer_conversation_estimates_more_tokens():
    curta = {"messages": [{"role": "user", "content": "oi"}]}
    longa = {"messages": [{"role": "user", "content": "palavra " * 500}]}
    assert estimate_input_tokens(longa) > estimate_input_tokens(curta) > 0


def test_system_and_tools_count_towards_the_estimate():
    base = {"messages": [{"role": "user", "content": "oi"}]}
    assert estimate_input_tokens({**base, "system": "instrução longa " * 50}) > estimate_input_tokens(
        base
    )
    assert estimate_input_tokens({**base, "tools": [{"name": "x" * 500}]}) > estimate_input_tokens(
        base
    )


def test_an_empty_body_still_estimates_one_token():
    """`max(1, ...)`: a caller that asks about nothing gets 1, never 0. A zero
    would read as "this request costs nothing" instead of "too small to
    measure"."""
    assert estimate_input_tokens({}) == 1


def test_the_estimate_is_four_characters_per_token():
    """Pins the divisor, not just the ordering: `json.dumps` of this one
    message is 133 characters, and 133 // 4 == 33. A different
    `CHARS_PER_TOKEN` moves this number."""
    assert estimate_input_tokens({"messages": [{"role": "user", "content": "x" * 100}]}) == 33


def test_a_falsy_system_or_tools_field_adds_nothing():
    """The `if body.get(key)` guard: an empty string or an empty list must not
    contribute the two characters of its own JSON rendering."""
    base = {"messages": [{"role": "user", "content": "x" * 100}]}
    assert estimate_input_tokens({**base, "system": "", "tools": []}) == estimate_input_tokens(base)
