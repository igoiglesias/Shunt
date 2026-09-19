"""Leitura de ferramentas e raciocinio nas duas formas, incluindo as torcidas.

Provedor real manda lista com item que nao e objeto, ferramenta sem nome e
`choices` sem `message`. Nenhum desses casos pode levantar dentro do log: a
requisicao ja foi respondida quando esta leitura acontece.
"""

import pytest

from app.core.dispatcher import ShuntRequest, _tools_called, tools_offered
from app.core.observability import RequestLog, as_event, recorder


def request_with(tools):
    return ShuntRequest("anthropic", {"tools": tools}, {}, endpoint="messages")


@pytest.mark.parametrize(
    ("tools", "expected"),
    [
        ([{"name": "read"}], ["read"]),
        ([{"function": {"name": "read"}}], ["read"]),
        (["isso nao e objeto"], []),
        ([{"name": 7}], []),
        ([], []),
        (None, []),
    ],
)
def test_the_offered_tools_are_read_in_both_dialects_and_survive_junk(tools, expected):
    assert tools_offered(request_with(tools)) == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"content": [{"type": "tool_use", "name": "read"}]}, (["read"], 0)),
        ({"content": [{"type": "thinking", "thinking": "..."}]}, ([], 1)),
        ({"content": ["isso nao e bloco"]}, ([], 0)),
        ({"content": [{"type": "tool_use"}]}, ([], 0)),
        (
            {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "pensou",
                            "tool_calls": [{"function": {"name": "write"}}],
                        }
                    }
                ]
            },
            (["write"], 1),
        ),
        ({"choices": [{"finish_reason": "stop"}]}, ([], 0)),
        ({"choices": ["isso nao e escolha"]}, ([], 0)),
        ({"choices": [{"message": {"tool_calls": ["isso nao e chamada"]}}]}, ([], 0)),
        ({}, ([], 0)),
    ],
)
def test_the_called_tools_are_read_in_both_dialects_and_survive_junk(body, expected):
    assert _tools_called(body) == expected


def test_the_default_recorder_is_a_no_op_so_importing_costs_nothing():
    """Quem importa o modulo fora de um processo com banco nao paga nada."""
    assert recorder().enabled is False


def test_a_request_that_answered_after_an_attempt_is_marked_as_a_fallback():
    event = as_event(
        RequestLog(
            request_id="r",
            requested_model="m",
            rule="exact",
            matched="m",
            candidate="vendor/cheap",
            attempts=["free: 400 (attempt 1)"],
        )
    )
    assert event["fell_back"] is True


def test_a_request_that_answered_first_try_is_not_a_fallback():
    event = as_event(
        RequestLog(
            request_id="r",
            requested_model="m",
            rule="exact",
            matched="m",
            candidate="vendor/free",
        )
    )
    assert event["fell_back"] is False
