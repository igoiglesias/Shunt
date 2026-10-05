"""Normalizacao do effort: so na traducao entre dialetos (spec R5)."""

import pytest

from app.translate.effort import EFFORTS, normalize_effort


def test_os_quatro_valores_canonicos_sao_estes():
    assert EFFORTS == ("low", "medium", "high", "xhigh")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
        ("HIGH", "high"),
        ("  Medium  ", "medium"),
        ("\txhigh\n", "xhigh"),
        ("none", "low"),
        ("NONE", "low"),
        ("minimal", "low"),
        (" Minimal ", "low"),
        ("max", "xhigh"),
        ("MAX", "xhigh"),
    ],
)
def test_string_conhecida_vira_o_effort_canonico(raw, expected):
    assert normalize_effort(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "extreme",
        "hi gh",
        "x-high",
        "maximum",
        None,
        3,
        2.5,
        True,
        {"effort": "high"},
        ["high"],
    ],
)
def test_qualquer_outra_coisa_e_descartada(raw):
    assert normalize_effort(raw) is None
