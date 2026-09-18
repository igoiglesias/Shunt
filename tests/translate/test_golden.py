import json
from pathlib import Path

from app.translate.to_anthropic import openai_response_to_anthropic
from app.translate.to_openai import anthropic_request_to_openai

GOLDEN = Path(__file__).parent.parent / "golden"


def load(name):
    return json.loads((GOLDEN / name).read_text())


def test_request_golden_pair_matches():
    got = anthropic_request_to_openai(load("anthropic_request_tool_loop.json"), "vendor/free", 8192)
    assert got == load("openai_request_tool_loop.json")


def test_response_golden_pair_matches():
    got = openai_response_to_anthropic(load("openai_response_tool_call.json"), "claude-opus-4-5")
    assert got == load("anthropic_response_tool_call.json")


def test_second_turn_does_not_degrade_the_history():
    """O histórico traduzido reentra no proxy no turno seguinte; os ids de tool
    precisam sobreviver à ida e à volta."""
    first = anthropic_request_to_openai(
        load("anthropic_request_tool_loop.json"), "vendor/free", 8192
    )
    ids = [c["id"] for m in first["messages"] for c in m.get("tool_calls", [])]
    results = [m["tool_call_id"] for m in first["messages"] if m["role"] == "tool"]
    assert set(results) <= set(ids), "tool_result perdeu a ligação com o tool_use"
