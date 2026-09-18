from app.schemas.anthropic import AnthropicRequest


def test_request_keeps_tools_and_tool_choice():
    req = AnthropicRequest(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": "oi"}],
        tools=[{"name": "read", "description": "lê",
                "input_schema": {"type": "object", "properties": {}}}],
        tool_choice={"type": "any", "disable_parallel_tool_use": True},
        stop_sequences=["FIM"],
    )
    assert req.tools[0]["name"] == "read"
    assert req.tool_choice["type"] == "any"
    assert req.stop_sequences == ["FIM"]


def test_content_blocks_survive_validation():
    req = AnthropicRequest(
        model="m", max_tokens=16,
        messages=[{"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a"}},
        ]}],
    )
    assert req.messages[0].content[0]["type"] == "tool_use"
