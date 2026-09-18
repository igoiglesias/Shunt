from typing import Any

from pydantic import BaseModel


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class AnthropicRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = 1024
    system: str | list[dict[str, Any]] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
