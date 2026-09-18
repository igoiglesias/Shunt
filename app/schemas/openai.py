from typing import Any

from pydantic import BaseModel


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    stop: str | list[str] | None = None
    stream_options: dict[str, Any] | None = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int = 16
    temperature: float | None = None
    stream: bool = False


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
