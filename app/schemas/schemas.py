from typing import Any

from pydantic import BaseModel


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    stream: bool | None = False

class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int | None = 16
    temperature: float | None = 1.0
    stream: bool | None = False

class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]

class AnthropicMessage(BaseModel):
    role: str  # "user" ou "assistant"
    content: str | list[dict[str, Any]]

class AnthropicRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    system: str | list[dict[str, Any]] | None = None
    max_tokens: int = 1024
    temperature: float | None = 1.0
    stream: bool | None = False