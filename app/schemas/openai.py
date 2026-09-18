"""Schemas do dialeto OpenAI: o que entra nas rotas e o que o proxy emite.

As duas regras estao escritas em `app/schemas/anthropic.py`, e valem igual
aqui: entrada com `extra="allow"` e repasse por `model_dump(exclude_unset=True)`;
saida modelada so para o que o proxy constroi ele mesmo.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

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
    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str | list[str]
    max_tokens: int = 16
    temperature: float | None = None
    stream: bool = False


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[str]


class OpenAIError(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIError

    @classmethod
    def of(cls, error_type: str, message: str) -> OpenAIErrorResponse:
        """`param` e `code` vao presentes e nulos de proposito: a biblioteca
        oficial le os quatro campos, e um envelope sem eles a deixa sem como
        classificar o que aconteceu."""
        return cls(error=OpenAIError(message=message, type=error_type))


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModel]

    @classmethod
    def of(cls, entries: list[dict[str, Any]]) -> OpenAIModelList:
        return cls(
            data=[OpenAIModel(**{k: e[k] for k in ("id", "created", "owned_by")}) for e in entries]
        )
