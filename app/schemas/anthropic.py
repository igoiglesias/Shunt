"""Schemas do dialeto Anthropic: o que entra nas rotas e o que o proxy emite.

Duas regras governam este modulo e o seu par em `openai.py`:

- **Entrada valida sem derrubar o desconhecido.** `extra="allow"` mantem no
  objeto qualquer campo que este proxy ainda nao conhece, e o repasse usa
  `model_dump(exclude_unset=True)`, que devolve exatamente o que o cliente
  mandou -- nem um default a mais. Um provedor ganha campo novo antes de nos
  sabermos dele; um schema estrito o faria sumir em silencio no caminho.
- **Saida so modela o que o proxy constroi.** Envelope de erro, catalogo de
  modelos e estimativa de tokens sao autoria nossa, entao tem forma fixa e
  vale prende-la aqui. Corpo que veio do provedor NAO passa por schema de
  saida: filtrar a resposta dele e o mesmo defeito, do outro lado.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]]


class AnthropicRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

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


class CountTokensRequest(BaseModel):
    """`/v1/messages/count_tokens` recebe a mesma requisicao, sem `max_tokens`."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[AnthropicMessage]
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None


class CountTokensResponse(BaseModel):
    input_tokens: int


class AnthropicError(BaseModel):
    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    type: str = "error"
    error: AnthropicError

    @classmethod
    def of(cls, error_type: str, message: str) -> AnthropicErrorResponse:
        return cls(error=AnthropicError(type=error_type, message=message))


class AnthropicModel(BaseModel):
    type: str = "model"
    id: str
    display_name: str
    created_at: str


class AnthropicModelList(BaseModel):
    data: list[AnthropicModel]
    has_more: bool = False
    first_id: str | None = None

    @classmethod
    def of(cls, entries: list[dict[str, Any]]) -> AnthropicModelList:
        """`first_id` vem do catalogo, nao do chamador: uma instalacao sem
        modelo nenhum responde `null`, que e o que os parsers leem como
        "nao ha nada aqui" -- e `entries[0]` estouraria."""
        models = [
            AnthropicModel(**{k: e[k] for k in ("id", "display_name", "created_at")})
            for e in entries
        ]
        return cls(data=models, first_id=models[0].id if models else None)
