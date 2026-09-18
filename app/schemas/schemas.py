from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]

class AnthropicMessage(BaseModel):
    role: str  # "user" ou "assistant"
    content: Union[str, List[Dict[str, Any]]]

class AnthropicRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    max_tokens: int = 1024
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False