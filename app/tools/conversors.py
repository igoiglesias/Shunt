from typing import Any

from app.schemas.schemas import AnthropicRequest


def transform_anthropic_to_openai(req: AnthropicRequest) -> dict[str, Any]:
    """Converte o payload de envio da Anthropic para o formato OpenAI."""
    openai_messages = []

    # 1. Extrai o 'system' da raiz e converte na primeira mensagem do array OpenAI
    if req.system:
        if isinstance(req.system, str):
            sys_text = req.system
        else:
            sys_text = "".join(
                b.get("text", "") 
                for b in req.system 
                if isinstance(b, dict) and b.get("type") == "text"
            )
        openai_messages.append({"role": "system", "content": sys_text})

    # 2. Converte o histórico de mensagens
    for msg in req.messages:
        content_val = msg.content
        if isinstance(content_val, list):
            content_val = "\n".join(
                b.get("text", "") 
                for b in content_val 
                if isinstance(b, dict) and b.get("type") == "text"
            )
        openai_messages.append({"role": msg.role, "content": content_val})

    return {
        "model": req.model,
        "messages": openai_messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": req.stream
    }

def transform_openai_to_anthropic(openai_resp: dict[str, Any]) -> dict[str, Any]:
    """Converte a resposta recebida da OpenAI para a estrutura exigida pelo Claude Code."""
    finish_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use"
    }
    
    choice = openai_resp.get("choices", [{}])[0]
    msg_text = choice.get("message", {}).get("content", "")
    raw_finish = choice.get("finish_reason", "stop")
    usage = openai_resp.get("usage", {})

    return {
        "id": openai_resp.get("id", "msg_proxy").replace("chatcmpl", "msg"),
        "type": "message",
        "role": "assistant",
        "model": openai_resp.get("model", ""),
        "content": [
            {
                "type": "text",
                "text": msg_text
            }
        ],
        "stop_reason": finish_map.get(raw_finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0)
        }
    }