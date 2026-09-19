"""Extrai o texto que entrou e o texto que saiu, para a tela de auditoria.

Tres decisoes moram aqui, e as tres existem porque conversa e o dado mais
sensivel que atravessa este proxy:

- **Desligado por padrao.** `SHUNT_STORE_BODIES=1` liga. Um proxy que comeca
  gravando o que o operador digita seria uma surpresa desagradavel.
- **Teto por campo.** Um prompt de agente carrega o repositorio inteiro; sem
  corte, um dia de uso enche o disco. O tamanho ORIGINAL e guardado, para que a
  tela diga quanto ficou de fora em vez de fingir que a conversa acabou ali.
- **Credencial some.** O corpo nao costuma carregar chave, mas um agente que
  cola um `.env` num prompt carrega. O que parece chave vira `[redigido]`.

O que se guarda e TEXTO, e nao o JSON cru: quem abre esta tela quer ler a
conversa, e nao decifrar blocos de conteudo.
"""

import json
import os
import re

# Teto por campo, em caracteres. 64 KB cobre uma conversa longa inteira e ainda
# deixa a leitura rapida na tela.
DEFAULT_LIMIT = 64_000

# O que parece credencial em texto livre. Deliberadamente conservador: um falso
# positivo esconde uma linha, um falso negativo grava uma chave para sempre.
SECRET_PATTERNS = (
    # Oito caracteres depois do prefixo: uma chave de OpenRouter e
    # `sk-or-v1-...`, e exigir dezesseis deixava essa passar.
    re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9_\-]{8,}", re.IGNORECASE),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}"),
    re.compile(r"\bey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    # Sem ancora de linha: um `.env` colado no meio de um paragrafo conta.
    re.compile(r"\b[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)\s*=\s*\S+"),
)
REDACTED = "[redigido]"


def enabled() -> bool:
    """`SHUNT_STORE_BODIES` ligado? Vazio, ausente ou "0" contam como nao."""
    return os.environ.get("SHUNT_STORE_BODIES", "").strip().lower() in {"1", "true", "sim", "on"}


def limit() -> int:
    raw = os.environ.get("SHUNT_BODY_LIMIT", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_LIMIT
    except ValueError:
        return DEFAULT_LIMIT
    return max(200, value)


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def _block_text(block: object) -> str:
    """Um bloco de conteudo virando a linha que um humano le.

    Uma ferramenta chamada e um resultado de ferramenta sao parte da conversa
    tanto quanto o texto: sem eles, a leitura de um turno agentico fica cheia
    de buracos.
    """
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    kind = block.get("type")
    if kind == "text":
        return str(block.get("text", ""))
    if kind == "thinking":
        return f"[pensando] {block.get('thinking', '')}"
    if kind == "tool_use":
        argumentos = json.dumps(block.get("input", {}), ensure_ascii=False)
        return f"[ferramenta {block.get('name', '?')}] {argumentos}"
    if kind == "tool_result":
        return f"[resultado de ferramenta] {_content_text(block.get('content'))}"
    if kind == "image":
        return "[imagem]"
    return f"[{kind or 'bloco'}]"


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(filter(None, (_block_text(block) for block in content)))
    return ""


def prompt_text(body: dict) -> str:
    """A conversa que subiu, nos dois dialetos, como texto corrido."""
    lines = []
    system = body.get("system")
    if system:
        lines.append(f"system: {_content_text(system)}")
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "?")
        text = _content_text(message.get("content"))
        if message.get("tool_calls"):
            calls = [
                f"[ferramenta {(call.get('function') or {}).get('name', '?')}]"
                f" {(call.get('function') or {}).get('arguments', '')}"
                for call in message["tool_calls"]
                if isinstance(call, dict)
            ]
            text = "\n".join(filter(None, [text, *calls]))
        lines.append(f"{role}: {text}")
    # `/v1/completions` nao tem mensagens, tem prompt cru.
    if not lines and body.get("prompt"):
        lines.append(str(body["prompt"]))
    # `/v1/embeddings` tambem e texto que alguem quis ver de novo.
    if not lines and body.get("input"):
        lines.append(_content_text(body["input"]))
    return "\n\n".join(lines)


def answer_text(body: dict) -> str:
    """A resposta ja traduzida, nos dois dialetos, como texto corrido."""
    if body.get("content"):
        return _content_text(body["content"])
    lines = []
    for choice in body.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        if message.get("reasoning_content"):
            lines.append(f"[pensando] {message['reasoning_content']}")
        if message.get("content"):
            lines.append(str(message["content"]))
        for call in message.get("tool_calls") or []:
            function = (call or {}).get("function") or {}
            lines.append(f"[ferramenta {function.get('name', '?')}] {function.get('arguments', '')}")
        if choice.get("text"):
            lines.append(str(choice["text"]))
    return "\n".join(lines)


def capture(prompt: str, answer: str) -> dict | None:
    """O par pronto para gravar, ou None quando nao ha nada que valha a pena.

    Devolve o tamanho ORIGINAL de cada lado junto do texto cortado: sem isso a
    tela nao teria como dizer que a conversa continua.
    """
    if not enabled():
        return None
    ceiling = limit()
    prompt, answer = redact(prompt or ""), redact(answer or "")
    if not prompt and not answer:
        return None
    return {
        "prompt": prompt[:ceiling],
        "answer": answer[:ceiling],
        "prompt_bytes": len(prompt),
        "answer_bytes": len(answer),
        "truncated": len(prompt) > ceiling or len(answer) > ceiling,
    }
