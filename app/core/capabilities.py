import json
from dataclasses import dataclass

from app.config.settings import Settings
from app.core.resolver import Candidate


@dataclass(frozen=True)
class Requirements:
    tools: bool
    vision: bool
    streaming: bool
    input_tokens: int
    # A resposta ocupa a MESMA janela do prompt. Medido no trafego real: um
    # candidato declarado com 131.072 de contexto recusava com 413 pedidos que
    # a conta so-do-prompt dizia caber, porque o `max_tokens` do cliente
    # entrava no orcamento do provedor e nao no nosso.
    output_tokens: int = 0


def estimate_tokens(payload: dict) -> int:
    """Estimativa grosseira por caracteres. Serve para descartar candidato que
    claramente não cabe, não para cobrança.

    Conta `messages` e `tools`: no pivô OpenAI o `system` já está dentro de
    `messages`, mas o array `tools` (definições de função com schema JSON
    completo) fica de fora se não for somado, e num harness de código ele
    costuma ser boa parte do total real de tokens."""
    text = json.dumps(
        {"messages": payload.get("messages", []), "tools": payload.get("tools", [])},
        ensure_ascii=False,
    )
    return len(text) // 4


# OpenAI puts an image in a content part typed `image_url`; Anthropic types the
# same thing `image` and carries it under `source`. The dispatcher probes the
# requirements with the first candidate's payload, so when that candidate speaks
# the client's own protocol the probe is the untranslated body -- an Anthropic
# request with a screenshot in it. Recognising only the OpenAI spelling made the
# vision requirement vanish there and let a blind model take the request.
IMAGE_PART_TYPES = frozenset({"image_url", "image"})


def _has_image(payload: dict) -> bool:
    for message in payload.get("messages", []):
        content = message.get("content")
        if isinstance(content, list) and any(
            part.get("type") in IMAGE_PART_TYPES for part in content if isinstance(part, dict)
        ):
            return True
    return False


def _requested_output(payload: dict) -> int:
    """O `max_tokens` do cliente, ou zero.

    Ausente ou nao numerico conta como zero de proposito: inventar um teto
    derrubaria candidato que cabe por causa de um palpite nosso.
    """
    raw = payload.get("max_tokens")
    return int(raw) if isinstance(raw, int | float) and not isinstance(raw, bool) else 0


def requirements_of(payload: dict) -> Requirements:
    return Requirements(
        tools=bool(payload.get("tools")),
        vision=_has_image(payload),
        streaming=bool(payload.get("stream")),
        input_tokens=estimate_tokens(payload),
        output_tokens=_requested_output(payload),
    )


def filter_chain(
    chain: list[Candidate], req: Requirements, settings: Settings
) -> tuple[list[Candidate], list[tuple[str, str]]]:
    kept: list[Candidate] = []
    dropped: list[tuple[str, str]] = []
    for candidate in chain:
        if candidate.transparent or candidate.alias is None:
            kept.append(candidate)
            continue
        model = settings.models[candidate.alias]
        if req.tools and not model.supports.tools:
            dropped.append((candidate.alias, "no tool support"))
        elif req.vision and not model.supports.vision:
            dropped.append((candidate.alias, "no vision support"))
        elif req.streaming and not model.supports.streaming:
            dropped.append((candidate.alias, "no streaming support"))
        elif req.input_tokens + req.output_tokens > model.context_window:
            # Com os dois numeros no rastro, o operador ve por quanto o pedido
            # passou do teto -- e decide se mexe no catalogo ou no cliente.
            needed = req.input_tokens + req.output_tokens
            dropped.append(
                (
                    candidate.alias,
                    f"context window too small ({needed} > {model.context_window})",
                )
            )
        else:
            kept.append(candidate)
    return kept, dropped
