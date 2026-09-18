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


def requirements_of(payload: dict) -> Requirements:
    return Requirements(
        tools=bool(payload.get("tools")),
        vision=_has_image(payload),
        streaming=bool(payload.get("stream")),
        input_tokens=estimate_tokens(payload),
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
        elif req.input_tokens > model.context_window:
            dropped.append((candidate.alias, "context window too small"))
        else:
            kept.append(candidate)
    return kept, dropped
