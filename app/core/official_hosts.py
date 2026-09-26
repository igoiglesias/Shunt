"""Hosts oficiais embutidos em codigo (spec R1, docs/superpowers/specs/
2026-09-23-harness-auth-responses-design.md): o destino de um candidato
transparente quando o provedor derivado do nome do modelo nao esta declarado
no catalogo. Nunca vem de DB nem de seed -- e codigo, como manda a spec."""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

# O mesmo `Literal` do `ProviderConfig.protocol` (`app/config/settings.py`):
# o `OfficialHost` vira um `ProviderConfig` em `_provider_config`, e os dois
# campos tem de casar sem cast.
Protocol = Literal["openai", "anthropic"]


@dataclass(frozen=True)
class OfficialHost:
    base_url: str
    protocol: Protocol


OFFICIAL_HOSTS: Mapping[str, OfficialHost] = {
    "anthropic": OfficialHost("https://api.anthropic.com", "anthropic"),
    "openai": OfficialHost("https://api.openai.com/v1", "openai"),
}