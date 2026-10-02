"""Hosts oficiais embutidos em codigo (spec R1, docs/superpowers/specs/
2026-09-23-harness-auth-responses-design.md): o destino de um candidato
transparente quando o provedor derivado do nome do modelo nao esta declarado
no catalogo. Nunca vem de DB nem de seed -- e codigo, como manda a spec."""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import httpx

# O mesmo `Literal` do `ProviderConfig.protocol` (`app/config/settings.py`):
# o `OfficialHost` vira um `ProviderConfig` em `_provider_config`, e os dois
# campos tem de casar sem cast.
Protocol = Literal["openai", "anthropic"]


@dataclass(frozen=True)
class OfficialHost:
    base_url: str
    protocol: Protocol

    @property
    def origin(self) -> str:
        """O host+esquema da base, sem path/query/fragment (ex.: o `/v1` do
        OpenAI nao entra no origin). O catch-all do repasse (T5) monta a URL
        com o path do cliente sobre este valor, e duplicar o prefixo
        (`/v1/v1/files`) era o defeito medido na secao 2 do plano."""
        return str(httpx.URL(self.base_url).copy_with(path="/", query=None, fragment=None)).rstrip("/")


OFFICIAL_HOSTS: Mapping[str, OfficialHost] = {
    "anthropic": OfficialHost("https://api.anthropic.com", "anthropic"),
    "openai": OfficialHost("https://api.openai.com/v1", "openai"),
}