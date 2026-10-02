"""Regras puras do relay de rotas desconhecidas (o catch-all da T5 as usa).

O relay repassa a requisicao byte a byte ao host oficial: nada aqui toca o
corpo, e as funcoes so decidem para onde ela vai e quais headers/query
sobrevivem a passagem. O token do Shunt nunca viaja: o que casou como token
sai da requisicao antes de subir."""
from collections.abc import Mapping
from urllib.parse import parse_qsl

from app.core.official_hosts import Protocol

# Headers que pertencem ao hop local (transporte entre cliente e proxy) ou sao
# segredo deste proxy: mandar qualquer um ao upstream quebraria o repasse
# byte a byte ou vazaria credencial. `accept-encoding` NAO entra aqui de
# proposito: o corpo volta intocado, entao o gzip do upstream precisa chegar.
RELAY_DROP: frozenset[str] = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "upgrade",
        "proxy-authorization",
        "proxy-connection",
        "x-shunt-token",
        "cookie",
    }
)

# Hopping headers da RESPOSTA, que nao fazem sentido repassados ao cliente.
RESPONSE_DROP: frozenset[str] = frozenset({"transfer-encoding", "connection", "keep-alive"})


def relay_target(headers: Mapping[str, str]) -> Protocol:
    """Que host oficial recebe a rota desconhecida?

    Nomes de header sao comparados sem distinguir maiusculas e minusculas
    (normalizados para `.lower()`): no Starlette/httpx eles ja chegam
    minusculos, mas esta funcao e pura e aceita qualquer dict.

    A ordem e decisao 2 do plano: primeiro os sinais fortes de cada
    protocolo (sem isto um bearer OAuth da Anthropic sem headers
    `anthropic-*` cairia no `authorization` e iria para a OpenAI), e por
    ultimo o dialeto do chamador, onde a falta de sinal e Anthropic.
    """
    # Importe tardio: o catch-all (T5) importa ESTE modulo em `app/routers/v1.py`,
    # entao importar `v1` no topo criaria ciclo de importacao no boot.
    from app.routers.v1 import ANTHROPIC_AGENTS, detect_protocol

    lower = {k.lower(): v for k, v in headers.items()}
    user_agent = lower.get("user-agent", "").lower()
    authorization = lower.get("authorization", "")
    bearer = authorization.removeprefix("Bearer ").strip() if authorization.startswith("Bearer ") else ""
    if any(name.startswith("anthropic-") for name in lower):
        return "anthropic"
    if "x-api-key" in lower:
        return "anthropic"
    if bearer.startswith("sk-ant-"):
        return "anthropic"
    if any(agent in user_agent for agent in ANTHROPIC_AGENTS):
        return "anthropic"
    if any(name.startswith("openai-") for name in lower):
        return "openai"
    if "openai" in user_agent:
        return "openai"
    found = detect_protocol(headers)
    return "openai" if found == "openai" else "anthropic"


def relay_headers(headers: Mapping[str, str], token_headers: frozenset[str]) -> dict[str, str]:
    """Headers a enviar ao upstream.

    Sobe tudo menos `RELAY_DROP` (hopping local e segredos -- o token do
    Shunt e a `cookie` nunca viajam) e menos os nomes em `token_headers`
    (os headers onde o token do Shunt casou, registrados pela T2 na
    entrada). `accept-encoding` passa de proposito, para o gzip do upstream
    chegar byte a byte. A comparacao e sem distinguir maiusculas; os nomes
    de entrada sao preservados como vieram.
    """
    drop = RELAY_DROP | {name.lower() for name in token_headers}
    return {name: value for name, value in headers.items() if name.lower() not in drop}


def relay_params(query: str) -> list[tuple[str, str]]:
    """Query a enviar ao upstream: a mesma que o cliente mandou, sem `token`.

    `token` sai em qualquer caixa (era ali que uma credencial poderia ter
    sido deixada); o restante -- incluindo valores vazios e a ordem --
    sobrevive, para o upstream ver exatamente a query do cliente.
    """
    return [(name, value) for name, value in parse_qsl(query, keep_blank_values=True) if name.lower() != "token"]
