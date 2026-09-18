"""Estimativa local de tokens de entrada.

Existe por uma razao so: `/v1/messages/count_tokens` e um endpoint Anthropic
sem equivalente na API OpenAI. Quando o candidato resolvido nao fala Anthropic
nao ha a quem encaminhar a pergunta, e responder um erro seria pior do que
responder um numero aproximado -- o cliente usa essa contagem para decidir
compactacao de contexto, nao para cobrar ninguem.

A aproximacao e deliberadamente grosseira (caracteres do JSON serializado
dividido por `CHARS_PER_TOKEN`), porque o tokenizador real varia por modelo e
nenhum deles e o do destino.
"""

import json

CHARS_PER_TOKEN = 4


def estimate_input_tokens(body: dict) -> int:
    """Estimativa local para quando o destino nao e Anthropic e nao existe
    endpoint equivalente. Serve para o cliente decidir compactacao de contexto,
    nao para cobranca."""
    parts = [json.dumps(body.get("messages", []), ensure_ascii=False)]
    for key in ("system", "tools"):
        if body.get(key):
            parts.append(json.dumps(body[key], ensure_ascii=False))
    # `max(1, ...)`: zero leria como "este pedido nao custa nada" em vez de
    # "pequeno demais para medir".
    return max(1, sum(len(p) for p in parts) // CHARS_PER_TOKEN)
