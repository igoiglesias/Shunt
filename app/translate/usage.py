"""O bloco `usage`, com os sinais de cache que os dois dialetos carregam.

Existe separado dos dois tradutores porque a regra e a MESMA nos dois sentidos, e
duplicada ela se perderia num deles -- que foi o que aconteceu: o cache do
provedor era descartado na traducao, entao nem o cliente nem o painel sabiam se
o prefixo repetido estava sendo reaproveitado.

A regra que manda aqui: **ausente e zero sao coisas diferentes**. Medido nos tres
provedores desta instalacao, com duas requisicoes identicas:

  llama.cpp local     2a chamada: `cached_tokens` 2814 de 2818 -- cacheia sozinho
  groq gpt-oss-120b   nenhum `prompt_tokens_details` -- nao diz nada
  openrouter/free     `cached_tokens` 0 nas duas -- diz que nao cacheou

Zero e medicao; ausente e silencio. Preencher o silencio com zero faria a tela
afirmar "0% de cache" sobre um provedor que so nao informa, e essa e exatamente
a conclusao errada que se quer evitar.
"""

# Anthropic separa leitura e escrita de cache em dois campos de topo; o dialeto
# OpenAI junta os dois dentro de `prompt_tokens_details`.
READ = ("cache_read_input_tokens", "cached_tokens")
WRITE = ("cache_creation_input_tokens", "cache_write_tokens")


def _int(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def openai_usage_to_anthropic(usage) -> dict:
    """`usage` do provedor OpenAI no formato que o cliente Anthropic le."""
    usage = usage if isinstance(usage, dict) else {}
    out = {
        "input_tokens": _int(usage.get("prompt_tokens")),
        "output_tokens": _int(usage.get("completion_tokens")),
    }
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return out
    for anthropic_name, openai_name in (READ, WRITE):
        value = details.get(openai_name)
        if isinstance(value, int) and not isinstance(value, bool):
            out[anthropic_name] = value
    return out


def anthropic_usage_to_openai(usage) -> dict:
    """`usage` do provedor Anthropic no formato que o cliente OpenAI le."""
    usage = usage if isinstance(usage, dict) else {}
    out: dict[str, object] = {
        "prompt_tokens": _int(usage.get("input_tokens")),
        "completion_tokens": _int(usage.get("output_tokens")),
    }
    details: dict[str, int] = {}
    for anthropic_name, openai_name in (READ, WRITE):
        value = usage.get(anthropic_name)
        if isinstance(value, int) and not isinstance(value, bool):
            details[openai_name] = value
    if details:
        out["prompt_tokens_details"] = details
    return out


def cache_of(usage) -> tuple[int | None, int | None]:
    """(lido do cache, escrito no cache), ou None quando o provedor nao disse.

    Aceita os dois dialetos porque o gravador ve o `usage` ja traduzido para o
    dialeto de QUEM PEDIU, e as duas formas chegam ali.
    """
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}

    def pick(anthropic_name: str, openai_name: str) -> int | None:
        for source, key in ((usage, anthropic_name), (details, openai_name)):
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    return pick(*READ), pick(*WRITE)
