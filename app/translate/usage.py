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
    # Alguns provedores enviam usage numerico nao-inteiro (medido no plano
    # bypass-stream-bugs: `prompt_tokens: 5.0`); truncar para int e a leitura
    # honesta, descartar como silencio faria o painel afirmar "nao informou".
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


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


def tokens_of(usage) -> tuple[int | None, int | None]:
    """Os tokens de entrada e saida de um `usage`, em qualquer dialeto.

    Medido (bug da analise): quem fala Anthropic traz `input_tokens`, quem fala
    OpenAI traz `prompt_tokens`. O relatorio do painel e da analise precisa do
    numero de qualquer um dos dois, e a expressao que escolhe um caiu no outro.
    O ponto unico evita que uma quarta copia apareca desatualizada.

    Zero e medicao, nao silencio: `input_tokens: 0` tem prioridade sobre
    `prompt_tokens`, porque e o provedor dizendo quanto gastou. Usar `or`
    aqui trocaria um zero verdadeiro pelo valor do outro dialeto -- o mesmo
    erro que o modulo inteiro evita no cache.
    """
    usage = usage if isinstance(usage, dict) else {}
    entry, output = usage.get("input_tokens"), usage.get("output_tokens")
    if entry is None:
        entry = usage.get("prompt_tokens")
    if output is None:
        output = usage.get("completion_tokens")
    return _int(entry), _int(output)


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
