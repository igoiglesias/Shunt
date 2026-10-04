"""Effort canonico e a normalizacao usada so na traducao entre dialetos.

O Shunt conhece quatro niveis: low, medium, high, xhigh. Cada lado tem
vocabulario proprio (a OpenAI aceita `none`/`minimal`, a Anthropic aceita
`max`); na traducao o valor vira um dos quatro, ou e descartado e o campo de
destino nao e escrito. No mesmo dialeto nada passa por aqui: o corpo do
harness segue como veio, inclusive com valor que so o upstream entende.
"""

EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh")

# Valores de um dialeto que nao existem no outro, levados ao canonico mais
# proximo. `none` vira `low` e nao "sem effort": descartar mudaria o
# comportamento do upstream mais do que o nivel mais baixo muda.
_ALIASES = {"none": "low", "minimal": "low", "max": "xhigh"}


def normalize_effort(value: object) -> str | None:
    """Devolve o effort canonico, ou None quando o valor deve ser descartado.

    So string conta; caixa e espacos nas pontas nao importam. Numero, dict,
    lista, bool, None ou string desconhecida viram None.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in EFFORTS:
        return text
    return _ALIASES.get(text)
