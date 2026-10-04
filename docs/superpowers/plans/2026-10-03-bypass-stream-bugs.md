# Plano: bugs do bypass/telemetria encontrados no server vivo (porta 8000)

Contexto: lemos o historico real do `stats.db` (24.772 eventos) do server em
producao local. Tres causas-raiz verificadas por reproducao isolada.

## Global Constraints

- Docstrings e comentarios em portugues, sem acentos no codigo.
- Nenhum fix sem teste RED antes (regra 4 do CLAUDE.md global).
- Testes de comportamento, nao mirroring: assercao sobre status/body/wire,
  nunca sobre retorno de helper interno.
- Nao adicionar dependencia, nao pinar versao.
- Python 3.14, pytest asyncio_mode=auto, respx para HTTP mocked.

## Task 1: telemetry do provider em toda falha de streaming

`_absorb()` (`app/core/dispatcher.py:1047`) so seta `tally.provider` depois
do commit. 2238 de 2320 erros (96,5%) ficam sem provider no banco.

Causa-raiz: `tally.provider` e escrito apenas nos caminhos de sucesso
(`dispatcher.py:1291` passthrough, `:1354` erro-em-banda, `:1372` sucesso).
No 499 (cliente desconectou), 502 de cadeia esgotada, e falha
pre-commit, o `finally` de `dispatch_stream` grava `provider=None`.

Comportamento exigido: a linha do banco identifica QUAL candidato estava
sendo tentado (ou o ultimo tentado) mesmo quando o stream morre.

Restricoes:
- `_Tally` ja carrega `trace`; o provider do candidato tentado e
  `candidate.provider`, conhecido ANTES do loop de leitura.
- Nao inventar provider quando a cadeia foi pulada inteira por skip
  (T4/T5): nesse caso nenhum upstream foi chamado e provider deve
  continuar None, porque o 400 nao e de nenhum provider.
- O caminho bufferizado (`dispatch`, `dispatcher.py:463`) usa
  `result.real_provider`; ele nao tem o bug. Nao tocar.

## Task 2: usage ausente nao pode constar como tokens zero

Evento `e2582e4b3703401087ffcf94c5592de4`: 502, 391248ms, `output_tokens=0`
num stream que entregou 11 kB de texto. O provider (atria) nunca enviou
`usage` no chunk final.

Causa-raiz: `OpenAIStreamToAnthropic.usage()` devolve
`{"input_tokens": 0, "output_tokens": 0}` quando nenhum chunk trouxe
`usage`. A linha de log grava isso como se fosse uma medicao real.

Comportamento exigido: o painel deve distinguir "0 tokens medidos" de
"provider nao reportou usage".

Restricoes:
- `RequestLog` (`app/stats/models.py`) tem `input_tokens: int` NOT NULL e
  `output_tokens: int` NOT NULL, e a coluna do DB e NOT NULL. A
  distinguibilidade tem que caber nesse contrato, ou a task migra a
  coluna (explicito na decisao da task).
- `usage.py::openai_usage_to_anthropic` e o ponto unico de traducao de
  usage; qualquer marcador nasce la.
- Nao quebrar a leitura do painel existente
  (`app/stats/queries.py`, `app/stats/analysis.py`).

## Task 3: teste de streaming 200 no host oficial (lacuna de cobertura)

Nenhum teste faz `stream: True` chegar a
`https://api.anthropic.com/v1/messages` ou `https://api.openai.com/v1` com
`NO_ANTHROPIC_DECLARED` (`tests/core/test_dispatcher.py:357`,
`default_model=None`, provider ausente). Os tres testes existentes com
essas settings (`test_dispatcher_stream.py:1803/1840/1868`) esperam 400.

Comportamento exigido por cada teste (todos RED antes do fix, se algum
passar de cara e porque a lacuna nao e de producao):

3a. Stream OpenAI→Anthropic no host oficial: provider openai ausente do
    catalogo, caller pede `gpt-5` com `stream: True`, credencial do
    caller no `authorization`. Assoertar: netloc `api.openai.com`, path
    `/v1/chat/completions`, `authorization` do caller repassado, SSE
    traduzido no fio do cliente (eventos Anthropic: message_start,
    content_block_delta, message_delta/message_stop), status 200.

3b. Stream Anthropic→OpenAI no host oficial: provider anthropic ausente,
    caller pede `claude-*` com `stream: True`, `x-api-key` do caller.
    Assoertar: netloc `api.anthropic.com`, path `/v1/messages`, SSE
    OpenAI no fio (`data: {...}` + `[DONE]`), status 200.

3c. Mutacao-guarda: quebrar a traducao SSE so no caminho transparente
    deve falhar o teste. (Coberto por 3a/3b se as assercoes forem sobre
    o wire.)

Restricoes:
- Usar respx como os vizinhos (`tests/core/test_dispatcher_stream.py`).
- O provider OpenAI oficial tem `/v1` na base; o Anthropic nao. O path
  final nao pode duplicar prefixo (`/v1/v1/messages`).
- `NO_ANTHROPIC_DECLARED` ja existe e e importado de
  `tests/core/test_dispatcher.py`.

## Ordem

1, 2, 3. Task 1 primeiro porque e a que da visibilidade para medir as
outras no painel.
