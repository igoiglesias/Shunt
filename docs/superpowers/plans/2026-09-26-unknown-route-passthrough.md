# Plano: passthrough transparente de toda rota que o Shunt não implementa

Status: plano verificado e NÃO executado. O usuário resolveu todas as decisões (seção 5).

Origem: o `implementation-planner` escreveu o plano em duas rodadas: o desenho inicial e a revisão com as decisões do usuário. Quatro verificadores só-leitura conferiram tudo contra o código em 2026-09-26. Nenhum comportamento divergiu do código. As correções de linha e de referência de teste já estão aplicadas abaixo.

## 1. Objetivo

Toda requisição cujo par (método, path) o Shunt não implementa é repassada byte a byte ao host oficial do protocolo do chamador (Anthropic ou OpenAI), com a credencial do próprio chamador. O repasse exige um token Shunt válido, e esse token nunca sai para o upstream.

O repasse é gravado no banco e aparece na tela de auditoria com o tipo `relay`. Ele NÃO conta como chamada de modelo: nenhum total, taxa, série, snapshot, SSE, dossiê ou análise o inclui.

Continuam intactos: as rotas implementadas (`/v1/messages`, `count_tokens`, `chat/completions`, `completions`, `embeddings`, `models`), o admin, o painel, `/docs` e a forma `/t/<token>/`.

## 2. Estado atual (verificado)

**Roteamento e ponto de entrada**
- **O 404 de hoje vem do Starlette.** O app não tem catch-all nem handler de 404 (conferido com `grep` em `app/`). Medido com TestClient:
  - `GET /api/oauth/usage` → `404 {"detail":"Not Found"}`;
  - `OPTIONS /v1/messages` → `405 {"detail":"Method Not Allowed"}`.
- **App e rotas finais.** `app/main.py:253-261` monta o app com `FastAPI(..., redirect_slashes=False)` e registra `TokenPrefixMiddleware`. Os routers entram em `:345-354`, com `dashboard` e `audit` atrás de `require_admin`. `/health`, `/` e `/shunt.css` são as últimas rotas (`:366`, `:387`, `:398`).
- **Um catch-all registrado por último vence o match PARTIAL.** Medido com FastAPI 0.141.1 e Starlette 1.6.0:
  - `OPTIONS /v1/messages` e `DELETE /health` caem no catch-all;
  - `POST /v1/messages` continua na rota real;
  - `TRACE` → 405.

  Em `app.routes`, os routers incluídos são objetos `fastapi.routing._IncludedRouter` que aceitam `.matches(scope)`. Medido: para `OPTIONS /v1/messages`, a chamada devolve `Match.PARTIAL`.
- **Middleware de prefixo.** `app/core/prefix.py:14-20` reescreve o path e grava o token em `scope["state"]["shunt_path_token"]`. A query continua com `token=...`, e o relay precisa removê-lo.
- **Rotas locais que dividem espaço com a Anthropic.** O painel usa `/api/stats*` (`dashboard.py:160,201,293`) e `/api/requests*` e `/api/analysis*` (`audit.py:269-606`). O admin fica em `/admin/*` (`admin_dashboard.py:18` com prefixo `/admin`: `/admin/painel` e `/admin/requests`). O FastAPI serve `/openapi.json`, `/docs`, `/docs/oauth2-redirect` e `/redoc`. Como a Anthropic também usa `/api/*`, a reserva é por rota registrada, não por prefixo.

**Protocolo, token e repasse**
- **Dialeto.** `app/routers/v1.py:110` define `ANTHROPIC_AGENTS = ("claude-cli", "anthropic")`. `v1.py:119-134` define `detect_protocol`: `anthropic-version` ou `x-api-key` → anthropic; UA → anthropic; `authorization` → openai; senão `unknown`. `v1.py:138-144` define `protocol_of`: fora de `/v1/messages*` e `/v1/models*`, devolve sempre `openai`. `main.py:338-342` usa `protocol_of` no envelope do `TokenRejected`.
- **Harness.** O stub medido é `docs/superpowers/measurements/2026-09-23-harness-stub.jsonl`:
  - linha 2: `HEAD /api/hello` com os headers `connection: keep-alive`, `user-agent: Bun/1.4.3`, `accept: */*`, `host` e `accept-encoding`, sem nenhum token;
  - linha 6: `HEAD /t/***/api/hello`, com os mesmos headers.
- **Token.** `app/core/token_auth.py:147-185` define `require_shunt_token`, que marca `credential_is_token` (`:177`) sem dizer qual header casou. `mask_path` fica em `:38` e `mask_query` em `:42`. `EXEMPT_PATHS` está vazio (`:85`).
- **Repasse** (`app/core/dispatcher.py`):
  - `TRANSPARENT_DROP`: `:90-100`;
  - `error_body`: `:127-141`;
  - `outbound_headers`: `:183-202`;
  - `_exception_text`: `:340-351`;
  - envio: `build_request` + `send(stream=True)` em `:1244-1253`, com `finally: aclose` em `:1295-1296`.
- **Host oficial e URL.** `app/core/official_hosts.py:13` define o `Literal` `Protocol`. `:22-25` define `OFFICIAL_HOSTS`, com `openai → https://api.openai.com/v1`. `OfficialHost` não tem `origin`. Medido: juntar `/v1/files` a esse `base_url` produz `/v1/v1/files`.
- **Cliente.** `app/core/upstream.py:276` define `pool.client`. Para um provedor fora do catálogo, ele cria um cliente próprio e o fecha na saída (`:297-301`). O `TIMEOUT` é connect 10, read 60, write 30 e pool 10 (`config.py:34-37`).
- **gzip.** Medido com respx: `aiter_raw()` entrega os bytes comprimidos intactos.

**Banco e painel**
- **Colunas.** Em `app/stats/models.py`:
  - `rule`: `String(16)` NOT NULL (`:51`), com os valores `exact`, `family`, `default` e `transparent` (`resolver.py:138-144`) e `none` (`v1.py:353`);
  - `route`: `String(64)` (`:45`);
  - `input_tokens` e `output_tokens`: NOT NULL com default 0 (`:61-62`);
  - índice em `started_at` (`:98`).

  `tests/stats/test_models.py:89` checa só parte dos índices, não o conjunto inteiro.
- **Migração.** `app/stats/engine.py:111-137` define `add_missing_columns`, que faz `ALTER TABLE ADD COLUMN` compilando só o tipo (`:132-133`), sem default. `_create_schema` chama `create_all` e depois `add_missing_columns` (`:227-229`). Os testes de banco antigo seguem o padrão de `tests/stats/test_engine.py:252` e `:290`.
- **Recorder.**
  - `_enqueue` é privado (`app/stats/recorder.py:131-143`).
  - `record` publica no barramento e depois enfileira (`:145-152`).
  - A escrita faz `RequestEvent(**event)` (`:213`).
  - O `except` de `:218-223` envolve o laço inteiro do lote (`:196-215`): um evento com chave desconhecida derruba o commit do lote inteiro e só loga warning. Nenhum teste cobre um lote misto (`test_recorder.py:89-96` usa `batch_size=1`).
- **SSE.** `app/routers/dashboard.py:313-339` assina o barramento, e `dashboard.html:1250-1272` o consome.
- **Consultas.** Onde cada leitura de `RequestEvent` acontece:
  - `app/stats/queries.py`, agregados:
    - `totals`: 3 selects em `:70`, `:94` e `:100`;
    - `series`: `:211` e `:223`;
    - `_grouped`: `:308-330`, usado por `by_model`, `by_provider`, `by_route` e `by_requested_model`;
    - `_providers_of`: `:368`;
    - `by_project`: `:425-447`;
    - `pairs`: `:484-496`;
    - `errors_by_type`: `:511`;
    - `chain_health`: `:562`;
    - `tool_usage`: `:611`;
    - `recent`: `:638`;
    - `facets.distinct`: `:934`.
  - `snapshot` (`:983`) só compõe esses agregados.
  - Leituras que ficam inclusivas: `search_events` (`:815`), `event_detail` (`:900`), `body_of` (`:905`) e `delete_events` (`:969`).
  - Filtros: `_search_clauses` (`:759-777`) é chamada posicionalmente em `:851-855` e por nome em `app/stats/dossier.py:340`.
  - Dossiê: `dossier.py:344` e `:348` leem `RequestEvent` com essas mesmas cláusulas e agregam em Python (`:372-378`). Por isso, um filtro injetado em `:340` cobre o dossiê.
  - `dossier.py:334-336` recusa filtro desconhecido com `ValueError`. `FILTER_FIELDS` fica em `:46-64`. `analysis.py:188` chama `build`.

  `grep -rln RequestEvent app/` só acha `queries.py`, `recorder.py`, `models.py` e `dossier.py`.
- **API de auditoria** (`app/routers/audit.py`):
  - `CSV_COLUMNS`: `:33`;
  - `filters_from`: `:103`, com o contrato "valor que não parseia é filtro ausente" em `:105-107` e `:133-138`, testado em `tests/stats/test_audit_api.py:100`;
  - `_ANALYSIS_FILTER_PARAMS`: `:161`;
  - `_FILTER_PARAMS`: `:201`;
  - `_analysis_filters`: `:464`, que faz `pop("order_by")` em `:469`.

  `tests/test_openapi_api.py:32-33` deriva os parâmetros documentados dessas funções.
- **Tela de auditoria** (`app/templates/audit.html`):
  - `.tag`: `:58`;
  - bloco `.refine`: `:357-389`;
  - `compact(0)` devolve `"0"`: `:427-428`;
  - `SELECTS`: `:453-458`;
  - `fill`, chamada por id: `:467-470`;
  - laços sobre `SELECTS`: `:509` e `:1142`;
  - `servedLabel`: `:603-608`;
  - `rowOf`: `:610-642`, com a soma de tokens em `:611`;
  - `detailOf`: `:855-918`, com a soma de tokens em `:856` e "Regra de rota" em `:897`.

  Nenhum teste afirma a contagem de `<select>` ou o conjunto de ids de filtro.
- **Plano de replay** (`2026-09-26-harness-bypass-replay.md`): não foi implementado. O código não tem `token_headers`.

## 3. Desenho

1. **Catch-all.** Criar `app/routers/relay.py` com `@router.api_route("/{path:path}", methods=[GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS], include_in_schema=False)` e incluí-lo em `main.py` como ÚLTIMA rota, depois de `estilo()`. O handler segue esta ordem:
   - a. path em `RESERVED_EXACT` (`/`, `/health`, `/shunt.css`, `/openapi.json`, `/docs`, `/docs/oauth2-redirect`, `/redoc`) ou começando com `/admin/` → 404 local `{"detail":"Not Found"}`;
   - b. `Match.PARTIAL` numa rota local fora de `/v1` (percorrer `request.app.routes`, sem contar o próprio catch-all) → 405 local;
   - c. `await require_shunt_token(request)`; um `TokenRejected` sobe para o handler de `main.py:338`;
   - d. repasse (item 4).

   `/v1/*` com método não implementado vai ao upstream. Não existe 404 por falta de sinal.
2. **`relay_target(headers) -> Protocol`** em `app/core/relay.py`. `detect_protocol` NÃO muda e é reutilizada como última etapa. Ordem:
   - a. Sinais fortes da Anthropic: um header cujo nome comece com `anthropic-`, `x-api-key`, `authorization: Bearer sk-ant-...` ou um UA em `ANTHROPIC_AGENTS` → `anthropic`.
   - b. Sinais fortes da OpenAI: um header começando com `openai-` ou um UA contendo `openai` → `openai`.
   - c. `detect_protocol(headers)`: `openai` quando há só `authorization`; `unknown` → `anthropic`.

   A regra (a) precisa vir antes de `detect_protocol`, que olha `authorization` antes de `anthropic-beta`. Sem ela, um bearer OAuth da Anthropic iria para a OpenAI.

   **Consequência:** `HEAD /api/hello` do Bun, sem token, recebe 401 no envelope Anthropic. Com `/t/<token>/`, ele é repassado a `api.anthropic.com`.
3. **Token.** O relay exige um token válido. `require_shunt_token` passa a gravar também `request.state.token_headers: frozenset[str]` e mantém `credential_is_token`. Antes do envio, o relay remove `x-shunt-token`, `?token=`, o prefixo e todo header listado em `token_headers`, e a requisição segue sem eles. O relay nunca injeta a chave do catálogo.
4. **Repasse verbatim.**
   - Método: verbatim.
   - Path: `request.url.path` sobre `OfficialHost.origin`.
   - Query: `parse_qsl(..., keep_blank_values=True)`, sem `token`, passada como `params=`.
   - Corpo: `await request.body()` cru.
   - Headers de saída: todos os do cliente, menos `RELAY_DROP = {host, content-length, connection, keep-alive, transfer-encoding, te, upgrade, proxy-authorization, proxy-connection, x-shunt-token, cookie}` e menos `token_headers`. `accept-encoding` passa.
   - Resposta: status verbatim; headers menos `{transfer-encoding, connection, keep-alive}`; corpo em `ClosingStreamingResponse` com `aiter_raw()`. O `finally` fecha `response` e o `AsyncExitStack`.
   - Cliente: `pool.client(f"relay-{protocolo}", ProviderConfig(base_url=origin, protocol=...))`.
   - Sem retry, deadline nem slot.
   - Falha antes do status: `httpx.TimeoutException` → 504; qualquer outro `httpx.HTTPError` → 502. O corpo é `error_body(protocolo, status, _exception_text(err))`.
5. **Envelope do 401.** `protocol_of` devolve `relay_target(headers)` quando o path não começa com `/v1/`.
6. **Registro fora dos agregados.**
   - **Coluna nova.** `RequestEvent.kind: Mapped[str | None] = mapped_column(String(16), nullable=True)`. Não reutilizar `rule`: ela significa "que regra de rota casou" e aparece na tela como "Regra de rota". `kind` precisa ser nullable porque `add_missing_columns` não aplica default. Linhas antigas ficam NULL, e **NULL significa modelo**.
     - Medido com sqlite3: `k != 'relay'` exclui o NULL. Por isso o filtro obrigatório é `_model_rows() = or_(RequestEvent.kind.is_(None), RequestEvent.kind != "relay")`.
     - Sem índice novo.
   - **Escrita.** `as_event` passa a emitir `"kind": "model"`. `Recorder.store(event)` é público e só enfileira, sem publicar no barramento; assim o relay nunca aparece no SSE nem em `recent`. `log_relay(RelayLog)` emite uma linha JSON pelo logger `shunt` e chama `store` com um evento completo:
     - identificação: `kind="relay"`, `route=mask_path(path)`, `dialect=<protocolo alvo>`;
     - sem modelo: `requested_model=""`, `rule="none"`, `matched`, `provider` e `candidate_model` = `None`;
     - resultado: `status`; `error_type` = `upstream_timeout` no 504, `upstream_unreachable` no 502 e `None` no resto; `duration_ms`;
     - campos NOT NULL: `input_tokens` e `output_tokens` = 0 (o banco não aceita NULL);
     - o resto vazio: `ttft_ms=None`, `attempts=[]`, `fell_back=False`, `tools_*=[]`, `thinking_blocks=0`; `project`, `session_id` e `cached_*` = `None`.
   - **A coluna precisa existir antes da primeira escrita.** Uma chave desconhecida derruba o lote inteiro (`recorder.py:196-223`).
   - **Agregados excluem o relay.** `_model_rows()` entra em todos os statements agregados de `queries.py` listados na seção 2.
   - **Auditoria inclui o relay.** `search_events`, `event_detail`, `body_of` e `delete_events` não mudam.
   - **Filtro de tipo.** `_search_clauses(..., kind: str | None = None)` ganha o parâmetro `kind` por ÚLTIMO:
     - `None` → sem cláusula;
     - `"model"` → `_model_rows()`;
     - `"relay"` → `kind == "relay"`;
     - qualquer outro valor → `ValueError`.
   - **Dossiê.** `dossier.py:340` passa `kind="model"` fixo, o que cobre `:344` e `:348`. `kind` NÃO entra em `FILTER_FIELDS`, e `_analysis_filters` faz `pop("kind", None)`.
   - **Borda HTTP.** `filters_from` ganha `kind`; um valor desconhecido vira filtro ausente, conforme o contrato de `audit.py:105-107`. `_FILTER_PARAMS` documenta `kind` com `enum ["model","relay"]`. `CSV_COLUMNS` ganha `kind`.
   - **Ausente, não zero.** Para `kind == "relay"`, `_as_event` devolve `input_tokens`, `output_tokens`, `ttft_ms`, `cached_input_tokens` e `cache_write_tokens` como `None`, e inclui `kind`. No CSV, a célula fica vazia.
   - **Tela.**
     - `<select id="kind">` estático no `.refine`, com as opções "todas", "só modelo" e "só relay", e `kind: "tipo"` em `SELECTS`.
     - `servedLabel` devolve "repasse".
     - `rowOf` ramifica por `kind`, porque em JS `null + null === 0` e `compact(0)` devolve `"0"`. A linha mostra "—" em Pedido, Tokens e Sinais, e `<span class="tag relay">repasse · {dialect}</span>`.
     - `detailOf` esconde "Regra de rota", "Tokens", "Cache", "Raciocínio", "Cadeia" e "Ferramentas".
     - `dashboard.html` não muda.

## 4. Tarefas

Em cada tarefa, o portão é de TAREFA: RED real antes do GREEN, rodando só os testes focados e os vizinhos afetados. Cole o comando e a saída de cada RED e de cada GREEN. O implementador NÃO roda o portão de história nem o navegador.

Antes de T1, quem coordena mede a baseline: `uv run pytest -q tests --ignore=tests/browser`, `uv run ruff check app tests` e `uv run mypy app`. A coleta de hoje tem 1536 testes; a execução não foi medida.

### T1 — `OfficialHost.origin`
- **Arquivos:** `app/core/official_hosts.py`, `tests/core/test_official_hosts.py`.
- **Mudança:** `origin` igual a `str(httpx.URL(base_url).copy_with(path="/", query=None, fragment=None))`, sem a barra final. Os valores de `OFFICIAL_HOSTS` não mudam.
- **RED:** `openai.origin == "https://api.openai.com"` e `anthropic.origin == "https://api.anthropic.com"`; hoje falha com `AttributeError`.
- **Rodar:** `uv run pytest -q tests/core/test_official_hosts.py`.

### T2 — `token_headers`
- **Arquivos:** `app/core/token_auth.py:147-185`, `tests/core/test_token_auth.py`.
- **Mudança:** inicializar `frozenset()` na entrada e no caminho isento; no laço de `:171-178`, acrescentar o nome do header que casou. `credential_is_token` continua existindo.
- **RED 1:** com `{"x-api-key": TEST_SHUNT_TOKEN, "authorization": "Bearer sk-real"}`, espera `frozenset({"x-api-key"})`.
- **RED 2:** sem token em credencial, espera `frozenset()`.
- **Rodar:** `uv run pytest -q tests/core/test_token_auth.py`.

### T3 — Regras puras do relay
- **Arquivos novos:** `app/core/relay.py`, `tests/core/test_relay.py`.
- **Conteúdo:** `relay_target`, `relay_headers(headers, token_headers)`, `relay_params(query)`, `RELAY_DROP`, `RESPONSE_DROP`.
- **REDs:**
  1. `anthropic-beta: oauth-2025-04-20` + `authorization: Bearer x` + UA `Bun/1.4.3` → `anthropic`.
  2. Só o UA `claude-cli` → `anthropic`.
  3. Só `Bearer sk-ant-...` → `anthropic`.
  4. `openai-organization`, ou o UA `openai-python` → `openai`.
  5. Só `authorization: Bearer sk-1` → `openai`.
  6. Os headers exatos da linha 2 do stub → `anthropic`; `{}` → `anthropic`.
  7. `relay_headers` descarta `Host`, `Content-Length`, `Connection`, `x-shunt-token` e `cookie`, e mantém `accept-encoding` e `anthropic-beta`.
  8. Com `x-api-key` em `token_headers`, `x-api-key` sai da requisição e `authorization` fica.
  9. `relay_params("beta=true&token=t&x=")` → `[("beta","true"),("x","")]`.
- **Rodar:** `uv run pytest -q tests/core/test_relay.py tests/routers/test_models.py`.

### T4 — Coluna `kind`, migração, `Recorder.store`, `log_relay`
- **Arquivos:** `app/stats/models.py`, `app/stats/recorder.py`, `app/core/observability.py`, `tests/stats/test_engine.py`, `tests/stats/test_recorder.py`, `tests/core/test_observability.py`.
- **Mudança:** conforme o desenho 6 (escrita), com uma docstring em `models.py` explicando por que NULL significa modelo.
- **REDs:**
  1. `test_engine.py`: um banco antigo com `request_events` sem `kind`, no padrão de `:252`. Depois de `build_engine`, a coluna `kind` existe, `add_missing_columns(engine) == []` e a linha antiga devolve `("antiga", None)`.
  2. `test_recorder.py`: pela API pública (`start`, `store`, `aclose`, como em `:68-78`), com um assinante do barramento, `store({... kind: "relay" ...})` grava 1 linha, e a fila do assinante fica vazia. Hoje falha com `AttributeError: store`.
  3. `test_observability.py`: `log_request` com um recorder espião grava um evento com `kind == "model"`.
  4. `test_observability.py`: `log_relay` com `caplog` e um espião produz a linha `kind == "relay"`, sem nenhum valor de credencial nem `token=` em claro. O espião registra 1 `store` e 0 `record`, e o evento tem `rule == "none"`, `requested_model == ""`, `input_tokens == 0` e `candidate_model is None`.
- **Rodar:** `uv run pytest -q tests/stats/test_engine.py tests/stats/test_recorder.py tests/core/test_observability.py tests/stats/test_models.py`.

### T5 — Catch-all e repasse, com log
- **Arquivos:** `app/routers/relay.py` (novo), `app/main.py`, `app/routers/v1.py:138-144`, `tests/routers/test_relay.py` (novo, com respx).
- **Fixture:** `app.state.settings = Settings(providers={}, models={}, routes=[])` e `app.state.pool = UpstreamPool(settings)`.
- **REDs:** hoje todos dão 404 ou 405. Um por vez:
  1. `GET /api/oauth/usage?beta=true&token=X` com `anthropic-beta: oauth-...`, `authorization: Bearer sk-ant-oat-1` e `x-shunt-token`.
     - Espera 200 e o corpo idêntico.
     - `x-request-id` volta ao cliente.
     - A query enviada é `b"beta=true"`.
     - Os headers enviados não contêm `x-shunt-token`, o `host` do cliente nem `cookie`.
  2. `POST /v1/messages/batches` com o corpo `b'{"a":1,'` e `x-api-key: sk-ant-1`. O upstream recebe exatamente esses bytes. Um 400 do upstream volta como 400, com o corpo dele.
  3. `GET /v1/files` com `openai-organization` + `Bearer sk-1` → `https://api.openai.com/v1/files`, e não `/v1/v1/files`.
  4. `x-api-key: TEST_SHUNT_TOKEN` + `anthropic-version` → o valor do token não sai em nenhum header, e `call_count == 1`.
  5. Headers da linha 2 do stub com `without_shunt_token` → 401 no envelope Anthropic e `call_count == 0`. O segundo RED esperado é o envelope openai, antes do ajuste em `protocol_of`.
  6. `without_shunt_token` + só `Bearer sk-1` → 401 no envelope OpenAI.
  7. Nenhuma destas vai ao upstream (`call_count == 0`):
     - `DELETE /health` → 405;
     - `POST /api/stats` → 405, ou o status medido de `require_admin` (colar a saída);
     - `GET /admin/nao-existe` → 404;
     - `GET /docs` → 200 HTML.
  8. `OPTIONS /v1/messages` com `anthropic-version` vai ao upstream.
  9. `HEAD /t/<TEST_SHUNT_TOKEN>/api/hello` com os headers da linha 6 do stub → o upstream recebe `HEAD /api/hello`, sem prefixo e sem `token`, e o status volta verbatim.
  10. Resposta gzip → o cliente recebe os bytes comprimidos, o header `content-encoding` e o mesmo `content-length`.
  11. `httpx.ConnectError` → 502 no envelope do chamador; `httpx.ReadTimeout` → 504.
  12. Com um recorder espião:
      - depois de um relay 200, há exatamente 1 `store`, com `kind == "relay"`, `route == "/api/oauth/usage"`, `dialect == "anthropic"` e `status == 200`, e 0 publicações;
      - depois de um `ReadTimeout`, `status == 504` e `error_type == "upstream_timeout"`.
- **Rodar:** `uv run pytest -q tests/routers/test_relay.py tests/routers/test_messages.py tests/routers/test_models.py tests/test_openapi.py tests/routers/test_error_surface.py tests/routers/test_v1_auth.py`.

### T6 — Cancelamento fecha o upstream
- **Arquivo:** `tests/routers/test_relay.py`.
- **Teste:** transporte httpx com um `AsyncByteStream` que registra `aclose()`. Consumir um chunk de `body_iterator` e chamar `aclose()`; o teste espera `closed is True`. Cancelar a task durante `__anext__`; o `CancelledError` sobe e `closed is True`.
- **RED:** o teste nasce verde e fica como caracterização do fechamento do upstream. Nenhum teste de mutação (decisão 6).
- **Rodar:** `uv run pytest -q tests/routers/test_relay.py tests/test_shutdown.py`.

### T7 — Agregados excluem o relay; busca e dossiê
- **Arquivos:** `app/stats/queries.py`, `app/stats/dossier.py`, `tests/stats/test_queries.py`, `tests/stats/test_search.py`, `tests/stats/test_dossier.py`.
- **Mudança:** `_model_rows()` em todos os agregados da seção 2; `_search_clauses(..., kind=None)`; `dossier.py:340` passa `kind="model"`.
- **Linha de relay para os testes.** Em `test_queries.py`, que já tem `row(**over)` em `:50` e a fixture `seeded` em `:79`, criar:
  - `relay_row(**over) = row(kind="relay", requested_model="", rule="none", matched=None, provider=None, candidate_model=None, route="/api/oauth/usage", dialect="anthropic", status=502, error_type="upstream_unreachable", duration_ms=99999, attempts=["x: 502 (attempt 1)"], tools_offered=["read"], tools_called=["read"])`.
  - Os valores extremos são de propósito: se a linha vazar, ela mexe em erro, p95, cadeia, ferramentas e facetas.
- **REDs.** Em cada um, semear `seeded` + 1 `relay_row` e afirmar o MESMO número do teste vizinho já existente:
  1. `totals`: `requests == 4`, `errors == 1` e `p95_duration_ms == 400`.
  2. `series`: a soma de `requests` é 4.
  3. `by_model`.
  4. `by_provider`.
  5. `by_route`: sem `/api/oauth/usage`.
  6. `by_requested_model`.
  7. `by_project`: a linha "sem projeto" não cresce.
  8. `pairs`.
  9. `errors_by_type`: sem `upstream_unreachable`.
  10. `chain_health`: `requests == 4`, sem o pulo `x`.
  11. `tool_usage`: `read` chamado 1 vez, não 2.
  12. `recent`: o relay não aparece.
  13. `facets`: `routes` e `error_types` sem os valores do relay.
  14. `snapshot`: `totals.requests == 4`.
  15. Uma linha com `kind=None` CONTA em `totals`, o que prova o `is_(None)`.
  16. `test_search.py`: sem `kind`, o relay aparece; com `kind="relay"`, só ele; com `kind="model"`, só os outros; com `kind="x"`, `ValueError`.
  17. `test_dossier.py`: com um `relay_row`, `volume.requests` e `errors` não mudam; `build(filters={"kind": "relay"})` levanta `ValueError`.
- **Rodar:** `uv run pytest -q tests/stats/test_queries.py tests/stats/test_search.py tests/stats/test_dossier.py tests/stats/test_analysis.py tests/stats/test_dashboard_api.py`.

### T8 — API de auditoria: filtro `kind`, CSV, ausente-não-zero
- **Arquivos:** `app/routers/audit.py`, `app/stats/queries.py` (`_as_event`), `tests/stats/test_audit_api.py`, `tests/test_openapi_api.py`.
- **REDs:**
  1. `GET /api/requests` com um relay semeado → o evento vem com `kind == "relay"` e `input_tokens`, `output_tokens` e `ttft_ms` iguais a `None`.
  2. `?kind=relay` → só ele; `?kind=model` → só os outros; `?kind=qualquer` → todos, sem 4xx.
  3. `GET /api/requests/export` → o cabeçalho contém `kind`, e a linha do relay tem a célula `input_tokens` VAZIA.
  4. `POST /api/analysis?kind=relay` não devolve 500, e o dossiê ecoa os filtros sem `kind`.
  5. `tests/test_openapi_api.py` quebra quando `filters_from` ganha a chave antes de `_FILTER_PARAMS` (colar a saída), e depois volta a ficar verde.
- **Rodar:** `uv run pytest -q tests/stats/test_audit_api.py tests/test_openapi_api.py tests/stats/test_analysis_api.py`.

### T9 — Tela de auditoria
- **Arquivos:** `app/templates/audit.html`, `tests/stats/test_dashboard_page.py`.
- **Mudança:** conforme o desenho 6 (tela).
- **RED de tarefa:** o HTML de `/admin/requests` contém `id="kind"` e as três opções.
- **Rodar:** `uv run pytest -q tests/stats/test_dashboard_page.py tests/stats/test_audit_api.py`.
- **Validação de tela (roda quem coordena, não o implementador):** `frontend-validator` + skill `visual-qa` + skill `ssr-ui-mobile-first`, headless, a 390 px e 1366 px, com um relay e um modelo na lista.
  - Confere que a linha de relay mostra "—" e "repasse".
  - Confere que o detalhe esconde as seções de modelo.
  - Confere que o select `kind` filtra e persiste na URL.
  - Os screenshots são lidos, não só salvos. A revisão de design fica com `frontend-design:frontend-design`.

### T10 — E2E, README e docstrings (PORTÃO DE HISTÓRIA)
- **Fake provider:** `tests/e2e/fake_provider.py:79-87` registra só 5 paths POST, e `handle` faz `await request.json()` (`:54-55`). Acrescentar, DEPOIS desses 5 paths, um `api_route("/{path:path}")` com os 7 métodos. O handler guarda `await request.body()` em `raw_bodies` e responde o próximo `Scripted`.
- **E2E** (`tests/e2e/test_relay_e2e.py`):
  1. `GET /api/oauth/usage?beta=true` com OAuth Anthropic → host `api.anthropic.com`, o mesmo path e a query `beta=true`, sem `x-shunt-token`.
  2. `POST /v1/messages/batches` com corpo cru → os bytes chegam idênticos.
  3. `HEAD /api/hello` com UA Bun e sem token → 401 no envelope Anthropic, sem chamada ao upstream.
  4. `HEAD /t/<token>/api/hello` chega ao fake como `HEAD /api/hello`.
  5. `POST /v1/messages` continua passando pela cadeia `opus`.
  6. Depois dos relays:
     - `GET /api/requests?kind=relay` (com sessão admin) lista os relays;
     - `GET /api/stats` mostra `totals.requests` igual ao número de `POST /v1/messages` feitos;
     - `recent` não contém nenhuma rota `/api/`.
  7. Um relay com path de 100+ caracteres é gravado inteiro no SQLite de arquivo.
- **README:** nova seção "Routes Shunt does not implement", depois de `README.md:167` (o separador está em `:169`). Ela explica que o token é obrigatório, que o destino sai dos sinais nos headers com Anthropic como padrão, e que essas chamadas aparecem na tela de requisições com o tipo `relay` sem entrar nos números do painel. Acrescentar uma frase sobre o filtro de tipo em "The request screen" (`README.md:494`) e outra à `DESCRIPTION` de `main.py:169`.
- **Docstrings** em português sem acento:
  - `models.py`: por que NULL significa modelo;
  - `queries.py`: por que `_model_rows` entra em todo agregado;
  - `relay.py`: FULL vence PARTIAL, o gzip passa cru e o relay usa a origem em vez do `base_url`.
- **Fechamento (roda quem coordena):** a suíte sem browser, `ruff` e `mypy`, comparados com a baseline medida antes de T1. Depois, `strict-code-reviewer` e a skill `story-gate`, SEM o passo de mutação (decisão 6).

## 5. Decisões (resolvidas pelo usuário em 2026-09-26)

1. O relay exige o token do Shunt, e o token nunca vai ao upstream numa requisição transparente.
2. Sem sinal claro, vale a regra do `detect_protocol` (bearer sozinho vai para a OpenAI; o resto vai para a Anthropic), com os sinais fortes avaliados antes. Nunca há 404 por falta de sinal.
3. Toda chamada repassada é gravada e aparece no painel de auditoria, mas não conta como chamada de modelo.
4. Quando a credencial é o próprio token do Shunt, o header é removido e a requisição segue.
5. `/v1/*` com método não implementado vai ao upstream.
6. Nenhuma tarefa faz teste de mutação: nem mutação manual, nem a skill `mutation-sweep`, nem o passo de mutação da `story-gate`.

## 6. Assumido (não verificado)

- Os headers reais das chamadas de login, OAuth e `/usage` do Claude Code: o stub só tem `/api/hello` e `/v1/messages`. Capture uma sessão com `/usage` antes de T3.
- Que o harness tolera 401 em `/api/hello` sem token. Só a tolerância ao 404 foi medida (`2026-09-23-harness-findings.md:26`).
- Que o Turso/libsql remoto também ignora o tamanho de `VARCHAR(64)` em `route`. Só o sqlite3 local foi medido. Se o remoto impuser o limite, crie uma coluna nova `Text` pela mesma migração.
- Que `.matches(scope)` nas rotas atrás de `require_admin` não executa a dependência. O RED 7 de T5 confirma.
- Que o `content-length` do upstream junto de `aiter_raw` não conflita no uvicorn real. O E2E e um `curl --compressed` contra `make dev` confirmam.

## 7. Fora de escopo

- O plano de replay de `model`. Este plano ADICIONA `token_headers` e mantém `credential_is_token`; o replay remove o bool depois.
- Índice para `kind`; `kind` como filtro do dossiê; relay na fita ao vivo.
- Retry, deadline e slots no relay; websocket; `TRACE` e `CONNECT`; tradução de protocolo em rotas desconhecidas; injeção de chave do catálogo.
- Teste de mutação, em qualquer forma e em qualquer tarefa (decisão 6).
