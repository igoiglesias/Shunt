# BRIEF DE IMPLEMENTAÇÃO — Último degrau = replicação da request original do harness (Shunt)

Repo: `/home/iglesias/Documents/Projetos_Pessoais/Shunt`, branch `feat/shunt-proxy`. Parta do código **atual da árvore** (não commitado): `app/core/official_hosts.py`, `app/core/resolver.py`, `app/core/dispatcher.py`, `app/routers/v1.py` e testes já contêm uma implementação que seguiu o brief antigo (`/tmp/shunt-bypass-impl-brief.md`) e a spec R1/R2. Esse desenho está **superado** pela regra de negócio abaixo. Atenção: o `git diff` da árvore também carrega trabalho não relacionado (observabilidade, audit, engine, shutdown) — não toque nele.

## 0. Regras de trabalho (obrigatórias, repasse literal)

- **Regra do usuário:** nenhuma afirmação sobre comportamento sem a medição que a sustenta. Cada fato sobre código com `file:line` lido agora. Rotule cada afirmação como *medi* (rodei e vi a saída — cole comando + linha decisiva), *li* (conferi a fonte, `file:line`) ou *assumi*. Quando não puder verificar, escreva exatamente "não consigo verificar a partir daqui". Ausência de erro não é evidência de funcionamento; exit 0 prova que rodou, não que fez o que devia.
- **Portão cobrado: NÍVEL TAREFA.** Teste unitário de todo comportamento novo ou alterado, com RED real (cole a saída do pytest falhando) ANTES do GREEN, rodando os testes focados e os vizinhos afetados. Teste ausente é defeito da tarefa. Não cobra portão de história, nem navegador, nem E2E de plataforma.
- TDD estrito, um RED por comportamento: escreve o teste -> roda -> cola a falha -> implementa o mínimo -> roda -> cola o verde. Nunca agrupe vários REDs num só ciclo. Teste que nasce verde: diga que nasceu verde e por quê (só aceitável para caracterizar comportamento que já existia; para comportamento novo é sinal de teste errado).
- Comentários e docstrings em português SEM acentos, estilo do repo: docstring de módulo/função registra o **porquê** e o que foi medido. Remova das docstrings toda menção a "spec R1/R2" que deixar de valer.
- Nunca abra navegador. Nunca rode `make cov`/alvos que abram algo. Não rode `tests/browser`.
- No relatório final: o que mudou por arquivo; RED/GREEN de cada item; a lista do que foi *assumi*; e o que "não consigo verificar a partir daqui".

## 1. DECISÃO PENDENTE (o usuário ainda não escolheu; este brief implementa a opção A)

**Regra de negócio (texto literal do usuário, é a autoridade e vence a spec R1/R2 e o brief antigo):**

> "se existir rota, siga a rota até o ultimo fallback (como já existe hoje). depois do ultimo fallback, se o modelo default estiver setado, envie para o modelo default e pare por ai, mesmo que ele não responda. se o modelo default não estiver setado, envie para o modelo original, o que o harness chamou (é só replicar a request original do harness, já tem tudo nela)."
> "não precisa de rota oficial, se ele intercepta o request do harness ele já tem tudo que é preciso para o bypass."

Casos explícitos: (a) modelo não configurado no Shunt; (b) fim do fallback com `default_model=None`. Os dois vão para a **replicação da request original**.

**Opção A (implementada aqui):** o destino da replicação é o host oficial do **protocolo do chamador**, nunca o prefixo do nome do modelo e nunca o catálogo: chamador `anthropic` -> `https://api.anthropic.com`; chamador `openai` -> `https://api.openai.com/v1`. Modelo intacto, corpo sem tradução (mesmo protocolo), headers verbatim menos `TRANSPARENT_DROP` e menos o(s) header(s) cujo valor é o token do Shunt, query string original repassada (menos `token`). Consequências:
- Qualquer nome (`o4-mini`, `gemini-2.5-pro`, `x/y`, `gpt-5` pedido por Claude Code, `modelo-sem-dono`) vai para a replicação. `UnknownProviderError` deixa de ser levantado pelo último degrau (continua existindo só para alias de rota não configurado — ver T1).
- O skip R2 (`_transparent_skip_reason`, dispatcher.py:270-321) **some**.
- `PROVIDER_HINTS` (resolver.py:7-12) deixa de decidir o último degrau, mas **continua usado** por `_candidate` -> `_transparent` (resolver.py:46, :59) para alias de rota que não está em `models`. Não remover.
- Provider declarado com nome `anthropic`/`openai` **nunca** recebe a credencial do harness: a entrada do catálogo só é usada por rota/default, com a chave do catálogo. Isso fecha o vazamento de `outbound_headers` (dispatcher.py:184-189: verbatim para qualquer `base_url` declarado).
- Sem credencial do provedor encaminhável (o harness só mandou o token do Shunt, ou nada), a replicação é **pulada** com mensagem que diz o que falta — não vai ao host oficial sem auth, e a chave do catálogo **não** é injetada (a request original não a tinha).
- Custo em testes (todos rotulados *li*): `tests/core/test_dispatcher.py:1510 test_transparent_credentials_go_verbatim_to_the_declared_base_url` **some** e é substituído pelo inverso (gateway declarado recebe a chave do catálogo, nunca a do harness — T3.b); `:1443 test_transparent_across_protocols_translates_both_halves` e `:1490 test_transparent_translation_uses_the_clients_cap_not_our_default` (GATEWAY, premissa "transparente traduz") são reescritos sobre alias de rota não configurado (T3.b); `:525`, `:549`, `:460` (R2) somem; `tests/routers/test_v1_auth.py:140`, `:160`, `:219` (injeção da chave configurada no transparente) viram "replicação pulada com 400, sem chamada"; `:181` e `:199` mudam de URL/asserção (T5).

**Opção B (se o usuário escolher):** se houver provider declarado cujo nome é o do protocolo do chamador (`anthropic` para chamador anthropic, `openai` para chamador openai) e ele for marcado como gateway, ele é o destino; senão, host oficial. Observação *li*: não existe flag "gateway" em `ProviderConfig` (`app/config/settings.py:8-16`); B exige ou uma coluna nova + tela admin (fora deste brief) ou a convenção por nome. O que muda em relação a A: (1) T3.a — `_provider_config` para candidato `replay` consulta `settings.providers[req.protocol]` antes de `OFFICIAL_HOSTS`; (2) T3.b — `outbound_headers` de replay com destino declarado: credencial do harness verbatim se houver, senão a chave do catálogo (mantém `test_v1_auth.py:140/:160/:181/:219` como estão hoje, URL `api.anthropic.test`); (3) T3.c — o pulo por falta de credencial só vale quando o destino é o host oficial ou o declarado não tem chave; (4) os testes GATEWAY (`:1443`, `:1490`, `:1510`) sobrevivem com o nome de rota original; (5) T7 (README) descreve a precedência declarado > oficial. Tudo o mais (replay por protocolo do chamador, fim do R2, query string, preflight antes do SSE, count_tokens, E2E) é igual.

## 2. Contrato (opção A)

### 2.1 Resolver — `app/core/resolver.py`

```python
@dataclass(frozen=True)
class Candidate:
    alias: str | None
    provider: str
    model: str
    protocol: str
    transparent: bool = False
    # Replicacao da request original do harness: ultimo degrau quando nao ha
    # default_model. Destino = host oficial do PROTOCOLO DO CHAMADOR; o
    # catalogo nao e consultado; headers do harness saem verbatim.
    replay: bool = False


def replay(requested: str, protocol: str) -> Candidate:
    """`provider` e o nome do host oficial, que coincide com o protocolo."""
    if protocol not in OFFICIAL_HOSTS:
        raise ValueError(f"no official host for caller protocol {protocol!r}")
    return Candidate(alias=None, provider=protocol, model=requested, protocol=protocol,
                     transparent=True, replay=True)
```

- `resolve(requested, settings, protocol)` e `last_resort(requested, settings, protocol)` ganham o terceiro parâmetro **posicional obrigatório** (o protocolo do chamador, `"anthropic"` ou `"openai"`). Sem default keyword: um chamador OpenAI esquecido seria um bug silencioso.
- `last_resort`: `default_model` setado -> `[_candidate(default)]`; senão -> `[replay(requested, protocol)]`. **Nunca mais devolve `[]`.**
- `resolve`: ordem exact -> family -> default -> replay, inalterada. A última regra devolve `Resolution("transparent", None, [replay(...)])` (manter o nome de regra `"transparent"`: ele aparece em `RequestLog.rule` e no painel).
- `_transparent` volta à semântica de HEAD (`git show HEAD:app/core/resolver.py`, `_transparent` :48-67): provider derivado por `PROVIDER_HINTS`/`"/"`, **precisa estar em `settings.providers`**, senão `UnknownProviderError`. Remova o fallback `OFFICIAL_HOSTS` (resolver.py:70-79) e o import (resolver.py:4). `_transparent` fica usado **só** por `_candidate` (alias de rota ausente em `models`).
- `_chain` (resolver.py:89-110) inalterada em lógica; só repassa `protocol` a `last_resort`.

### 2.2 Hosts oficiais — `app/core/official_hosts.py`

Manter o arquivo e os valores (`anthropic` -> `https://api.anthropic.com`, `openai` -> `https://api.openai.com/v1`, official_hosts.py:21-24). Reescrever a docstring (official_hosts.py:1-4, que cita "spec R1") para: destino da replicação da request original, indexado pelo **protocolo do chamador**. Tipar como `Mapping[Protocol, OfficialHost]`.

### 2.3 Token — `app/core/token_auth.py` + `ShuntRequest`

- `require_shunt_token` passa a registrar `request.state.token_headers: frozenset[str]` = nomes (minúsculos) dos cabeçalhos de credencial (`x-api-key`, `authorization`) cujo valor casou com um token do Shunt (token_auth.py:171-178 já itera os dois). `credential_is_token` (token_auth.py:151, :154, :177) é substituído por isso.
- `ShuntRequest` (dispatcher.py:155-163): `credential_is_token: bool` **sai**; entram `token_headers: frozenset[str] = frozenset()` e `query: str = ""` (query string original, crua).
- Credencial encaminhável = `{"x-api-key", "authorization"} ∩ {cabeçalhos presentes} − token_headers`. Motivação medida: o stub real do Claude Code traz `authorization` **e** `x-api-key` juntos (`docs/superpowers/measurements/2026-09-23-harness-stub.jsonl:3`, *li*); o bool atual descarta a chave real quando o outro header é o token (token_auth.py:170-178).

### 2.4 Dispatcher — `app/core/dispatcher.py`

- `_provider_config(candidate, settings)`: `if candidate.replay: h = OFFICIAL_HOSTS[candidate.protocol]; return ProviderConfig(base_url=h.base_url, protocol=h.protocol)`; senão `settings.providers[candidate.provider]` (KeyError impossível: todo candidato não-replay vem do catálogo, `Settings._check_references` settings.py:37-52).
- `outbound_headers`: verbatim (menos `TRANSPARENT_DROP` ∪ `req.token_headers`) **somente** se `candidate.replay`. Todo o resto (rota, default, alias de rota não configurado) usa a chave do catálogo como hoje (dispatcher.py:193-198).
- `_replay_skip_reason(candidate, req) -> str | None` substitui `_transparent_skip_reason` (dispatcher.py:270-321, apagar inteiro). Regra única: candidato `replay` sem credencial encaminhável é pulado com: `f"{label}: replay to {host} skipped: no provider credential to forward in x-api-key/Authorization (the shunt token never leaves); send the provider credential or set default_model"`.
- Guarda 400 dos dois laços (dispatcher.py:667-669 e :1207-1212): mantém-se (cadeia inteira pulada por replay e nada tentado), mensagem vira `"not dispatched: " + "; ".join(trace)` (sem "no candidate answered"), e no streaming `tally.error_type = "invalid_request_error"` (hoje `"api_error"` em :1210; o bufferizado grava `invalid_request_error` via `_error_type_of`, :422-426).
- Função pública `replay_blocker(req, resolution) -> str | None`: devolve o motivo se **toda** a cadeia é replay e o replay seria pulado. Usada por `v1.py` para decidir antes do cabeçalho SSE e pelo `count_tokens`.
- Query string: `replay_params(req) -> list[tuple[str, str]]` = `parse_qsl(req.query, keep_blank_values=True)` sem `token`. Passada como `params=` em `_attempts` (dispatcher.py:722-727), `_stream_candidate` (:1244-1249) e no `count_tokens` (v1.py:549-553) **só quando `candidate.replay`**; senão `params=None`.
- `_missing_credential` (dispatcher.py:324-332): inalterado (replay é `transparent=True`, devolve None). `_payload` (:227-245), `_chain_for` (:487-543, mantém transparente em :504) e `_transparent_cap`: inalterados — o replay é mesmo protocolo, logo `{**req.body, "model": candidate.model}` sem reescrita de `max_tokens` (transparent + alias None).

### 2.5 Router — `app/routers/v1.py`

- `_serve` (v1.py:385-427): guarda o `resolution` (hoje descarta, :406); passa `protocol` ao `resolve`; monta `ShuntRequest` com `token_headers=request.state.token_headers` e `query=request.url.query`; **antes** de `if streaming and body.get("stream")` (:417): `if (reason := replay_blocker(shunt_request, resolution))`: `_record(route, protocol, 400, started, requested_model=..., rule=resolution.rule, matched=resolution.matched, error_type="invalid_request_error")` e `JSONResponse(400, error_body(protocol, 400, "not dispatched: " + reason))`. O `except UnknownProviderError` (:407-416) fica (alias de rota não configurado ainda levanta).
- `count_tokens` (v1.py:506-…): `resolve(..., "anthropic")`; monta `shunt_request` antes da condição; encaminha só se `candidate.protocol == "anthropic" and not (candidate.replay and _replay_skip_reason(candidate, shunt_request))`; senão estimativa local (semântica já documentada em :507-514: "encaminha quando dá, estima quando não dá"). Replay encaminha com `params=replay_params(...)` e headers verbatim para `https://api.anthropic.com/v1/messages/count_tokens`.

## 3. Fatos verificados (rótulo em cada linha)

- *medi* `ls .codegraph` -> "No such file": sem índice codegraph; navegação por leitura direta.
- *li* official_hosts.py:21-24: `anthropic -> https://api.anthropic.com` (sem `/v1`), `openai -> https://api.openai.com/v1`.
- *medi* httpx 0.28 join: `base https://api.openai.com/v1 + /chat/completions -> https://api.openai.com/v1/chat/completions`; `+ /embeddings -> .../v1/embeddings`; `base https://api.anthropic.com + /v1/messages -> https://api.anthropic.com/v1/messages`. Logo `PATHS` (dispatcher.py:71-78) casa com os dois bases: chamador openai em `chat`/`completions`/`embeddings` vai para `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`; chamador anthropic em `messages` vai para `/v1/messages`. `("anthropic","embeddings")` e `("anthropic","completions")` não existem em `PATHS` -> "endpoint not supported" (mas o `_serve` fixa protocolo por rota: `/v1/completions` e `/v1/embeddings` são sempre `"openai"`, v1.py:457 e a rota seguinte; logo o replay desses endpoints é sempre openai).
- *li* resolver.py:7-12 `PROVIDER_HINTS`; :35-46 `_candidate` (alias ausente -> `_transparent(requested, settings, model=alias)`); :49-86 `_transparent` com fallback `OFFICIAL_HOSTS` em :70-79; :89-110 `_chain` (dedup por `(alias, model)` :98-107); :113-132 `last_resort` (default :127-128; `[]` quando `UnknownProviderError` :129-132); :135-144 `resolve`.
- *li* dispatcher.py:90-101 `TRANSPARENT_DROP` (inclui `x-shunt-token` e `cookie`); :155-163 `ShuntRequest` (`credential_is_token` :163); :175-180 `_client_presented_credential`; :183-198 `outbound_headers` (verbatim :184-189 para **qualquer** transparente com credencial; chave do catálogo :193); :253-267 `_provider_config`; :270-321 `_transparent_skip_reason` (R2 só para provider não declarado, :309-320); :324-332 `_missing_credential`; :439 e :1066 `resolve(...)`; :539 `last_resort(...)`; :487-543 `_chain_for` (transparente mantido :504-506); :546-… `_dispatch` (skip :592-596; guarda 400 :667-669 com prefixo "no candidate answered: "); :722-727 `client.post(PATHS[...])` sem `params`; :899 `_stream_error(req, message, status=502)`; :1059-1215 `_stream_chain` (skip :1103-1107; guarda :1207-1212, `tally.error_type = "api_error"` :1210); :1244-1249 `build_request` sem `params`; :422-426 `_error_type_of`.
- *li* v1.py:385-427 `_serve` (`resolve` descartado :406; `except UnknownProviderError` :407-416; SSE :417-420); :433/:443/:457 protocolo fixo por rota; :506-… `count_tokens` (`chain[0]` :525; `if candidate.protocol == "anthropic"` :536; `ShuntRequest` :538-544; `_provider_config` :547; path `/v1/messages/count_tokens` :550); :346-360 assinatura de `_record`.
- *li* token_auth.py:147-185 `require_shunt_token` (`credential_is_token=False` :154; laço `x-api-key`/`authorization` :171-178, marca True se **qualquer** um casa :177).
- *li* upstream.py:276-312 `pool.client(provider, config)`: provider fora do catálogo vivo ou `base_url` diferente -> cliente próprio por uso, fechado na saída; se o catálogo declara `anthropic` com `base_url` **igual** a `https://api.anthropic.com`, o replay compartilha o cliente/gate dele (`try_slot(candidate.provider)`, :328-333). Não mudar o pool.
- *medi* `starlette.datastructures.URL('http://x/v1/messages?beta=true&token=t').query == 'beta=true&token=t'`; `parse_qsl` menos `token` -> `[('beta','true')]`.
- *li* `docs/superpowers/measurements/2026-09-23-harness-findings.md:26-27`: Claude Code chama `POST /v1/messages?beta=true` (sempre com a query), também na forma `/t/<token>/`. *li* `…/2026-09-23-harness-stub.jsonl:3`: os headers reais trazem `authorization`, `x-api-key` e `x-shunt-token` ao mesmo tempo (valores mascarados; não consigo verificar a partir daqui qual dos dois é o token).
- *medi* (pelo coordenador, nesta sessão) suíte sem browser: 1531 passed; mypy limpo; ruff 1 erro I001 em `tests/core/test_dispatcher_stream.py:25` (`from tests.core.test_dispatcher import CLIENT_HEADERS, SETTINGS, TRANSPARENT, NO_ANTHROPIC_DECLARED` fora de ordem; `ruff --fix` resolve).
- *li* tests/core/test_dispatcher.py: `SETTINGS` :14-33 (só `openrouter`, rota `("opus", ["free","cheap"])`, sem default); `KEYED` :189; `TRANSPARENT` :204-214 (declara `anthropic` em `https://api.anthropic.test`); `CLIENT_HEADERS` :216-224; `NO_ANTHROPIC_DECLARED` :357-366; `OFFICIAL_ANTHROPIC_OK` :368-376; `GATEWAY` :1424-1433 (`anthropic` em `https://gateway.exemplo/v1`, protocolo openai); `ok_payload` :42. Testes atuais do bypass: :380, :398/:407, :419, :443, :460, :475, :488, :503, :525, :549. Testes que chamam o tail declarado: :248, :1348, :1443, :1490, :1510, :1780; :589 (`test_endpoint_without_a_path…`, chamador openai em `embeddings` com `claude-opus-4-5`, `{}`, espera trace exato de 2 linhas "endpoint not supported").
- *li* tests/core/test_dispatcher_stream.py: `run` :176; `:429 test_transparent_mode_streams_raw_bytes_with_the_clients_own_headers` (TRANSPARENT, api.anthropic.test); guarda 400: `:1804`, `:1840`, `:1876`.
- *li* tests/core/test_resolver.py: `build()` :7-42 declara openrouter/anthropic/local/openai; :112-133, :135-149, :152-174, :177-204 (testes do brief antigo); :275-279 `test_last_resort_is_empty_when_no_provider_can_be_guessed`; :320-325 `test_chain_unchanged_when_no_default_and_no_provider_can_be_guessed`; :215-222, :225-233, :236-242 (alias de rota não configurado — continuam valendo).
- *li* tests/test_config_routes.py:23, :45, :57 chamam `resolve(alias, settings)`; tests/core/test_dispatcher_concurrency.py não chama `resolve` (o grep só achou `_enter_last_resort`).
- *li* tests/core/test_token_auth.py:86-105 (`credential_is_token` True/False); *medi* `grep credential_is_token`: dispatcher.py 4, token_auth.py 3, v1.py 3, test_dispatcher.py 3, test_dispatcher_stream.py 1, test_token_auth.py 4.
- *li* tests/routers/test_v1_auth.py: `TRANSPARENT_SETTINGS` :35-46 (`anthropic` em api.anthropic.test, chave `sk-guardada`); :140, :160, :181, :199, :219.
- *li* tests/routers/test_messages.py: `ASK` :13-17 (`claude-opus-4-5`); `client()` :39-42; `:107-116 test_exhausted_chain_omits_the_shunt_model_header` (my-opus + `x-api-key: sk-do-cliente`); count_tokens :194, :208, :233, :250, :275, :321, :349. Diff de "my-opus"/"modelo-sem-dono" (*medi* `git diff -U0`): test_admin_tokens.py:184-191 (`modelo-sem-dono`), test_error_surface.py:82-93 e :115-121, test_messages.py:107-116, test_openai_routes.py:109-116.
- *li* tests/conftest.py:24-45: fixture autouse `shunt_token_everywhere` injeta `x-shunt-token` em todo `TestClient` (inclusive E2E).
- *li* tests/e2e/conftest.py:11-52 `E2E_SETTINGS` (declara `anthropic` em `http://fake.anthropic`, rotas `("opus",["qwen","free"])`, `("haiku",["free"])`, default None); :65-70 `shunt` fixture. tests/e2e/fake_provider.py:87-97 registra os paths com e sem `/v1`; :100-113 `ScriptedTransport` entrega **toda** request ao fake, qualquer host, e guarda `calls` (httpx.Request). tests/e2e/test_official_host_bypass_e2e.py:30-64: caso (a) só, sem rota de família, não afirma ausência de `x-shunt-token`. E2E afetados: test_transparent_e2e.py:86-98 (`modelo-sem-dono` espera 400 com "provider"); test_fallback_e2e.py:69-83, :119-135, :137-148, :206-239.
- *li* README.md:281-289 e :700-702 descrevem a regra transparente antiga (prefixo -> provider declarado; 400 se não declarado).

## 4. Tarefas (ordem que evita retrabalho)

### T1 — Resolver: último degrau = `replay` por protocolo do chamador
Arquivos: `app/core/resolver.py`, `app/core/official_hosts.py` (docstring + tipo), `tests/core/test_resolver.py`, `tests/test_config_routes.py`, e os call sites `app/core/dispatcher.py:439, :539, :1066`, `app/routers/v1.py:406, :524` (só passar `req.protocol` / `"anthropic"`; nada mais no dispatcher nesta tarefa).

Comportamento: contrato 2.1/2.2. Sequência RED/GREEN (um por vez):
1. `test_last_resort_without_default_is_the_replay_of_the_original_request`: `last_resort("o4-mini", settings_sem_default, "anthropic")` -> `[Candidate(alias=None, provider="anthropic", model="o4-mini", protocol="anthropic", transparent=True, replay=True)]`. RED previsto: `TypeError: last_resort() takes 2 positional arguments but 3 were given`.
2. `test_replay_follows_the_caller_protocol_not_the_model_prefix`: `resolve("claude-sonnet-5", settings_sem_default, "openai").chain[-1]` -> provider/protocol `openai`, replay True; e `resolve("gpt-5", ..., "anthropic")` -> `anthropic`. RED: mesmo TypeError ou provider errado.
3. `test_family_route_tail_is_the_replay_when_no_default` (caso b): settings só `local`, `routes=[("haiku",["qwen"])]`, `resolve("claude-haiku-4-5-20251001", s, "anthropic")` -> `chain[0].alias == "qwen"`, `chain[-1].replay is True`, `model` intacto. RED: hoje `chain[-1]` vem de `_transparent` (replay False).
4. `test_a_name_with_no_deducible_provider_still_gets_the_replay_tail`: `resolve("um-modelo-qualquer", s, "anthropic")` -> rule `"transparent"`, 1 candidato replay. RED: `UnknownProviderError`.
5. `test_default_model_closes_the_chain_and_no_replay_follows_it`: com `default_model="cheap"` a cadeia termina no default e **nenhum** candidato tem `replay=True`. Pode nascer verde — diga.
6. `test_unconfigured_route_alias_still_requires_a_declared_provider`: `routes=[("fable",["opus"])]`, providers **sem** `anthropic`, `resolve("claude-fable-5-1", s, "anthropic")` -> `UnknownProviderError`. RED: hoje `_transparent` cai no `OFFICIAL_HOSTS` (resolver.py:70-79) e devolve candidato.
7. Reescrever/remover: `:112-133` (vira o teste 2/4), `:135-149` (remover: já não levanta), `:152-174` e `:177-204` (fundir nos testes 3/4 com `replay`), `:275-279` e `:320-325` (agora há tail: `last_resort(...) == [replay]`; cadeia `["qwen", None]`). Todos os demais `resolve(x, s)` do arquivo ganham o 3º argumento `"anthropic"`; `tests/test_config_routes.py:23,45,57` idem. `test_family_chain_ends_with_the_transparent_original_when_no_default` (:285-296) e `:266-272` passam a afirmar `replay is True` e provider `== protocolo do chamador`.
Rode: `uv run pytest -q tests/core/test_resolver.py tests/test_config_routes.py tests/core/test_official_hosts.py`. Depois `uv run mypy app` (os call sites do dispatcher/v1 precisam do 3º argumento para compilar).

### T2 — Token: `token_headers` no lugar de `credential_is_token`
Arquivos: `app/core/token_auth.py`, `app/core/dispatcher.py` (só `ShuntRequest` + os 2 usos em :184 e :302 lendo `bool(req.token_headers)` provisoriamente), `app/routers/v1.py:398-404, :538-544`, `tests/core/test_token_auth.py:86-105`, e as 4 construções `credential_is_token=True` em `tests/core/test_dispatcher.py` / `test_dispatcher_stream.py` (viram `token_headers=frozenset({"x-api-key"})`).

RED/GREEN:
1. `test_a_shunt_token_in_x_api_key_alongside_a_real_bearer_names_only_that_header`: headers `{"x-api-key": TEST_SHUNT_TOKEN, "authorization": "Bearer sk-real"}` -> `req.state.token_headers == frozenset({"x-api-key"})`. RED: `AttributeError: 'State' object has no attribute 'token_headers'`.
2. Renomear `:86` e `:97` para afirmar `token_headers == {"x-api-key"}` / `== frozenset()`.
3. `ShuntRequest`: RED `TypeError: __init__() got an unexpected keyword argument 'token_headers'` num teste unitário de `outbound_headers` que já pode ser o de T3.b — ou um teste mínimo de construção.
Rode: `uv run pytest -q tests/core/test_token_auth.py tests/routers/test_v1_auth.py tests/core/test_dispatcher.py -x` (espere falhas em test_v1_auth que só T3/T5 resolvem; cole e siga).

### T3 — Dispatcher: replay, headers, pulo, guarda, query
Arquivo: `app/core/dispatcher.py` (+ `tests/core/test_dispatcher.py`, `tests/core/test_dispatcher_stream.py`). Ao tocar `test_dispatcher_stream.py`, corrija o import I001 da linha 25 (item 12) — `uv run ruff check --fix tests/core/test_dispatcher_stream.py` e cole a saída.

a. `_provider_config` para replay ignora o catálogo. Teste: `GATEWAY` (declara `anthropic`), candidato `replay(..., "anthropic")` -> `base_url == "https://api.anthropic.com"`, `api_key is None`. RED: devolve `https://gateway.exemplo/v1`. Reescrever `:380-395` (candidato replay) e `:398-410` (candidato de rota).
b. `outbound_headers`: (i) replay -> verbatim menos `TRANSPARENT_DROP` menos `token_headers`: headers `{"x-api-key": "tok", "authorization": "Bearer real", "x-shunt-token": "tok"}`, `token_headers={"x-api-key"}` -> saída tem `authorization`, não tem `x-api-key` nem `x-shunt-token`. RED: hoje `bool(token_headers)` -> chave do catálogo -> sem `authorization`. (ii) alias de rota não configurado com provider declarado recebe a chave do catálogo, nunca o header do harness: `GATEWAY` + `routes=[("opus", ["opus-no-gateway"])]` (**não** use alias igual ao requested: `_chain` resolver.py:98-107 o dedupa contra o tail), `dispatch(ShuntRequest("anthropic", body claude-opus-4-5, CLIENT_HEADERS))` -> respx `https://gateway.exemplo/v1/chat/completions` recebe `authorization == "Bearer sk-da-config"` e **não** recebe `sk-do-cliente`/`oauth-da-assinatura`; corpo traduzido (openai) com `max_tokens` do cliente (cobre `_transparent_cap`). RED: hoje verbatim (dispatcher.py:184-189). Este teste **substitui** `:1510`, `:1443` e `:1490`.
c. `_replay_skip_reason`: (i) replay sem credencial encaminhável (headers `{}`; ou só `x-api-key` com `token_headers={"x-api-key"}`) -> string contendo `"no provider credential"` e `"default_model"`; (ii) com `authorization` real -> None; (iii) candidato de rota -> None. Depois `dispatch` com cadeia só replay e headers `{}` em `NO_ANTHROPIC_DECLARED` -> `route.call_count == 0`, 400, `error.type == "invalid_request_error"`, mensagem começa com `"not dispatched: "` e **não** contém `"no candidate answered"`. RED: hoje sem credencial o replay vai ao host (chamada respx registrada) — cole. Reescrever `:443`, `:475`, `:488`, `:503`; **apagar** `:460`, `:525`, `:549` (R2) e substituí-los por: `test_anthropic_caller_asking_gpt5_is_replayed_to_anthropic_with_the_model_intact` (respx `https://api.anthropic.com/v1/messages` 200 `OFFICIAL_ANTHROPIC_OK`; corpo enviado tem `"model": "gpt-5"`; headers verbatim) e o simétrico openai (`claude-sonnet-5` em `chat` -> `https://api.openai.com/v1/chat/completions`, `ok_payload`).
d. Guarda 400 no streaming: `tally.error_type == "invalid_request_error"` e linha de log `status == 400`; mensagem sem "no candidate answered". Reescrever `:1804`, `:1840` (o de gpt-5 vira replay com sucesso SSE, ver e) e manter `:1876`. RED: `"api_error"`.
e. Sucesso no host oficial, streaming (item 2): mover `:429` para `https://api.anthropic.com/v1/messages` com `NO_ANTHROPIC_DECLARED`, bytes crus iguais, headers verbatim, `x-shunt-token` ausente. RED: hoje `TRANSPARENT` declarado captura em api.anthropic.test (rota respx antiga não casa).
f. Lado OpenAI (item 3): três testes com `NO_ANTHROPIC_DECLARED`, chamador `"openai"`, header `{"authorization": "Bearer sk-do-cliente"}`: `chat` (`o4-mini`) -> `https://api.openai.com/v1/chat/completions`; `completions` -> `/v1/completions`; `embeddings` (`text-embedding-3-small`) -> `/v1/embeddings`; corpo com o model intacto, `authorization` verbatim. Podem nascer verdes após a/b/c — registre.
g. Caso (b) no dispatcher (item 1), bufferizado e streaming: `routes=[("opus", ["free"])]` em settings **sem** `anthropic`, respx `https://api.test/v1/chat/completions` -> 400, respx `https://api.anthropic.com/v1/messages` -> 200; resultado 200, `real_model == "claude-opus-4-5"`, trace contém a linha do `free`. Streaming: mesmo cenário com SSE anthropic no oficial (reaproveite o texto de `test_dispatcher_stream.py:467-475`).
h. Query string (item 11): `ShuntRequest(..., query="beta=true&token=abc")` replay -> `str(route.calls[0].request.url) == "https://api.anthropic.com/v1/messages?beta=true"`; candidato de rota com a mesma query -> URL sem query. RED: hoje sem `params`. *assumi*: o padrão `respx.post("https://api.anthropic.com/v1/messages")` casa com query presente — se o RED vier como "não mockado" em vez de asserção de URL, registre e use `respx.post(url__regex=...)` ou `params__contains`.
i. Ajustes de vizinhos que o RED da suíte apontar (rode `uv run pytest -q tests/core` e cole a lista): `:248` (URL -> api.anthropic.com; `host == "api.anthropic.com"`), `:1348` (+ `CLIENT_HEADERS`, URL oficial), `:1780` (idem), `:589` (com chamador openai o replay é openai e `("openai","embeddings")` existe: sem credencial ele é pulado; ajuste o trace esperado para 2 linhas + a linha de replay pulado, mantendo 502), e qualquer teste que afirme trace exato ou `len(trace)`: todo chain sem default ganha uma linha de replay pulado quando o teste manda `{}`.

### T4 — Router: decidir antes do SSE, query, count_tokens
Arquivos: `app/routers/v1.py`, `tests/routers/test_messages.py`, `tests/routers/test_v1_auth.py`.
1. Preflight (item 4): `POST /v1/messages` com `{"stream": true}`, modelo sem rota, só `x-shunt-token`, settings sem default -> **HTTP 400 JSON** (`content-type: application/json`, envelope Anthropic, `"not dispatched"`), sem `text/event-stream`. RED: hoje 200 + `event: error` (v1.py:417-420 entra no stream antes de decidir). Idem `/v1/chat/completions` (envelope OpenAI). Verifique também que a linha de `_record` sai com `rule="transparent"`.
2. Query: teste de router com `c.post("/v1/messages?beta=true", ...)` + `x-api-key` real, respx oficial -> URL recebida termina em `?beta=true`. RED: sem query.
3. `count_tokens` (item 5): (i) só `x-shunt-token`, sem `anthropic` no catálogo -> `route.call_count == 0`, 200 com estimativa local; RED: hoje chama `https://api.anthropic.com/v1/messages/count_tokens` sem auth (v1.py:536-553). (ii) com `x-api-key` real -> encaminha para o oficial, header verbatim, `x-shunt-token` ausente (reescreva `tests/routers/test_messages.py:287-315` já existente para afirmar isso).
4. `tests/routers/test_v1_auth.py`: `:140` e `:160` -> 400 `"not dispatched"`, `route.call_count == 0`, `TEST_SHUNT_TOKEN` ausente de qualquer chamada (mantém a intenção "o token nunca sai"); `:181` -> URL `https://api.anthropic.com/v1/messages`, `x-api-key == "sk-do-cliente"`, `x-shunt-token` ausente; `:199`/`:219` -> 200 com estimativa local e `route.call_count == 0` (renomeie os testes para o que afirmam). Envolva todos em `@respx.mock` (já estão) — o respx sem rota casando levanta, o que é a garantia de "nada saiu para a rede".

### T5 — Testes "my-opus" (item 9) e caso (b) no nível do router
Arquivos: `tests/routers/test_messages.py:107-116`, `tests/routers/test_error_surface.py:82-93, :115-121`, `tests/routers/test_openai_routes.py:109-116`, `tests/routers/test_admin_tokens.py:184-191`.
- Com A, "my-opus" não protege mais nada: **todo** nome ganha replay. Volte os quatro ao modelo original (`claude-opus-4-5`; admin_tokens volta a `claude-sonnet-5`) e apague os comentários "spec R1". Os que mandam `{}`/só `x-shunt-token` (error_surface, openai_routes, admin_tokens) ficam protegidos pelo pulo por falta de credencial; adicione `@respx.mock` ao de admin_tokens se ele não tiver (garante zero rede). O de `test_messages.py:107` manda `x-api-key: sk-do-cliente`: acrescente `respx.post("https://api.anthropic.com/v1/messages").mock(return_value=httpx.Response(400, json={"error": {"message": "sem sorte"}}))` e afirme que a cadeia exaurida inclui o replay (`"claude-opus-4-5"` na mensagem) e o header `x-shunt-model` ausente.
- Novo, caso (b) no router (bufferizado): `SETTINGS` de `test_messages.py` (rota opus -> openrouter) + `x-api-key` real, openrouter 400, oficial 200 -> 200, `x-shunt-model == "claude-opus-4-5"`. RED se rodado antes de T3.g? Não — T3 já entrou; nascerá verde: registre como caracterização.

### T6 — E2E (portão de tarefa: só o que este brief muda; não é campanha)
Arquivo: `tests/e2e/test_official_host_bypass_e2e.py` (+ ajustes em `tests/e2e/test_transparent_e2e.py:86-98`, `tests/e2e/test_fallback_e2e.py`).
1. Caso (b), bufferizado: `settings.routes=[("haiku", ["free"])]`, `del settings.providers["anthropic"]`, `default_model=None`; `provider.queue(Scripted(status=400, json_body=...), Scripted(json_body=OFFICIAL_ANSWER))`; POST `/v1/messages?beta=true` com `x-api-key` real e modelo `claude-haiku-4-5` -> 200; `provider.calls[0].url.netloc == b"fake.openrouter"`, `provider.calls[1].url.netloc == b"api.anthropic.com"`, `.path == "/v1/messages"`, `.url.query == b"beta=true"`; `"x-shunt-token" not in provider.calls[1].headers` (item 10); `provider.bodies[1]["model"] == "claude-haiku-4-5"`. RED previsto antes de T3.h: query ausente (se rodar tudo depois, nasce verde — registre).
2. Caso (b), streaming: mesma rota, `Scripted(status=400)` + `Scripted(sse=[bytes anthropic])`, `stream: true` -> `event: message_start` chega, `provider.calls[1].url.netloc == b"api.anthropic.com"`.
3. Chamador OpenAI: `/v1/chat/completions` com `authorization: Bearer sk-oai`, modelo `o4-mini`, sem rota -> `calls[0].url.netloc == b"api.openai.com"`, path `/v1/chat/completions`, body model `o4-mini`, `authorization` verbatim, `x-shunt-token` ausente.
4. Só token do Shunt (fixture autouse), sem credencial -> 400 `"not dispatched"`, `provider.calls == []` — substitui `test_transparent_e2e.py:86-98` (`modelo-sem-dono`, que hoje espera "provider" na mensagem).
5. Ajuste o teste existente `:30-64` para afirmar também `"x-shunt-token" not in call.headers` e a docstring sem "spec R1".
6. `test_fallback_e2e.py:69-83` e `:137-148`: sem credencial o replay é pulado, logo a 3ª `Scripted` não é consumida — afirme `len(provider.calls)` conforme o novo comportamento e que a mensagem cita o replay pulado; `:119-135` deve seguir verde (cadeia `[replay]` pulada -> 400 com "no tool support" no trace); `:206-239` passa a ir ao oficial (netloc) — atualize docstring/asserção.

### T7 — Documentação
`README.md:281-289` e `:700-702`: regra 4 passa a ser "Replay: no pattern matched and there is no default (or the chain ran out without a default): Shunt re-sends the caller's original request — same model, same body, same headers, same query — to the official host of the caller's protocol (Anthropic for `/v1/messages`, OpenAI for the OpenAI routes). The caller's own credential is what authenticates it; with only a Shunt token and no provider credential the request is refused with 400 saying what is missing." Docstrings: `official_hosts.py:1-4`, `resolver.py:50-56`, `resolver.py:114-125`, `dispatcher.py:184-189`, `:253-261`, `:157-163`, `token_auth.py:147-152`. Sem teste (doc).

## 5. Fora de escopo
- `UpstreamPool` (`app/core/upstream.py`), seed, admin UI, hot reload.
- Plano 2026-09-23 Tasks 2.x/3.x: `/v1/responses`, passagem direta, cache de credencial, destino `chatgpt`, coluna/flag "gateway" (opção B).
- Suíte `tests/browser`, campanha E2E de plataforma (`e2e-test-campaign`), portão de história, mutação.
- Mudanças não relacionadas já na árvore (observabilidade, audit, engine, shutdown, templates): não reverter, não estender.

## 6. Ao final (cole todas as saídas)
```
uv run pytest -q tests/core/test_resolver.py tests/test_config_routes.py tests/core/test_official_hosts.py
uv run pytest -q tests/core/test_token_auth.py tests/core/test_dispatcher.py tests/core/test_dispatcher_stream.py
uv run pytest -q tests/routers tests/e2e
uv run pytest -q tests --ignore=tests/browser
uv run ruff check app tests
uv run mypy app
```
Ponto de partida medido pelo coordenador: 1531 passed / mypy limpo / ruff 1 erro (I001). Reporte os três números finais, os testes apagados (nome + motivo) e os que nasceram verdes.

## 7. Suposições do planejador (verificar no RED)


1. *assumi* que `respx.post(url)` casa uma requisição com query string presente; se não casar, o RED de T3.h aparece como "route not mocked" e o teste precisa de `params__contains`/`url__regex`. O implementador confirma no RED.
2. *assumi* que repassar `?beta=true` ao host oficial é o comportamento correto ("replicar a request original"); não medi a Anthropic aceitando/precisando da query — só li que o Claude Code sempre a envia (`harness-findings.md:26-27`).
3. *assumi* que nenhum teste fora dos listados afirma trace exato/`len(trace)` em cadeia sem default com headers `{}`; a lista real sai do primeiro `uv run pytest -q tests/core` após T1+T3 (T3.i pede para colar).
4. *assumi* que `request.url.query` está disponível no `Request` do FastAPI do mesmo jeito que na `URL` do Starlette que medi (mesma classe, mas não rodei dentro do app).
5. *assumi* que o `FakeProvider` do E2E aceita a query (`/v1/messages?beta=true`) sem 404: as rotas FastAPI ignoram query, mas não rodei.
6. *assumi* que ninguém hoje depende de `resolve(x, settings)` com 2 argumentos fora dos call sites listados (grep em `app` e `tests` cobriu `.py`; não olhei `docs/`).
7. Não consigo verificar a partir daqui qual dos dois headers (`authorization`/`x-api-key`) é o token no stub real do Claude Code (`harness-stub.jsonl:3`, valores mascarados) — motivo para `token_headers` por cabeçalho em vez do bool.
8. Não rodei a suíte nem mypy/ruff nesta sessão (só o `ruff check` e dois `python -c`); os números 1531/limpo/I001 são a medição do coordenador.
