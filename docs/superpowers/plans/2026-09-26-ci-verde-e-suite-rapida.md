# Plano: CI verde e suite mais rapida e mais leve

Status: executado na branch `ci/verde-e-suite-rapida`, CI pendente. Decisoes do usuario (finais): login honra `next=`; argon2 leve so nos testes; pytest-xdist; sqlite em memoria onde der.

Legenda de evidencia: **medi** (rodei e vi a saida), **li** (conferi a fonte, `file:line`), **assumi** (nem um nem outro).

## 1. Objetivo

1. O workflow `tests` do GitHub Actions volta a ficar verde dentro do prazo.
2. A suite fica mais rapida e mais leve:
   - sem sleep real de backoff no e2e;
   - sem argon2 de producao nos testes;
   - sqlite em memoria nos testes de consulta pura;
   - xdist;
   - navegador sem esperas fixas.

Producao so muda no `next=` do login.

## 2. Diagnostico (run do CI em `f6fc748`)

- **Cancelamento.** O job foi cancelado por `timeout-minutes: 30` (`.github/workflows/tests.yml:26`), nao por travamento. *medi, log do run:* job 14:48:20 -> cancelado 15:18:33; o pytest comecou 14:50:52; a primeira linha de progresso levou 23 min; a suite estava em 31%.
- **Mock ausente.** 30 falhas fora do navegador, com `respx AllMockedAssertionError: POST https://api.anthropic.com/v1/messages not mocked!` (last-resort transparente). *medi:* `3aff818` corrige; os 5 arquivos deram `99 passed`.
- **Navegador (57 falhas).**
  - **Redirect fixo.** `app/routers/admin_auth.py:121` redireciona sempre para `/admin/painel`. *li*
  - **Auditoria (43).** `open_audit` (`tests/browser/test_audit_browser.py:135-160`) espera `#rows tr` estando no painel. Alem disso, preenche `confirm`, campo que so existe no primeiro acesso (`app/templates/login.html:39-44`).
  - **Varredura de UI (9).** `abrir` (`tests/browser/test_ui_audit.py:18-25`) nao loga e usa caminhos sem `/admin`.
  - **Painel (5).** `tests/browser/test_dashboard_browser.py:168,275,780,799,827` chamam `/api/stats` sem cookie e recebem 401.
  - Tanto os 43 da auditoria quanto o teste do painel falham desde o commit que os criou: `b195719`.
- **Gate local diferente do CI.** `make check` ignora `tests/browser` (`Makefile:25-26`), e o CI roda tudo. *li*
- **Onde a suite gasta tempo.**
  - **Backoff real no e2e.** *medi, rodada com outro pytest concorrente:* 7 testes de e2e somam ~43 s de 139 s. `tests/e2e/` nao zera o `backoff` (`grep` vazio). O e2e roda em processo (`tests/e2e/conftest.py:65-70`).
  - **argon2 padrao.** *medi:* 75-145 ms por hash+verify no padrao, contra 0,05-0,1 ms com `PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)`. Um hasher leve verifica hash de producao: `cross-verify True`.
  - **sqlite em arquivo.** *medi:* `create_all` custa 104-128 ms em arquivo e 6-7 ms em memoria com `StaticPool`.

## 3. Decisoes de desenho

1. **Branch.** Criar `ci/verde-e-suite-rapida` a partir de `master`, fazer `git cherry-pick 3aff818` e abrir um PR. O workflow so dispara em push para `master`, PR para `master` e `workflow_dispatch` (`tests.yml:8-13`, *li*). **Decidido** (secao 6).
2. **`safe_next(raw) -> str`, em `app/core/auth.py`.** Aceita so caminho relativo de mesma origem sob `/admin`; o fallback e `/admin/painel`.
   - **Rejeita:**
     - valor vazio ou `None`;
     - `\`, caractere de controle ou espaco;
     - qualquer `scheme` ou `netloc`;
     - valor que nao comeca com `/`, ou que comeca com `//`;
     - valor que, apos `unquote`, comeca com `//` ou `/\`;
     - path fora de `/admin`;
     - `/admin/login`, para evitar laco.
   - **Fluxo:**
     - `login_redirect` passa a gerar `/admin/login?next=<quote(path?query)>`;
     - o GET do login repassa o `next` num campo oculto;
     - o POST aceita `next` e o repassa nos re-renders de erro;
     - o primeiro acesso (create) tambem honra o `next`.
3. **argon2 leve so nos testes, sem knob de producao.**
   - Uma fixture autouse em `tests/conftest.py` troca `app.core.security._hasher`. Isso funciona porque `hash_password` e `verify_password` leem `_hasher` na hora da chamada (`security.py:24,35`, *li*).
   - O marker `real_argon2` desliga a troca para o teste-guarda.
   - Os servidores em subprocesso (navegador) continuam com o hasher de producao: so um teste faz login real.
4. **sqlite em memoria so nos 7 arquivos de consulta pura:** `tests/stats/test_queries.py`, `test_search.py`, `test_dossier.py`, `test_analysis.py`, `test_models.py`, `test_user_models.py` e `test_bodies.py`.
   Os outros mantem arquivo, e o motivo fica num comentario em `tests/stats/conftest.py`:
   - **`Recorder` em thread** (`asyncio.to_thread`, `app/stats/recorder.py:182,190`): `test_recorder.py`, `test_wiring.py`, `test_body_wiring.py`, `test_*_api.py` e `tests/routers/test_admin_*.py`;
   - **boot via URL** (`build_engine` sem `StaticPool`): `tests/test_main.py`, `tests/config/test_boot.py`, `tests/config/test_seed.py`, `tests/core/test_config_watcher.py`, `tests/core/test_token_auth.py` e `tests/routers/test_v1_auth.py`;
   - **reabertura e migracao:** `tests/stats/test_engine.py`;
   - **subprocesso:** `tests/e2e/test_analysis_e2e.py` e `tests/browser/*`.
5. **xdist com `--dist loadfile`.** Assim os fixtures `scope="module"` do navegador sobem uma vez por arquivo.

## 4. Tarefas

**Regras que valem para toda tarefa:**
- **Portao:** de TAREFA, com RED real antes do GREEN e so os testes focados e os vizinhos rodando. Colar comando e saida em cada RED e em cada GREEN. So o T9 cobra portao de historia.
- **Revisao e fechamento:** `strict-code-reviewer` (portao de tarefa) e depois `mutation-sweep` antes de DONE.
  - **Nota (2026-09-26):** decisao do usuario move mutacao para so-quando-pedido; o `mutation-sweep` desta regra nao rodou.
- **Commits:**
  - `git add` explicito, nunca `-A`;
  - nunca incluir `app/core/official_hosts.py`, `tests/core/test_official_hosts.py` ou `stats.db.lock`;
  - nenhuma assinatura do Claude.

### T0 - Baseline e branch (sem commit de codigo)

- **Maquina ociosa:** `pgrep -af pytest` sem saida.
- **Medir:**
  - `uv run pytest -q tests --ignore=tests/browser --durations=25`;
  - `uv run pytest -q tests/browser -rs --maxfail=100` (headless; confere 43/9/5);
  - `uv run ruff check app tests` e `uv run mypy app`.
- **Criar o branch** conforme a decisao 1.

### T1 - O login honra `next=`

- **Arquivos:**
  - `app/core/auth.py:74-76` e a nova `safe_next`;
  - `app/routers/admin_auth.py:43-64` (GET) e `:67-123` (POST; o `:121` deixa de ser fixo);
  - `app/templates/login.html:25-49`;
  - `tests/core/test_auth.py` (novo; nao existe hoje, *medi*);
  - `tests/routers/test_admin_auth.py`.
- **REDs:**
  1. `safe_next` aceita `/admin/requests?q=1`, `/admin/painel` e `/admin`. Hoje falha com `ImportError`.
  2. `safe_next` devolve `/admin/painel` para: `None`, `""`, `//evil.com`, `https://evil.com`, `http:evil.com`, `/\evil.com`, `\\evil.com`, `javascript:alert(1)`, `/%2F%2Fevil.com`, `%2F%2Fevil.com`, `/%5Cevil.com`, `/api/stats`, `/docs`, `/`, `/admin/login`, `/admin/login?next=/admin`, `/admin\n`, ` /admin`.
  3. `GET /admin/requests?q=x` sem cookie -> 303 para `/admin/login?next=...`, com a codificacao fixada no teste.
  4. `GET /api/stats` sem cookie -> 401, sem `location`.
  5. `GET /admin/login?next=/admin/requests` -> o HTML traz o campo oculto com esse valor. Um valor `"><script>` sai escapado ou e descartado.
  6. Login valido com `next=/admin/requests?q=1` -> 303 para esse caminho. Cada vetor do RED 2 -> 303 para `/admin/painel`.
  7. Primeiro acesso com `next` -> 303 para o `next`.
  8. Senha errada com `next` -> 401, e o campo oculto e mantido.
- **Asserts existentes que continuam validos:** `"/admin/login" in location` (`test_admin_session.py:55,88,99` etc.) e `== "/admin/painel"` (`test_admin_auth.py:107,163`, `test_admin_users.py:134`, `test_admin_dashboard.py:98`).
- **Rodar:** `uv run pytest -q tests/core/test_auth.py tests/routers/test_admin_*.py tests/stats/test_dashboard_page.py`.
- **Risco:** open redirect. A mutacao precisa matar cada ramo de `safe_next`.
  - **Nota (2026-09-26):** decisao do usuario move mutacao para so-quando-pedido; essa mutacao nao rodou.

### T2 - Os testes de navegador voltam a passar

- **Auditoria** (`tests/browser/test_audit_browser.py`):
  - o `server` passa a receber um `ADMIN_SESSION_SECRET` fixo;
  - `open_audit` injeta o cookie `issue_jwt(1, SECRET, 3600)` e vai direto a `/admin/requests{query}`.
- **Teste novo** `test_login_leads_to_the_requested_page` (depende do T1): semeia um usuario, abre `/admin/requests?<filtro>` sem cookie, faz o login e espera a mesma URL e `#rows tr`.
- **Varredura de UI** (`tests/browser/test_ui_audit.py`): `abrir` injeta o cookie e usa `/admin/painel` e `/admin/requests`.
- **Painel** (`tests/browser/test_dashboard_browser.py`): as 5 chamadas `httpx.get` ganham o cookie, num helper do modulo.
- **RED e GREEN:**
  - **RED:** as contagens do T0 nos tres arquivos.
  - **GREEN:** `uv run pytest -q tests/browser -rs` (headless) sem falhas. Colar a linha final e o tempo. Nunca `--headed`.
- **Risco:** a varredura de UI pode achar defeitos reais nas telas `/admin`. Nesse caso, nao relaxar o teste: abrir uma tarefa de produto e perguntar ao usuario.

### T3 - O e2e nao dorme o backoff real

- **Arquivos:** `tests/e2e/conftest.py`, com a fixture autouse `no_real_backoff` (`monkeypatch.setattr(dispatcher, "backoff", lambda attempt: 0.0)`; o dispatcher importa `backoff` em `app/core/dispatcher.py:47`, *li*).
- **RED:** um sentinela em `tests/e2e/test_fallback_e2e.py`. Ele espiona `dispatcher.asyncio.sleep`, roda o cenario 503 -> 200 e afirma `sum(waits) == 0`. Hoje o espiao registra >= 1,0.
- **Medida:** `uv run pytest -q tests/e2e --durations=10`, antes e depois.
- **Risco:** um e2e futuro de deadline x backoff precisa desligar a fixture. Documentar isso na docstring.

### T4 - argon2 leve so nos testes, com guarda da producao

- **Arquivos:** `tests/conftest.py` (fixture autouse `light_argon2` e marker `real_argon2` registrado em `pytest_configure`) e `tests/core/test_security.py`. `app/core/security.py` NAO muda.
- **REDs:**
  1. **Guarda** (`real_argon2`): `argon2.extract_parameters(hash_password("x"))` bate com um `argon2.PasswordHasher()` novo. A comparacao e com o default da biblioteca, nao com numeros fixos. Provar que a guarda pega, trocando `_hasher` a mao, sem commit.
  2. Sem marker, `memory_cost == 8`. Hoje da `65536`.
  3. Um hash de producao verifica com a fixture leve ativa.
- **Antes:** `grep -rn "65536\|m=" tests`, para achar teste que dependa do formato do hash.
- **Medida:** `--durations=10` nos arquivos de login e seguranca, antes e depois.

### T5 - sqlite em memoria nos testes de consulta pura

- **Arquivos:**
  - `tests/stats/conftest.py:14-27`: `factory(url=None, ...)`; sem `url`, usar `create_engine("sqlite+pysqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})`. Acrescentar o comentario com quem mantem arquivo e por que.
  - Os 7 arquivos da decisao 4 passam a usar `make_engine()`.
- **RED:** `tests/stats/test_make_engine.py` (novo). `make_engine()` devolve um engine com `StaticPool`, as tabelas existem e duas `Session` enxergam a mesma linha. Hoje falha com `TypeError`.
- **Medida:** `uv run pytest -q tests/stats --durations=15`, antes e depois. A estimativa de ~12 s e *assumi*.

### T6 - pytest-xdist

- **Antes do `uv add --dev pytest-xdist`:** skill `dependency-bumps`. Sem pin; conferir PyPI/pytest-dev, typosquat e install script.
- **`Makefile`:**
  - `test`, `e2e` e `check` ganham `-n auto --dist loadfile`;
  - `browser` ganha `-n 2 --dist loadfile`, se a medida justificar.
- **Medidas obrigatorias:**
  1. Serial contra `-n auto --dist loadfile`, tempo de parede.
  2. `-n auto` 3 vezes seguidas e `--dist load` 1 vez. Qualquer falha que nao aparece em serial vai para `superpowers:systematic-debugging`; nunca skip.
  3. `git status --short` limpo depois das rodadas.
  4. A cobertura agregada aparece no `make check`.

### T7 - CI em dois jobs

- **Arquivo:** `.github/workflows/tests.yml`.
- **Job `unit`:**
  - sem Chromium;
  - `uv run pytest -q -rs -n auto --dist loadfile tests --ignore=tests/browser`;
  - `timeout-minutes` = 2x o tempo medido, com minimo de 10.
- **Job `browser`:**
  - instala o Chromium e roda `uv run pytest -q -rs --maxfail=5 tests/browser`;
  - `timeout-minutes` = 2x o tempo medido.
- **Comentario do topo:** passa a dizer que `unit` e `browser` sao os checks obrigatorios e que o `tests` antigo sai da lista.
- **Prova:** o PR roda os dois jobs verdes. Colar as duracoes.
- **Risco:** se `tests` estiver como obrigatorio na branch protection, o PR fica esperando para sempre. Quem muda e o usuario.

### T8 - Navegador sem esperas fixas (medida primeiro)

- **Medir antes:**
  - `uv run pytest -q tests/browser --durations=30`;
  - a soma dos `wait_for_timeout`: 5 900 ms em `test_dashboard_browser.py:322,337,339,354,361,383,441,658,678,727,748,768`, mais `test_ui_audit.py:24,47`.
- **Mudancas:**
  - cada `wait_for_timeout` vira uma espera da condicao que o assert seguinte verifica. O que prova AUSENCIA de mudanca fica, com comentario;
  - `page.set_default_timeout(5_000)` nos helpers;
  - `wait_for_function` cai de 10 000 para 5 000 so onde a medida mostrar folga.
- **Prova:** `--durations` antes e depois, e 3 rodadas verdes seguidas.

### T9 - Portao de historia

- **Skill `story-gate`:**
  - suite com `-n auto`;
  - `tests/browser` headless;
  - `make lint`, `make type` e `make check`;
  - mutacao sobre `safe_next` e sobre a fixture de argon2.
    - **Nota (2026-09-26):** decisao do usuario move mutacao para so-quando-pedido; essa mutacao nao rodou.
- **Tabela antes/depois** (T0 contra o final): parede serial e xdist, e2e, stats, navegador e os dois jobs de CI.
- **Documentacao:** README e `CLAUDE.md` do projeto passam a citar o `-n auto` e o `next=` no login.

## 5. Assumido (nao verificado)

1. O `:memory:` via `build_engine(url)` sem `StaticPool` vira um banco por conexao.
2. `StaticPool` com a thread do `Recorder` e arriscado. Seria verificado rodando esses arquivos 20x em memoria.
3. Os ganhos do T5 e do xdist.
4. ~~O `pytest-cov` agrega a cobertura com o xdist~~ -- **medido**: `TOTAL 4292 66 98%` em serial (1606 passed, 49.91s) e igual `TOTAL 4292 66 98%` com `-n auto --dist loadfile` (1606 passed, 23.69s). Falta so a parte do `pytest-xdist` nao ter install script.
5. A quantidade de vCPUs do runner e o estado da branch protection.
6. A origem do `stats.db.lock`: nao e o codigo de `app/`, e o `.gitignore` nao o cobre.

## 6. Decisoes do usuario (2026-09-26, finais)

1. **Branch:** sempre a partir de `master`. Criar `ci/verde-e-suite-rapida` de `master`, fazer `git cherry-pick 3aff818` e abrir o PR para `master`. Nao usar `feat/unknown-route-passthrough`.
2. **`next=`:** somente caminhos sob `/admin` (`/admin` ou `/admin/...`, exceto `/admin/login`). `/docs`, `/` e qualquer outro caminho caem no fallback `/admin/painel`. O RED 2 do T1 inclui `/docs` e `/` na lista de rejeitados.
3. **Branch protection:** o usuario troca o check obrigatorio `tests` por `unit` + `browser`. O agente nao altera essa configuracao; no T7, avisar o usuario no momento em que o PR abrir, para que ele faca a troca.

## 7. Fora de escopo

- Os parametros de producao do argon2, ou um knob de ambiente para eles.
- Outros plugins de pytest.
- Os `_make_engine` dos routers sem `dispose`.
- Corrigir os defeitos visuais que a varredura de UI achar.
- As mudancas nao commitadas em `app/core/official_hosts.py`.
