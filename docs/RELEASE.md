# Release Runbook

Este documento descreve o processo de release do Shunt. É a fonte de verdade operacional; o README cobre uso, este cobre o *como* fazer a release.

---

## 1. Versão

- Fonte única: `pyproject.toml [project] version`.
- Esquema: **SemVer** `MAJOR.MINOR.PATCH`. Pré-release usa `-rc.N` (ex.: `0.2.0-rc.1`).
- Não existe `__version__` em `app/__init__.py` — `_version()` em `app/main.py:151-166` lê o pyproject e alimenta `/openapi.json` `info.version`. `tests/test_openapi.py:41-46` testa o contrato.

---

## 2. O que é releaseado (e o que não)

| Item | Status | Detalhes |
|---|---|---|
| Código Python | ✅ | Tag `vX.Y.Z` no GitHub; `make prod` roda do checkout |
| `CHANGELOG.md` | ✅ | Seção da versão copiada para GitHub Release Notes |
| `docs/RELEASE.md` | ✅ | Este arquivo |
| `README.md` | ✅ | Seção Upgrading + links de release |
| PyPI | ❌ | Sem `[build-system]` em `pyproject.toml`; `_version()` trata `PackageNotFoundError` |
| Docker | ❌ | Não existe `Dockerfile` |
| Catálogo do operador | ❌ | Vive no banco, rollbackável pela tela (`config_versions`) |

Se um dia houver necessidade de PyPI/Docker, a decisão é separada e exige sua própria campanha de teste.

---

## 3. Passos da release

### Pré-requisitos
- Branch protection ativa em `master`: `unit` + `browser` obrigatórios (ver `GitHub Settings → Rulesets`).
- Branch default do repositório é `master` (confirmar em `Settings → General`).

### Fluxo
1. **Feature branch → PR para `master`** com CI verde (`unit` + `browser`).
2. **Merge no `master`** (merge commit, sem squash — preserva histórico).
3. **`make pre-release`** em checkout limpo de `master`:
   - `make check` (lint + type + suite + coverage)
   - `make e2e` (app contra `fake_provider.py`)
   - `make browser` (headless Playwright — fora do `check` por tempo)
   - Se qualquer um falhar: corrigir em nova branch, PR, repetir.
4. **`make release VERSION=x.y.z`** (ver Makefile):
   - Recusa se árvore não limpa (`git status --porcelain` vazio).
   - Recusa se não estiver em `master`.
   - Recusa se tag `vX.Y.Z` já existe.
   - Verifica que `CHANGELOG.md` tem seção `## [x.y.z] - YYYY-MM-DD`.
   - Bumpa `pyproject.toml` para `x.y.z` (se diferente), commita `chore: bump version to x.y.z`.
   - Cria tag anotada `vX.Y.Z` com mensagem do changelog.
   - **Não faz push automático** — a publicação é decisão humana.
5. Revisar o tag local (`git show vX.Y.Z`).
6. **`git push origin master --tags`**.
7. **GitHub Release**: criar Release na tag, colar a seção do `CHANGELOG.md` como Release Notes, marcar "This is a pre-release" se for `-rc`.

---

## 4. Migração de schema (banco)

- O Shunt usa `Base.metadata.create_all(engine)` no boot (`app/stats/engine.py:150`) + `add_missing_columns(engine)` (`:151`, implementação em `:106-132`).
- **Regra**: toda mudança de schema em release é **só `ADD COLUMN` nullable, sem default**.
  - Renomear coluna, mudar tipo, dropar coluna, ou adicionar coluna NOT NULL/DEFAULT exige um **passo de migração escrito** (arquivo `migrations/YYYYMMDD_descricao.py` ou equivalente) e a release é **bloqueada** até ele existir e ser testado.
  - Exemplo: v0.1.0 adiciona `kind` nullable sem default — coberto por `add_missing_columns`.
- Testes de migração: `tests/stats/test_engine.py` deve ter um teste que sobe o código novo contra um banco com schema da versão anterior (sem a coluna nova) e afirma que a coluna é acrescentada e a leitura segue.

---

## 5. Hotfix

- Sempre a partir de `master` (regra do projeto: branch novo sempre a partir de master; commit de outra branch via cherry-pick).
- Fluxo: `git checkout master && git checkout -b hotfix/x && fix + teste → PR → merge → make release VERSION=x.y.Z+1`.
- **Sem release branches** longevas: um mantenedor, uma versão suportada. Branch de release só se um dia houver duas versões em campo simultaneamente.
- Catálogo é hot-reloaded entre workers (`SHUNT_CONFIG_POLL_SECONDS`, `app/config/config.py:81`) e rollbackável pela tela — hotfix de catálogo nem precisa de release.

---

## 6. Verificações manuais (fase opcional, fora da CI)

Antes de taggear uma release que muda schema ou comportamento de relay:

1. **Boot do zero em banco vazio**: novo worktree + `sqlite` novinho, confirmar seed (o `seed_catalog_if_empty` só age com banco vazio, `app/main.py:119`) e que a primeira tela vira sign-up.
2. **Upgrade do banco real**: **cópia** do `stats.db` (nunca o original) com a versão anterior, bootar a nova versão, confirmar no log a linha "stats: colunas acrescentadas" e que painel/auditoria leem.
3. **Smoke contra provedor real**: se a release toca relay/fallback, testar contra a chave do operador (não automatizado, não em CI — upstream pago é fase opcional com chave do usuário).

---

## 7. Checklist de pre-release

- [ ] `make pre-release` verde (output colado no PR)
- [ ] `CHANGELOG.md` tem `## [x.y.z] - YYYY-MM-DD` com Added/Changed/Fixed/Security
- [ ] Árvore limpa (`git status --porcelain` vazio)
- [ ] Em `master`
- [ ] Tag `vx.y.z` não existe
- [ ] Testes de migração em `test_engine.py` cobrem a(s) coluna(s) nova(s)
- [ ] Seção "Upgrading" no README atualizada (se houver mudança visível ao operador)

---

## 8. Rollback de release

Se uma release taggeada quebrar em produção:
1. `git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z` (apaga tag local e remota).
2. `git revert <merge-commit>` no `master` (gera novo commit revertendo o merge).
3. `make release VERSION=x.y.Z-1` (re-tag da versão anterior se necessário).
4. Investigar e fixar em nova branch → PR → nova release.

O catálogo do operador não é afetado — vive no banco, rollbackável pela tela.

---

## 9. Referências rápidas

| Arquivo | O que contém |
|---|---|
| `pyproject.toml:3` | `version = "X.Y.Z"` |
| `app/main.py:151-166` | `_version()` — lê pyproject, fallback `0.0.0` |
| `app/main.py:258` | `FastAPI(version=_version())` → `/openapi.json` |
| `tests/test_openapi.py:41-46` | Testa `spec["info"]["version"] == pyproject version` |
| `app/stats/engine.py:106-132` | `add_missing_columns` — `ADD COLUMN` nullable |
| `app/stats/engine.py:150-151` | `create_all` + `add_missing_columns` no boot |
| `app/config/config.py:81` | `SHUNT_CONFIG_POLL_SECONDS` — hot reload |
| `README.md:243-244` | Config rollback (`config_versions`) |
| `Makefile` | Alvos `check`, `e2e`, `browser`, `pre-release`, `release` |