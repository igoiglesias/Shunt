# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Shunt is a local FastAPI proxy between a coding harness (Claude Code above all) and LLM providers. It exposes both the Anthropic surface (`/v1/messages`, `count_tokens`) and the OpenAI surface (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`), translates between the two protocols in both directions, and routes each request through a configurable chain of provider/model candidates with retry and fallback. It also ships an admin area (login, users, API tokens, catalog config) and a usage panel backed by a database. `README.md` is the user-facing reference for configuration, routing, translation and the panel.

## Commands

```bash
make dev        # uvicorn with --reload on port 8000
make test       # uv run pytest -q -n auto --dist loadfile tests --ignore=tests/e2e   (includes tests/browser)
make e2e        # uv run pytest -q -n auto --dist loadfile tests/e2e   (app end to end against tests/e2e/fake_provider.py)
make browser    # uv run pytest -q -n 2 --dist loadfile tests/browser   (Playwright, headless; ~1 min, kept out of `check`)
make lint       # uv run ruff check app tests
make type       # uv run mypy app
make check      # lint + type + pytest -n auto --dist loadfile --cov=app --cov-report=term-missing --ignore=tests/browser
make prod       # no reload, one worker per core, --timeout-graceful-shutdown 5

uv run pytest -q tests/routers/test_messages.py                  # one file
uv run pytest -q tests/routers/test_messages.py::test_name       # one test
uv sync / uv add <pkg> / uv lock --upgrade                       # deps, never pinned
```

Python 3.14. `pytest` runs with `asyncio_mode = "auto"`; HTTP to providers is mocked with `respx`. The make targets use `pytest-xdist` with `--dist loadfile` (one file per worker), so tests must not share global state or fixed paths across files. No coverage threshold is configured.

## Architecture

- `app/main.py` — app plus `lifespan` boot: logging, the stats `Recorder`, settings loaded from the database (seeding the catalog on first boot), the `UpstreamPool` (shared httpx client), and `admin_session_secret` (`ADMIN_SESSION_SECRET` or random per boot). Registers `v1`, the admin routers, and `dashboard`/`audit` behind `require_admin`.
- **Persistence is optional.** `app/stats/engine.py` builds a SQLAlchemy engine on libsql/Turso from `TURSO_DATABASE_URL` (+ `TURSO_AUTH_TOKEN`); schema comes from `Base.metadata.create_all` (no migration tool). Without the URL the proxy still boots with an empty catalog, and anything that needs the DB (Shunt tokens, admin, history) answers 503 instead of crashing.
- **Catalog lives in the DB.** `app/config/seed.py::CATALOG` is inserted only when the DB is empty (`seed_catalog_if_empty`); after that `app/config/settings.py::load_settings_from_db` is the source of truth (providers with raw `api_key` column, models with `ModelCaps`, routes with ordered candidates). Edit it through the admin config screens, not by growing `CATALOG`.
- **Runtime knobs** are `SHUNT_*` env vars read with `os.environ` in `app/config/config.py` (timeouts, retries, deadlines, queue sizes, panel limits), plus `SHUNT_STORE_BODIES` (`app/stats/bodies.py`) and `SHUNT_LOG_LEVEL` (`app/core/observability.py`).

### Request path (`/v1/*`)

1. `app/routers/v1.py` — `detect_protocol` picks the caller's dialect from headers; the body is parsed and validated by `REQUEST_SCHEMAS[(protocol, endpoint)]` (`app/schemas/`); malformed or non-object JSON becomes a 400 in the caller's error envelope (`error_body`).
2. Auth: an `x-shunt-token` header is looked up (sha256) in the `ApiToken` table; when valid, the proxy injects the provider key from the catalog. Without it, the caller's own credentials pass through for transparent providers (`outbound_headers` in `app/core/dispatcher.py`).
3. `app/core/resolver.py::resolve` turns the requested model (alias, prefix hint, transparent fallback) into a candidate chain. It runs **before** choosing between JSON and SSE, so an unknown model is still a 400 rather than an error after the stream header is sent.
4. `app/core/dispatcher.py` (`dispatch` / `dispatch_stream`, with `attempt.py`, `capabilities.py`, `upstream.py`) filters candidates by capability, picks the upstream path from (provider protocol, logical endpoint), retries or falls back, and records the result.
5. `app/translate/` holds the protocol bridge: request translation (`to_openai.py`, `to_anthropic_request.py`), response translation (`to_anthropic.py`), SSE translation (`sse_parse.py`, `sse_to_anthropic.py`, `sse_to_openai.py`), plus `usage.py` and `ids.py` (tool-call ids survive the round trip). Golden fixtures for the tool loop live in `tests/golden/`.

Streaming exists only on `/v1/messages` and `/v1/chat/completions`; `/v1/completions` and `/v1/embeddings` always return one JSON document.

### Admin and panel

- Admin session is a JWT in the `shunt_admin` cookie (`app/core/security.py`, `app/core/auth.py::require_admin`). `is_api_request` decides between a login redirect (HTML) and a 401 (API/HTMX). The redirect carries `?next=`; `safe_next` honours only same-origin paths under `/admin` (not `/admin/login`) and falls back to `/admin/painel`.
- UI is server-rendered Jinja2 in `app/templates/` with HTMX partials (`_*.html` are fragments swapped into full pages).
- `app/stats/` holds the ORM models, the async `Recorder`, and the queries/analysis/dossier code behind the usage panel.

## Tests

`tests/` mirrors `app/` (`core/`, `config/`, `routers/`, `stats/`, `translate/`), with shared fixtures in `tests/conftest.py`. `tests/e2e/` runs the whole app against a scripted fake provider; `tests/browser/` drives the panel with headless Playwright. Never run the browser suite headed.

## Workflow

Work in this repo goes through the available agents and skills, never done by hand in the main thread: `codebase-explorer` before reading unfamiliar code, `superpowers:brainstorming` before a behavior change, an implementer subagent with TDD, `strict-code-reviewer` before calling anything done, `frontend-validator` + `visual-qa` for any screen (including `/docs`), `mutation-sweep` before DONE, and `service-resilience` / `metrics-and-data-integrity` when touching the dispatcher, retries, timeouts or panel numbers. Audits, QA and root-cause investigations count as work too. Subagent reports are claims until verified (`verify-subagent-claims`).

## Conventions

- Docstrings and comments are in Portuguese (no accents in code comments); keep that when editing. Module docstrings record *why* a decision was made, often with what was measured — read them before changing behavior.
- Plans and specs: `docs/superpowers/plans/`, `docs/superpowers/specs/`. Per-task execution history (briefs, reviews, diffs, progress ledger): `.superpowers/sdd/`.
