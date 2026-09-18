# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Shunt is a FastAPI proxy that exposes both the OpenAI API surface (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`) and the Anthropic Messages surface (`/v1/messages`). Its purpose is to let an Anthropic-protocol client (Claude Code) talk to an OpenAI-compatible provider: the request is translated Anthropic -> OpenAI, forwarded upstream with the caller's own API key, and the response translated OpenAI -> Anthropic on the way back.

## Commands

```bash
make dev                  # uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
uv sync                   # install/refresh deps from uv.lock
uv add <package>          # add a dependency (no pinned version)
uv lock --upgrade         # upgrade all deps to latest
```

Python 3.14 (`.python-version`). No test suite, linter, or type checker is configured yet — when adding one, add the corresponding `make` target.

## Architecture

- `app/main.py` — FastAPI app; mounts `app/routers/v1.py` under prefix `/v1`.
- `app/routers/v1.py` — all endpoints. The OpenAI-shaped routes currently return hardcoded stub payloads; only `/v1/messages` does real upstream work.
- `app/schemas/schemas.py` — Pydantic request models for both protocols (`ChatCompletionRequest`, `CompletionRequest`, `EmbeddingRequest`, `AnthropicRequest`).
- `app/tools/conversors.py` — the protocol bridge, and the piece to touch when Anthropic/OpenAI shapes diverge:
  - `transform_anthropic_to_openai`: hoists the root-level `system` field into a leading `{"role": "system"}` message, and flattens Anthropic content-block lists into a single string by concatenating `type == "text"` blocks. Non-text blocks (images, `tool_use`, `tool_result`) are silently dropped.
  - `transform_openai_to_anthropic`: rewrites the `chatcmpl` id prefix to `msg`, maps `finish_reason` -> `stop_reason` (`stop`/`length`/`tool_calls` -> `end_turn`/`max_tokens`/`tool_use`), and renames `prompt_tokens`/`completion_tokens` -> `input_tokens`/`output_tokens`.
- `app/config/config.py` — `model_sources`, a plain list of `{provider, model, api_key}` dicts backing `GET /v1/models`. `pydantic-settings` and `dotenv` are dependencies but not yet wired up; config should migrate there rather than growing this literal.

Auth on `/v1/messages` is pass-through: the key arrives via `x-api-key` or `Authorization: Bearer`, and is forwarded upstream unchanged. The proxy holds no credentials of its own.

## Known gaps

- `TARGET_PROVIDER_URL` is referenced in `create_anthropic_message` but never defined or imported — `/v1/messages` raises `NameError` at runtime. Define it (from config/env) before testing that route.
- The package directories have no `__init__.py`; imports work as namespace packages, so run from the repo root.
- Streaming is unimplemented: `AnthropicRequest.stream` is accepted and forwarded upstream, but the response path only handles non-streaming JSON. `/v1/chat/completions` returns a fake two-chunk SSE stream.

## Conventions

Docstrings and inline comments are written in Portuguese; keep that when editing existing modules.
