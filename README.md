# Shunt

Shunt is a local proxy that lets a client speaking the **Anthropic Messages
protocol**, Claude Code above all, talk to a provider that speaks the
**OpenAI Chat Completions protocol**: llama.cpp on your own machine, Ollama,
OpenRouter, or anything else with an OpenAI-compatible endpoint.

Claude Code sends its request to Shunt. Shunt translates it, forwards it to
whichever model you configured, translates the answer back, and Claude Code
never learns that the model on the other end was not Claude.

It works in both directions. A client speaking OpenAI can reach a provider
speaking Anthropic through the same proxy.

---

## Table of contents

- [Quick start](#quick-start)
- [Pointing Claude Code at it](#pointing-claude-code-at-it)
- [How a request travels](#how-a-request-travels)
- [Configuration](#configuration)
- [How a model is chosen](#how-a-model-is-chosen)
- [What gets translated](#what-gets-translated)
- [Fallback, retries and timeouts](#fallback-retries-and-timeouts)
- [Endpoints](#endpoints)
- [The usage panel](#the-usage-panel)
- [The request screen](#the-request-screen)
- [Development](#development)
- [What Shunt does not do](#what-shunt-does-not-do)

---

## Quick start

You need Python 3.14 and [uv](https://docs.astral.sh/uv/), plus a provider to
talk to.

**1. Install the dependencies.**

```bash
uv sync
```

**2. Write your credentials.**

```bash
cp .env.sample .env
```

Then edit `.env`. It holds one line per provider, and the variable names must
match the `api_key_env` values in `app/config/config.py`:

```bash
LOCAL_API_KEY=local                  # same value you passed to llama-server --api-key
OPENROUTER_API_KEY=sk-or-v1-...
```

`.env` is in `.gitignore`. No credential belongs in any file that git tracks.

**3. Describe your provider** in `app/config/config.py`. The shipped catalogue
points at a llama.cpp server on `127.0.0.1:8181` with OpenRouter behind it as a
fallback. Change it to yours. The [Configuration](#configuration) section
explains every field.

**4. Start the server.**

```bash
make dev
```

That runs uvicorn on port 8000 with reload enabled.

**5. Check that it is up.**

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}

curl -H 'anthropic-version: 2023-06-01' http://127.0.0.1:8000/v1/models
# {"data":[{"type":"model","id":"qwen-local", ...}],"has_more":false,"first_id":"qwen-local"}
```

The health check deliberately touches neither your configuration nor your
provider, so it answers even when those are broken. That is exactly the
moment you need it to. It lives at `/health`; the home page is the usage
panel.

**6. Send a real message.**

```bash
curl -X POST http://127.0.0.1:8000/v1/messages \
  -H 'content-type: application/json' \
  -H 'x-api-key: whatever' \
  -d '{"model":"claude-haiku-4-5","max_tokens":200,
       "messages":[{"role":"user","content":"Reply with one word: capital of France?"}]}'
```

The answer comes back in Anthropic shape, and the `x-shunt-model` response
header names the model that actually ran.

---

## Pointing Claude Code at it

Three environment variables, and Claude Code talks to your model instead of
Anthropic's:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8000 \
ANTHROPIC_AUTH_TOKEN=local \
ANTHROPIC_MODEL=claude-sonnet-4-5 \
claude
```

- `ANTHROPIC_BASE_URL` sends Claude Code's traffic to Shunt.
- `ANTHROPIC_AUTH_TOKEN` is the credential Claude Code sends. Shunt does not
  check it: for a configured model it substitutes your own provider key, and
  for a transparent one it forwards this token untouched.
- `ANTHROPIC_MODEL` names the model Claude Code asks for. Shunt matches it
  against your routes, so `claude-sonnet-4-5` is a routing key, not a promise
  that Sonnet answers.

Unset `ANTHROPIC_API_KEY` first if you have one in your environment, or it takes
precedence and Claude Code warns you about it.

---

## How a request travels

```text
Claude Code                Shunt                         your provider
     |                       |                                  |
     |  POST /v1/messages    |                                  |
     |  (Anthropic shape)    |                                  |
     |---------------------->|                                  |
     |                       | 1. detect the caller's dialect    |
     |                       | 2. validate the body              |
     |                       | 3. resolve the model to a chain   |
     |                       | 4. drop candidates that cannot    |
     |                       |    do what the request needs      |
     |                       | 5. translate Anthropic -> OpenAI  |
     |                       | 6. swap in the configured key     |
     |                       |--------------------------------->|
     |                       |     POST /chat/completions        |
     |                       |<---------------------------------|
     |                       | 7. translate OpenAI -> Anthropic  |
     |<----------------------|                                  |
     |  Anthropic shape      |                                  |
     |  + x-shunt-model      |                                  |
```

If step 6 fails in a way worth retrying, Shunt retries the same candidate. If it
fails in a way that will not improve, Shunt moves to the next candidate in the
chain. Streaming follows the same path, one SSE event at a time.

---

## Configuration

Everything lives in `app/config/config.py`, the only file that changes from
machine to machine. Four values, each a plain Python literal.

### `providers`: where requests can go

```python
providers = {
    "local": {
        "base_url": "http://127.0.0.1:8181/v1",
        "protocol": "openai",
        "api_key_env": "LOCAL_API_KEY",
    },
}
```

| Field | Meaning |
| --- | --- |
| `base_url` | Root the provider serves from. Shunt appends the path itself (`/chat/completions`, `/v1/messages`). |
| `protocol` | `openai` or `anthropic`. This is what the provider speaks, and it decides which translation runs. |
| `api_key_env` | **Name** of the environment variable holding the key, never the key itself. Omit it for a provider that needs no credential. |

### `models`: what can answer

```python
models = {
    "qwen-local": {
        "provider": "local",
        "model": "qwen3.8-27b",
        "supports": {"tools": True, "streaming": True, "vision": True},
        "context_window": 32000,
        "max_output_tokens": 8192,
    },
}
```

The key (`qwen-local`) is your alias, used in routes. `model` is the name the
provider itself knows. For llama.cpp that is whatever `--alias` was given, not
the `.gguf` filename.

`supports` and `context_window` are how Shunt drops a candidate before wasting a
round trip on it. A request is skipped past a model that cannot serve it: one
carrying `tools` skips a model without tool support, one carrying an image
skips a model without vision, one asking to stream skips a model that cannot,
and one whose estimated input exceeds `context_window` skips that model too.

### `routes`: which models answer which request

```python
routes = [
    ("free",   ["free"]),
    ("haiku",  ["qwen-local", "free"]),
    ("sonnet", ["qwen-local", "free"]),
    ("opus",   ["qwen-local", "free"]),
]
```

Each entry is a pattern and an ordered chain. The pattern is matched against the
model name the client asked for; the chain is tried left to right. Because
matching is by substring, `haiku` catches `claude-haiku-4-5` and every other
version of that size.

### `default_model`

```python
default_model = "qwen-local"
```

What answers when no route matches. Set it to `None` to enable transparent mode
instead, described below.

---

## How a model is chosen

`resolve()` tries four rules, in this order, and stops at the first that
matches:

1. **Exact**: the requested name equals a route pattern.
2. **Family**: a route pattern appears inside the requested name. This is how
   `claude-sonnet-4-5` reaches the `sonnet` route.
3. **Default**: no pattern matched and `default_model` is set, so that model
   answers.
4. **Transparent**: no pattern matched and there is no default. Shunt guesses
   the provider from the model name and forwards the request carrying **the
   client's own credential**, contributing no configuration of its own. The
   guess reads the prefix: `claude-...` goes to the provider named `anthropic`,
   `gpt-...`/`o1...`/`o3...` to the one named `openai`, and any name containing
   a slash (`deepseek/deepseek-chat`) to the one named `openrouter`. A name
   that fits none of those, or that names a provider you have not declared, is
   answered with a 400 telling you to add the provider or write a route for it.

For the first three rules, `default_model` is inserted as the second candidate
of the chain, so a configured fallback exists even for a one-model route. The
transparent rule builds no chain: it is one candidate, and there is nothing to
fall back to.

---

## What gets translated

| Anthropic | OpenAI | Notes |
| --- | --- | --- |
| root `system` field | leading `system` message | |
| content blocks | single string, or parts list | Text blocks are joined; images become image parts. |
| `tool_use` block | `tool_calls[]` entry | `input` object becomes a JSON **string** in `function.arguments`. |
| `tool_result` block | `role: "tool"` message | Ordered so a strict provider accepts the turn. |
| `thinking` block | `reasoning_content` | Both directions. See below. |
| `stop_sequences` | `stop` | Capped at 4, the OpenAI limit. |
| `stop_reason` | `finish_reason` | `end_turn`/`max_tokens`/`tool_use` ↔ `stop`/`length`/`tool_calls`. |
| `input_tokens`/`output_tokens` | `prompt_tokens`/`completion_tokens` | |
| `msg_...` id | `chatcmpl-...` id | |
| `cache_control` | none | Dropped: an Anthropic billing instruction with no receiver on the OpenAI side. |

### Tool-call ids survive the round trip

Shunt keeps no state between requests, so the id mapping is a pure function of
the id string. An OpenAI `call_abc` is base64-encoded, with a short checksum,
into a `toolu_s...` id; the reverse decodes it back. An id that Shunt never
encoded, such as a native Claude Code id, is hashed into a deterministic
`call_...` instead, so turn two of a conversation still matches turn one.

### Reasoning becomes thinking

A reasoning model returns its reasoning in `reasoning_content` (llama.cpp,
DeepSeek) or `reasoning` (OpenRouter). Shunt reads both and emits an Anthropic
`thinking` block before the text block, in streaming and non-streaming alike.
It goes back up the same way: a `thinking` block the client sends in turn two
returns to the provider as `reasoning_content`, so the model is not made to
re-derive what it already paid tokens for.

No `signature` is attached to the block. Shunt cannot sign anything, and an
invented value would be read as a real one.

---

## Fallback, retries and timeouts

A failed attempt is classified into one of three outcomes:

| Situation | Outcome |
| --- | --- |
| status below 400 | **OK** |
| 408, or any 5xx | **retry** the same candidate |
| 429 with a `Retry-After` of 5s or less | **retry** after waiting |
| 429 with a longer or missing `Retry-After` | **skip** to the next candidate |
| any other 4xx | **skip** |
| transport error (connection refused, reset) | **retry** |
| any other exception | **skip** |

Up to **3 attempts** per candidate, with exponential backoff plus jitter.

For streaming there is one extra rule, and it is the important one: **fallback
is only possible before the first real event reaches the client.** Once a byte
of the answer has gone out, Shunt is committed to that candidate; a later
failure becomes an error event on the wire rather than a silent switch. A
candidate that sends nothing meaningful within **20 seconds** is abandoned and
the next one gets its turn. While that decision is still open, Shunt sends SSE
comments to keep the connection warm. Comments, not events: a strict client parser
never sees anything before `message_start`.

---

## Endpoints

| Method | Path | Serves |
| --- | --- | --- |
| `GET` | `/` | The usage panel. |
| `GET` | `/health` | Health check. Touches no configuration. |
| `POST` | `/v1/messages` | Anthropic Messages. Streaming supported. |
| `POST` | `/v1/messages/count_tokens` | Forwards to an Anthropic provider when there is one; estimates locally otherwise. |
| `POST` | `/v1/chat/completions` | OpenAI chat. Streaming supported. |
| `POST` | `/v1/completions` | OpenAI legacy completions. No streaming. |
| `POST` | `/v1/embeddings` | OpenAI embeddings. No streaming. |
| `GET` | `/v1/models` | Your catalogue, in the dialect the caller speaks. |
| `GET` | `/api/stats` | The panel's summary as JSON, cached for a few seconds. |
| `GET` | `/api/stats/stream` | One SSE event per finished request, read from memory. |
| `POST` | `/api/stats/clear` | Deletes the stored history. Needs `{"confirm": true}`. |
| `GET` | `/requests` | The request audit screen. |
| `GET` | `/api/requests` | Search the stored requests. Filters combine; paging is by cursor. |
| `GET` | `/api/requests/export` | The same search as CSV. |
| `GET` | `/api/requests/{id}` | One request, with its whole chain of attempts. |

`/v1/models` answers Anthropic shape to a caller sending `anthropic-version`,
`x-api-key` or a Claude user agent; OpenAI shape to one sending only
`Authorization: Bearer`; and a superset carrying both sets of keys when there is
no signal either way, so whichever parser reads it finds its own fields.

Errors follow the same rule: the envelope matches the protocol of whoever asked,
never the protocol of whatever failed.


---

## The usage panel

Open `http://127.0.0.1:8000/` and the home page shows what the proxy has been
doing: which models were asked for against which actually answered, tokens per
hour, time to first token, how often the first candidate sufficed, which tools
were offered against which were called, and every error by status. Requests
appear on the tape as they finish, pushed over SSE.

Persistence is optional and off until you configure it:

```bash
# .env
TURSO_DATABASE_URL=sqlite+libsql://your-database.turso.io
TURSO_AUTH_TOKEN=...
```

A local file works too, and is the easiest way to start — no account, no token:

```bash
TURSO_DATABASE_URL=sqlite+pysqlite:///./stats.db
```

With no database at all the panel still runs and still shows live traffic; it
just keeps no history, and says so in its footer.

**The panel cannot slow the API down, and that is measured rather than claimed.**
A finished request is put on a bounded in-memory queue and the response goes out;
a single background worker writes batches of up to 200 rows in one commit. A full
queue drops events and counts the drops — losing a statistic beats holding up a
request. Against a local instant provider, so that only the proxy's own cost is
visible:

| | p50 | p95 | p99 |
| --- | --- | --- | --- |
| No persistence | 2.59 ms | 3.77 ms | 4.33 ms |
| Recording to the database | 2.92 ms | 4.85 ms | 5.85 ms |
| Recording, panel open | 3.05 ms | 4.70 ms | 5.76 ms |
| Database unreachable | 2.58 ms | 3.75 ms | 4.16 ms |

### The request screen

`http://127.0.0.1:8000/requests` is the other half: the panel says how things
are going, this says what happened in one request, and how many look like it.

Type into the search box and it matches the request id, either model, the
provider, a tool name, or the reason a candidate was skipped. The chips beside
it are the five cuts that get asked for most — failures, streams, fallbacks,
requests that called a tool, anything slower than five seconds — and they
combine with everything else. Click a row and the panel beside it shows that
request's whole chain: each candidate that was skipped and why, then the one
that answered, with tokens, latency and time to first token.

The search lives in the URL, so an investigation is a link you can send to
someone. **Exportar CSV** hands the same result to a spreadsheet.

What the screen never shows is the prompt or the answer. They are not stored.

### Clearing the history

The panel's footer has a **Limpar histórico** button. It takes two clicks: the
first arms it and says how many requests are about to go, the second does it,
and leaving it alone disarms it after eight seconds. The same thing from a
terminal, for a script or a cron:

```bash
curl -X POST http://127.0.0.1:8000/api/stats/clear \
  -H 'content-type: application/json' \
  -d '{"confirm": true}'
# {"deleted": 194, "older_than_hours": null}

# Keep the last hour, drop everything older:
curl -X POST http://127.0.0.1:8000/api/stats/clear \
  -H 'content-type: application/json' \
  -d '{"confirm": true, "older_than_hours": 1}'
```

The `confirm` is required, and a POST without it is refused. Deleting history
never touches a request in flight: the recorder only inserts, and its queue is
not consulted here.

Recording costs about a third of a millisecond at the median on a 2.6 ms floor.
Against a real provider — 371 ms for a measured Groq call — that is under half a
percent, and an unreachable database costs nothing at all, because Shunt refuses
to hand a dead host to the libsql driver in the first place.

---

## Development

```bash
make dev     # uvicorn with reload, port 8000
make test    # the suite, without E2E
make lint    # ruff
make type    # mypy
make check   # lint + type + suite with coverage
make e2e     # the app end to end against a scripted provider
make browser # the panel in a headless browser
make prod    # no reload, one worker per core, access log off
```

The proxy holds no credentials of its own. For a configured model it reads the
environment variable your `providers` entry names; for a transparent one it
forwards the caller's own header.

---

## What Shunt does not do

Stated plainly so none of it reads as an oversight:

- **It does not make a small model behave like a large one.** Routing
  `claude-opus-4-5` to a 27B model gives you that 27B model, under a name Claude
  Code recognises.
- **It does not cache, log or store your conversations.** The usage panel keeps
  one row per request — which model, which provider, how many tokens, how long —
  and never the prompt or the answer. Tool-call ids are still encoded rather than
  remembered.
- **It does not filter a provider's response.** Whatever the provider answers is
  translated and handed on. A field Shunt has never heard of reaches you intact,
  and so does a field you would rather it dropped.
- **It does not implement `/v1/completions` or `/v1/embeddings` as streams.**
  Both answer a single JSON document.
- **An image inside a `tool_result` is lost.** Only the text blocks of a tool
  result survive the translation, so a screenshot-returning tool reports an
  empty result to the model.
