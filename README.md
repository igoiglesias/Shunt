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
- [Admin area](#admin-area)
- [The usage panel](#the-usage-panel)
- [The request screen](#the-request-screen)
- [Cache: what the provider reported](#cache-what-the-provider-reported)
- [Generation rate and project](#generation-rate-and-project)
- [Analysing a period](#analysing-a-period)
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

Then edit `.env`. It holds one key per provider of the shipped catalogue, plus
the database URL:

```bash
LOCAL_API_KEY=local                  # same value you passed to llama-server --api-key
OPENROUTER_API_KEY=sk-or-v1-...
GROQ_API_KEY=gsk_...
TURSO_DATABASE_URL=sqlite+pysqlite:///./stats.db
```

`.env` is in `.gitignore`. No credential belongs in any file that git tracks.

The provider keys are read **once**, on the first boot, when Shunt seeds its
catalogue into the database. After that the database is the source of truth:
changing a key in `.env` later changes nothing, so change it on the
configuration screen instead.

**3. Start the server.**

```bash
make dev
```

That runs uvicorn on port 8000 with reload enabled. On the first boot against an
empty database, Shunt inserts the catalogue from `app/config/seed.py`: a
llama.cpp server on `127.0.0.1:8080`, with OpenRouter and Groq behind it.

**4. Create the first admin.** Open `http://127.0.0.1:8000/admin/login`. While
the user table is empty, the login screen is a sign-up screen: the first
account you create there is the admin. Then go to **Configuração** and point
the catalogue at your own provider. The [Configuration](#configuration) section
explains every field.

**5. Check that it is up.**

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}

curl -H 'anthropic-version: 2023-06-01' http://127.0.0.1:8000/v1/models
# {"data":[{"type":"model","id":"qwen-local", ...}],"has_more":false,"first_id":"qwen-local"}
```

The health check deliberately touches neither your configuration nor your
provider, so it answers even when those are broken. That is exactly the
moment you need it to. It lives at `/health`; the home page redirects to the
usage panel, behind the login.

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

### With a Shunt token

A client can also present a token created on the **Tokens** screen, in the
`x-shunt-token` header:

```bash
ANTHROPIC_CUSTOM_HEADERS='x-shunt-token: <token>' \
ANTHROPIC_BASE_URL=http://127.0.0.1:8000 \
ANTHROPIC_AUTH_TOKEN=local \
claude
```

When the header is present, Shunt checks it: an unknown or expired token gets a
401, and with no database a 503. A valid token makes Shunt use the key stored
for the provider on every candidate, transparent ones included, so the
client's own credential never leaves the machine. A request without the header
is not checked, exactly as above.

The token is shown once, when it is created. Shunt stores only its SHA-256
hash.

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

The catalogue lives in the database: four tables for providers, models, routes
and the candidates of each route. You edit it on the **Configuração** screen
(`/admin/config`), and every change applies to the next request, with no
restart.

`app/config/seed.py` holds the catalogue Shunt starts from. At boot, when any
of providers, models or routes is empty, Shunt inserts the parts of that
catalogue that are missing. It never overwrites a provider or a model you
already have, so editing `seed.py` does not change a running installation.

Every change on the screen records a version. **Histórico de Versões** lists them,
and a rollback restores providers, models and routes from that version. The
history never stores a key: after a rollback each provider keeps its current
key.

Without a database there is no catalogue: Shunt starts with no providers, no
models and no routes, and every request that needs one is answered with a 400.

### Providers: where requests can go

| Field | Meaning |
| --- | --- |
| name | How models refer to the provider. The transparent rule also looks for the names `anthropic`, `openai` and `openrouter`. |
| `base_url` | Root the provider serves from. Shunt appends the path itself (`/chat/completions`, `/v1/messages`). |
| `protocol` | `openai` or `anthropic`. This is what the provider speaks, and it decides which translation runs. |
| API key | The key itself, stored in the database. Leave it empty for a provider that needs no credential. |

### Models: what can answer

| Field | Meaning |
| --- | --- |
| alias | Your name for the model, used in routes (`qwen-local`). |
| provider | Which provider serves it. |
| upstream model | The name the provider itself knows. For llama.cpp that is whatever `--alias` was given, not the `.gguf` filename. |
| capabilities | Tools, streaming, vision: one flag each. |
| context window, max output tokens | Limits in tokens. |

The capabilities and the context window are how Shunt drops a candidate before
wasting a round trip on it. A request is skipped past a model that cannot serve it: one
carrying `tools` skips a model without tool support, one carrying an image
skips a model without vision, one asking to stream skips a model that cannot,
and one whose estimated input exceeds `context_window` skips that model too.

### Routes: which models answer which request

The shipped catalogue has entries such as:

| Pattern | Chain |
| --- | --- |
| `haiku` | `groq-free`, `open-free`, `open-nemotron-ultra` |
| `sonnet` | `open-free`, `open-nemotron-ultra`, `groq-free` |
| `opus` | `qwen-local`, `open-nemotron-ultra`, `open-free` |

Each entry is a pattern and an ordered chain. The pattern is matched against the
model name the client asked for; the chain is tried left to right. Because
matching is by substring, `haiku` catches `claude-haiku-4-5` and every other
version of that size. The routes themselves are ordered too, and the first
matching pattern wins; the screen lets you reorder them.

### Default model

One model can be marked as the default. It answers when no route matches. With
no model marked, Shunt uses transparent mode instead, described below.

---

## How a model is chosen

`resolve()` tries four rules, in this order, and stops at the first that
matches:

1. **Exact**: the requested name equals a route pattern.
2. **Family**: a route pattern appears inside the requested name. This is how
   `claude-sonnet-4-5` reaches the `sonnet` route.
3. **Default**: no pattern matched and a default model is marked, so that model
   answers.
4. **Transparent**: no pattern matched and there is no default. Shunt guesses
   the provider from the model name and forwards the request carrying **the
   client's own credential** (or, with a valid `x-shunt-token`, the key stored
   for that provider), contributing no configuration of its own. The
   guess reads the prefix: `claude-...` goes to the provider named `anthropic`,
   `gpt-...`/`o1...`/`o3...` to the one named `openai`, and any name containing
   a slash (`deepseek/deepseek-chat`) to the one named `openrouter`. A name
   that fits none of those, or that names a provider you have not declared, is
   answered with a 400 telling you to add the provider or write a route for it.

For the first three rules, the default model is inserted as the second candidate
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

Every number above is an environment variable with that default:

| Variable | Default | Bounds |
| --- | --- | --- |
| `SHUNT_MAX_ATTEMPTS` | 3 | attempts per candidate |
| `SHUNT_RETRY_AFTER_BUDGET` | 5 | longest `Retry-After`, in seconds, that Shunt waits out |
| `SHUNT_FIRST_EVENT_DEADLINE` | 20 | seconds a stream may take to send its first real event |
| `SHUNT_TOTAL_DEADLINE` | 120 | seconds for the whole chain of a request that does not stream |
| `SHUNT_TIMEOUT_CONNECT` / `_READ` / `_WRITE` / `_POOL` | 10 / 60 / 30 / 10 | the HTTP client, in seconds |

The remaining `SHUNT_*` knobs (queue sizes, panel limits, cookie lifetime) are
listed with their defaults in `app/config/config.py`.

---

## Endpoints

| Method | Path | Serves |
| --- | --- | --- |
| `GET` | `/` | Redirects to `/admin/painel`. |
| `GET` | `/health` | Health check. Touches no configuration. |
| `POST` | `/v1/messages` | Anthropic Messages. Streaming supported. |
| `POST` | `/v1/messages/count_tokens` | Forwards to an Anthropic provider when there is one; estimates locally otherwise. |
| `POST` | `/v1/chat/completions` | OpenAI chat. Streaming supported. |
| `POST` | `/v1/completions` | OpenAI legacy completions. No streaming. |
| `POST` | `/v1/embeddings` | OpenAI embeddings. No streaming. |
| `GET` | `/v1/models` | Your catalogue, in the dialect the caller speaks. |
| `GET` | `/v1/models/{model_id}` | One model of the catalogue. |
| `GET` | `/admin/login` | Login; while there is no user, sign-up of the first admin. |
| `POST` | `/admin/logout` | Ends the session. |
| `GET` | `/admin/painel` | The usage panel. |
| `GET` | `/admin/requests` | The request audit screen. |
| `GET` | `/admin/config` | Providers, models, routes, default model and version history. |
| `GET` | `/admin/users` | Admin accounts. |
| `GET` | `/admin/tokens` | Shunt tokens for `x-shunt-token`. |
| `GET` | `/api/stats` | The panel's summary as JSON, cached for a few seconds. |
| `GET` | `/api/stats/stream` | One SSE event per finished request, read from memory. |
| `POST` | `/api/stats/clear` | Deletes the stored history. Needs `{"confirm": true}`. |
| `GET` | `/api/requests` | Search the stored requests. Filters combine; paging is by cursor. |
| `GET` | `/api/requests/facets` | The values the search filters offer. |
| `GET` | `/api/requests/export` | The same search as CSV. |
| `GET` | `/api/requests/{id}` | One request, with its whole chain of attempts. |
| `GET` | `/api/requests/{id}/body` | The stored conversation of one request, when recording is on. |
| `POST` | `/api/analysis` | Analyses a period with a model. |
| `GET` | `/api/analysis` | Stored analyses. |
| `GET` | `/api/analysis/{id}` | One stored analysis. |

Everything under `/admin` and `/api` needs an admin session. Without one, a
screen redirects to the login and an `/api` call gets a 401. The `/v1` routes
and `/health` never ask for it.

`/v1/models` answers Anthropic shape to a caller sending `anthropic-version`,
`x-api-key` or a Claude user agent; OpenAI shape to one sending only
`Authorization: Bearer`; and a superset carrying both sets of keys when there is
no signal either way, so whichever parser reads it finds its own fields.

Errors follow the same rule: the envelope matches the protocol of whoever asked,
never the protocol of whatever failed.

---

## Admin area

The screens under `/admin` share one login. There are no roles: every account
is an admin.

- **First access.** While the user table is empty, `/admin/login` creates the
  first account instead of checking one. Without a database it only offers the
  login, because there would be nowhere to store the account.
- **Usuários** adds, edits and removes accounts. Passwords are stored as
  Argon2 hashes.
- **Tokens** creates the tokens a client sends in `x-shunt-token`, optionally
  expiring after a number of days, and revokes them. See
  [With a Shunt token](#with-a-shunt-token).
- **Configuração** edits the catalogue. See [Configuration](#configuration).

The session is a signed JWT in an `httponly`, `samesite=strict` cookie that
lasts 12 hours (`SHUNT_ADMIN_COOKIE_MAX_AGE`, in seconds). It is signed with
`ADMIN_SESSION_SECRET`. Without that variable Shunt draws a random secret at
each boot, so every restart logs everyone out, and under `make prod` each worker
signs with a different secret. Set it for anything beyond a single `make dev`:

```bash
ADMIN_SESSION_SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
```

---

## The usage panel

Open `http://127.0.0.1:8000/admin/painel`, log in, and the panel shows what the
proxy has been doing: which models were asked for against which actually answered, tokens per
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

`http://127.0.0.1:8000/admin/requests` is the other half: the panel says how things
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

### Reading the conversation

Off by default, because the text of a conversation is the most sensitive thing
that passes through this proxy. Turn it on and the **Conversa** tab of any
request shows what went up and what came back:

```bash
SHUNT_STORE_BODIES=1
SHUNT_BODY_LIMIT=64000   # characters per side; 64000 is the default
```

The text is split back into turns, coloured by who spoke, with tool calls and
tool results in place — reading an agentic turn means following who said what.
There is a search box that highlights hits inside the conversation, a copy
button, and long turns collapse until you ask for the rest.

Three things happen to the text before it is stored. It goes in a **separate
table**, so the search reads hundreds of rows without dragging megabytes behind
it. Each side is **cut** at the limit, and the original size is kept so the
screen can say how much is missing rather than pretend the conversation ended
there. And anything that **looks like a credential** — an `sk-`/`ghp_`/`xoxb-`
token, an AWS key, a JWT, a `FOO_API_KEY=` line — is replaced with `[redigido]`
before it reaches the database, because an agent that pastes a `.env` into a
prompt would otherwise store the key forever.

The **Requisição** tab beside it shows the raw body that was sent, redacted and
cut the same way, with a button that copies it as a `curl` — the conversation
says what was said, this says how to reproduce it.

Clearing the history takes the stored conversations with it.

## Cache: what the provider reported

The panel has a **cache** column per model and a figure in the window summary:
how much of the input the provider said it served from cache. Shunt used to drop
that signal in translation — so neither the client nor the panel could tell
whether the repeated prefix was being reused at all.

**Absent and zero are different**, and the screen keeps them apart. Measured on
the three providers here, with two identical requests each:

| provider | what it reports |
|---|---|
| llama.cpp (local) | 2nd call: `cached_tokens` 2814 of 2818 — it caches on its own, for free |
| Groq `gpt-oss-120b` | no `prompt_tokens_details` at all — says nothing |
| OpenRouter free model | `cached_tokens: 0` on both — says it reused nothing |

A provider that says nothing shows `—`, never `0%`: calling silence "0% cache"
is how a reader concludes the cache is broken when it was never measured — and
it is exactly what led an analysis of this panel to recommend turning on a cache
that was already on. The rate counts only the requests whose provider reported,
and each row says how many those were.

The signal also reaches the client now: an OpenAI-shaped
`prompt_tokens_details.cached_tokens` becomes Anthropic's
`cache_read_input_tokens` (and back the other way), so a harness that displays
cache hits can display them through Shunt.

One thing Shunt does not do: add `cache_control` of its own. The OpenAI chat API
has no such field, so it cannot survive the translation to the providers
configured here; against an Anthropic-protocol candidate the client's own
`cache_control` passes through untouched, which is the correct behaviour for a
proxy that holds no credentials and rewrites no prompts.

## Generation rate and project

**tok/s** is on the panel, per model and per provider, and in the window
summary. It is output tokens over *generation* time: for a streamed request
that is `duration - time to first token`, because the wait before the first
token is queueing, prefill and whatever candidates were tried and skipped —
counting it would describe the chain, not the model. Outside streaming the two
cannot be separated, so the whole duration counts and the number comes out a
little lower than the truth; the column says so.

The rate is pooled — tokens summed over time summed — rather than a median of
per-request rates, so a 22-token reply weighs 22 tokens instead of one whole
vote. Requests with no output, or no measurable generation time, are in neither
half of the fraction, and each row carries how many of its requests were
actually measured. Nothing to measure shows `—`, never `0`: zero would be a
claim about speed.

**Projeto** answers where a request came from. Claude Code does not send the
directory in a header — it sends it in the body, in an environment block that
sits sometimes in a `system` message and sometimes inside a `<system-reminder>`
of the first user message. Shunt reads it as the request goes through and
stores it in a column of its own, so it works with conversation recording
**off** (measured: of the bodies that were stored, 249 of 261 were cut at the
64k limit and could not be parsed afterwards — reading it later does not work).
Requests of the same session that no longer repeat the block inherit the
project from the session.

A request with no project is normal — another client, or a version that renamed
the block — and the panel shows it as a row called **sem projeto** with its
count rather than hiding it, so the table still adds up to the window. The
request screen has a project selector, the column, and the CSV carries it.

An existing database gains the two columns at boot; no migration tool involved.

### Analysing a period

**Analisar este período** on the request screen hands the period you are looking
at to a model and asks it, as a specialist, how to improve the loop: skills,
tools, prompts, model choice, harness settings. The button inherits the active
filters, so the slice analysed is the slice on screen.

What it sends is not a dump of the database. It is a dossier built from the
queries the screens already use:

- volume, tokens, latency and errors for the period, by model and by provider;
- the chain — how often each candidate was skipped, and for which of the four
  reasons;
- tools offered against tools called, because a tool nobody ever calls pays
  prompt on every request;
- the slowest and the costliest requests, each with its chain;
- a sample of conversations, **only when recording is on**, already redacted and
  cut.

The request count comes from a `COUNT` over the whole period while the
aggregates read at most five thousand rows; when those differ the dossier says
`sampled: true` and the screen says so above the reading. The prompt requires
every recommendation to cite the number that supports it, and the **Dossiê** tab
beside the reading holds those numbers, so a recommendation can be checked
rather than believed.

The call goes out **through Shunt itself** — same resolution, same chain, same
credentials as any request from the harness — so the analysis shows up in the
panel like any other request, and the tokens it spent are counted there. The
default model reads the period, or send `{"model": "..."}` to pick another. Like
every `/api` route it needs the admin session, so a terminal call carries the
`shunt_admin` cookie copied from the browser:

```bash
curl -X POST 'http://127.0.0.1:8000/api/analysis?since=2026-09-19T00:00:00Z' \
  -b 'shunt_admin=<cookie>' \
  -H 'content-type: application/json' -d '{"model": "claude-opus-5"}'
```

An analysis is stored with the dossier that produced it and reused for the same
window — it costs tokens, and nobody wants to pay twice for the same period.
**Refazer**, or `{"refresh": true}`, pays again on purpose. An analysis
that failed is never cached.

### Clearing the history

The panel's footer has a **Limpar histórico** button. It takes two clicks: the
first arms it and says how many requests are about to go, the second does it,
and leaving it alone disarms it after eight seconds. The same thing from a
terminal, with the admin session cookie:

```bash
curl -X POST http://127.0.0.1:8000/api/stats/clear \
  -b 'shunt_admin=<cookie>' \
  -H 'content-type: application/json' \
  -d '{"confirm": true}'
# {"deleted": 194, "older_than_hours": null}

# Keep the last hour, drop everything older:
curl -X POST http://127.0.0.1:8000/api/stats/clear \
  -b 'shunt_admin=<cookie>' \
  -H 'content-type: application/json' \
  -d '{"confirm": true, "older_than_hours": 1}'
```

The cookie expires with the session, so a cron job needs a fresh one every 12
hours by default.

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

For a configured model, Shunt sends the provider key stored in the database.
For a transparent one it forwards the caller's own header, unless the request
carries a valid `x-shunt-token`. The keys reach the database from `.env` only
once, when the catalogue is seeded; after that they change on the
configuration screen.

---

## What Shunt does not do

Stated plainly so none of it reads as an oversight:

- **It does not make a small model behave like a large one.** Routing
  `claude-opus-4-5` to a 27B model gives you that 27B model, under a name Claude
  Code recognises.
- **It does not store your conversations unless you ask it to.** The usage panel
  keeps one row per request — which model, which provider, how many tokens, how
  long — and nothing of what was said. `SHUNT_STORE_BODIES=1` adds the text, in
  its own table, cut at a limit and with anything that looks like a credential
  redacted. Tool-call ids are still encoded rather than remembered.
- **It does not filter a provider's response.** Whatever the provider answers is
  translated and handed on. A field Shunt has never heard of reaches you intact,
  and so does a field you would rather it dropped.
- **It does not implement `/v1/completions` or `/v1/embeddings` as streams.**
  Both answer a single JSON document.
- **An image inside a `tool_result` is lost.** Only the text blocks of a tool
  result survive the translation, so a screenshot-returning tool reports an
  empty result to the model.
