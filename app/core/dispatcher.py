"""Walk the candidate chain: retry, fall back, and translate when protocols differ.

This is the module every request passes through. It owns the loop that
`app/core/attempt.py` deliberately does not: which candidate to try, how many
times, and when to give up on the whole chain.

Two rules here are security surface, not style:

- In routed mode the client's inbound headers NEVER reach the provider. Only
  the credential read from configuration goes out. A harness pointing at this
  proxy sends its own Anthropic key; forwarding it to a third-party provider
  would hand that provider the user's credential.
- In transparent mode the inbound headers go out verbatim, minus `host`,
  `content-length` and `accept-encoding` (the first two are wrong for the new
  destination and the new body; the third would ask for an encoding httpx did
  not negotiate). Copying only the API key is NOT enough: a Claude Code
  subscription authenticates with an OAuth bearer token plus `anthropic-beta`
  and `anthropic-version`, and dropping those breaks authentication silently.

`outbound_headers` is public on purpose -- later tasks (streaming, the
OpenAI-facing routes, token counting) build their own requests and must build
the same headers this module does.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import aclosing
from dataclasses import dataclass, field

import httpx

from app.config.settings import Settings
from app.core.attempt import (
    FIRST_EVENT_DEADLINE,
    MAX_ATTEMPTS,
    TOTAL_DEADLINE,
    Outcome,
    backoff,
    classify,
)
from app.core.capabilities import filter_chain, requirements_of
from app.core.observability import RequestLog, log_request
from app.core.resolver import Candidate, Resolution, resolve
from app.core.upstream import UpstreamPool
from app.schemas.openai import OpenAIErrorResponse
from app.stats import bodies
from app.translate.sse_parse import SSEDecoder, SSEEvent
from app.translate.sse_to_anthropic import OpenAIStreamToAnthropic
from app.translate.sse_to_openai import AnthropicStreamToOpenAI
from app.translate.to_anthropic import (
    openai_error_to_anthropic,
    openai_response_to_anthropic,
    reasoning_of,
)
from app.translate.to_anthropic_request import openai_request_to_anthropic
from app.translate.to_openai import anthropic_request_to_openai, anthropic_response_to_openai

# (protocolo do provedor, endpoint pedido pelo cliente) -> path no provedor.
# A Anthropic nao tem equivalente a /completions nem a /embeddings: o par
# simplesmente nao existe aqui, e o candidato e pulado por isso.
PATHS = {
    ("openai", "chat"): "/chat/completions",
    ("openai", "messages"): "/chat/completions",
    ("openai", "completions"): "/completions",
    ("openai", "embeddings"): "/embeddings",
    ("anthropic", "messages"): "/v1/messages",
    ("anthropic", "chat"): "/v1/messages",
}

# `host` names the wrong destination once we re-address the request; the other
# four all describe the inbound body, which we re-serialise before sending.
TRANSPARENT_DROP = frozenset(
    {"host", "content-length", "content-encoding", "transfer-encoding", "accept-encoding"}
)

DEFAULT_MAX_OUTPUT_TOKENS = 4096

# Nomes de erro da OpenAI. A tabela equivalente da Anthropic vive em
# `app/translate/to_anthropic.py`; as duas nao coincidem (a OpenAI nao tem
# `overloaded_error`, e classifica 4xx desconhecido como culpa do pedido e 5xx
# desconhecido como culpa do servidor).
OPENAI_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
}


# O path que o CLIENTE usou. `PATHS` responde o do provedor, que e outro
# sempre que o proxy traduz -- uma requisicao Anthropic servida por um provedor
# OpenAI entra em `/v1/messages` e sai em `/v1/chat/completions`.
ROUTES: dict[str, str] = {
    "messages": "/v1/messages",
    "chat": "/v1/chat/completions",
    "completions": "/v1/completions",
    "embeddings": "/v1/embeddings",
}


def error_body(protocol: str, status: int, message: str) -> dict:
    """O envelope de erro segue o protocolo de QUEM PERGUNTOU, nunca o do
    provedor que falhou.

    Enquanto `/v1/messages` era a unica rota, todo erro saia em forma
    Anthropic e isso estava certo por acidente. Com as rotas OpenAI abertas,
    um cliente que fala OpenAI recebendo `{"type": "error", ...}` encontra um
    corpo sem `error.param` e sem `error.code`, e a biblioteca oficial dele
    nao consegue classificar o que aconteceu.
    """
    if protocol == "anthropic":
        return openai_error_to_anthropic(status, message)
    return OpenAIErrorResponse.of(
        OPENAI_ERROR_TYPES.get(status, _openai_fallback_type(status)), message
    ).model_dump()


def _openai_fallback_type(status: int) -> str:
    return "server_error" if status >= 500 else "invalid_request_error"


class ChainExhausted(Exception):
    """Never raised by `dispatch`, which reports an exhausted chain as a
    `ShuntResult` carrying the last upstream status and the full trace. Do not
    write `except ChainExhausted` around `dispatch`: the branch would not fire."""


@dataclass
class ShuntRequest:
    protocol: str
    body: dict
    headers: dict[str, str]
    endpoint: str = "messages"  # "messages" | "chat" | "completions" | "embeddings"


@dataclass
class ShuntResult:
    status: int
    body: dict
    real_model: str | None = None
    trace: list[str] = field(default_factory=list)
    real_provider: str | None = None


def outbound_headers(req: ShuntRequest, candidate: Candidate, settings: Settings) -> dict:
    if candidate.transparent:
        # The client's own credentials go verbatim to whatever `base_url` this
        # provider entry names. That is the user's declared intent: they wrote
        # the entry. Do not read a protocol mismatch as evidence that the
        # destination is Anthropic -- `ProviderConfig.protocol` is free, so a
        # provider named `anthropic` may well be an OpenAI-compatible gateway.
        return {k: v for k, v in req.headers.items() if k.lower() not in TRANSPARENT_DROP}
    key = settings.api_key(candidate.provider)
    headers = {"content-type": "application/json"}
    if key and candidate.protocol == "anthropic":
        headers["x-api-key"] = key
        headers["anthropic-version"] = "2023-06-01"
    elif key:
        headers["authorization"] = f"Bearer {key}"
    return headers


def _cap(candidate: Candidate, settings: Settings) -> int:
    if candidate.alias:
        return settings.models[candidate.alias].max_output_tokens
    return DEFAULT_MAX_OUTPUT_TOKENS


def _transparent_cap(req: ShuntRequest) -> int:
    """The ceiling for a transparent translation is the CLIENT's own number.

    Byte-transparency is already gone the moment `PATHS` re-addresses the
    request to the provider's own path: once the URL is the provider's, the
    body has to be the provider's shape too. What stays transparent is that we
    contribute no alias configuration and no credential of ours. So the cap
    handed to the translator must be the client's `max_tokens` -- the
    translators clamp with `min(body_value, cap)`, and passing `_cap()` here
    would return `DEFAULT_MAX_OUTPUT_TOKENS` for `alias=None` and silently cut
    a client's 999999 down to 4096.
    """
    raw = req.body.get("max_tokens")
    return int(raw) if isinstance(raw, int | float) else DEFAULT_MAX_OUTPUT_TOKENS


def _payload(req: ShuntRequest, candidate: Candidate, settings: Settings) -> dict:
    if candidate.protocol == req.protocol:
        out = {**req.body, "model": candidate.model}
        raw = out.get("max_tokens")
        # Not in transparent mode: no candidate configuration exists there and
        # the client's number is not ours to rewrite. A non-numeric value is
        # left untouched on purpose, so the provider answers with its own 400
        # instead of us skipping the candidate over a coercion error.
        if not candidate.transparent and candidate.alias and isinstance(raw, int | float):
            out["max_tokens"] = min(int(raw), _cap(candidate, settings))
        return out
    cap = _transparent_cap(req) if candidate.transparent else _cap(candidate, settings)
    if req.protocol == "anthropic":
        return anthropic_request_to_openai(req.body, candidate.model, cap)
    return openai_request_to_anthropic(req.body, candidate.model, cap)


def _translate_response(data: dict, req: ShuntRequest, candidate: Candidate) -> dict:
    if candidate.protocol == req.protocol:
        return data
    requested = req.body.get("model", "")
    if req.protocol == "anthropic":
        return openai_response_to_anthropic(data, requested)
    return anthropic_response_to_openai(data, requested)


def _missing_credential(candidate: Candidate, settings: Settings) -> str | None:
    """Nome da variavel que o provedor declara e o ambiente nao tem, ou None.

    Declarar `api_key_env` e nao ter a variavel e erro de instalacao, nao de
    requisicao: chamar assim manda um pedido sem credencial, leva 401 e gasta
    uma tentativa para descobrir o que a configuracao ja sabia. Um provedor com
    `api_key_env=None` -- um llama.cpp aberto, por exemplo -- nao quer
    credencial nenhuma e continua sendo chamado.
    """
    if candidate.transparent:
        return None
    env = settings.providers[candidate.provider].api_key_env
    if env and not settings.api_key(candidate.provider):
        return env
    return None


def _exception_text(exc: BaseException | None) -> str:
    """Nome da classe mais a mensagem, porque a mensagem sozinha pode ser vazia.

    Medido contra um llama.cpp real: `str(httpx.ReadTimeout())` e a string
    vazia, e o cliente recebia `" - tried: qwen-local:  (attempt 1)"`, que nao
    diz nem que houve timeout. O nome da classe e o unico pedaco que o httpx
    sempre da.
    """
    if exc is None:
        return "transport failure"
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _retry_after(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    raw = response.headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


def _error_message(response: httpx.Response) -> str:
    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            body = response.json()
        except ValueError:
            return response.text
        if isinstance(body, dict):
            error = body.get("error")
            # Ollama and friends answer `{"error": "model not found"}` -- a bare
            # string where OpenAI puts an object. Reading `.get` off it raised.
            if isinstance(error, str):
                return error
            if isinstance(error, dict):
                return str(error.get("message", response.text))
    return response.text


def tools_offered(req: ShuntRequest) -> list[str]:
    """Nomes das ferramentas que o CLIENTE ofereceu, nos dois dialetos."""
    names = []
    for tool in req.body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name") or (tool.get("function") or {}).get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def _tools_called(body: dict) -> tuple[list[str], int]:
    """Ferramentas chamadas e blocos de raciocinio na resposta ja traduzida.

    Le a forma de quem perguntou: o corpo aqui e o que o cliente vai receber,
    entao Anthropic tem blocos em `content` e OpenAI tem `tool_calls` dentro de
    cada `choice`.
    """
    called: list[str] = []
    thinking = 0
    for block in body.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use" and isinstance(block.get("name"), str):
            called.append(block["name"])
        elif block.get("type") == "thinking":
            thinking += 1
    for choice in body.get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            continue
        if message.get("reasoning_content"):
            thinking += 1
        for call in message.get("tool_calls") or []:
            name = (call.get("function") or {}).get("name") if isinstance(call, dict) else None
            if isinstance(name, str):
                called.append(name)
    return called, thinking


def _error_type_of(body: dict) -> str | None:
    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("type"), str):
        return error["type"]
    return None


async def dispatch(req: ShuntRequest, settings: Settings, pool: UpstreamPool) -> ShuntResult:
    """Uma casca fina em volta de `_dispatch`, para que TODO caminho de saida
    registre a linha de log.

    Registrar dentro de `_dispatch` significaria uma chamada por `return`, e
    sao quatro: a cadeia vazia, a resposta traduzida, o corpo 2xx ilegivel e a
    cadeia exaurida. O caminho esquecido seria justamente o que ninguem
    exercita a mao.
    """
    started = time.monotonic()
    resolution = resolve(req.body.get("model", ""), settings)
    result = await _dispatch(req, settings, pool, resolution)
    usage = result.body.get("usage") or {}
    called, thinking = _tools_called(result.body)
    log_request(
        RequestLog(
            request_id=uuid.uuid4().hex,
            requested_model=req.body.get("model", ""),
            rule=resolution.rule,
            matched=resolution.matched,
            candidate=result.real_model,
            attempts=result.trace,
            input_tokens=usage.get("input_tokens") or usage.get("prompt_tokens") or 0,
            output_tokens=usage.get("output_tokens") or usage.get("completion_tokens") or 0,
            ttft_ms=None,  # so existe onde ha um primeiro evento a cronometrar
            duration_ms=int((time.monotonic() - started) * 1000),
            translated=result.real_model is not None,
            route=ROUTES.get(req.endpoint, req.endpoint),
            dialect=req.protocol,
            stream=False,
            status=result.status,
            error_type=_error_type_of(result.body),
            provider=result.real_provider,
            tools_offered=tools_offered(req),
            tools_called=called,
            thinking_blocks=thinking,
            body=bodies.capture(
                bodies.prompt_text(req.body), bodies.answer_text(result.body), req.body
            ),
        )
    )
    return result


async def _dispatch(
    req: ShuntRequest, settings: Settings, pool: UpstreamPool, resolution: Resolution
) -> ShuntResult:
    first = resolution.chain[0]
    probe_note: list[str] = []
    try:
        probe = _payload(req, first, settings)
    except Exception as err:  # noqa: BLE001 - a translator that cannot render the
        # probe must not decide the whole request. The untranslated body is a
        # worse estimate, not a fatal one, and a later candidate may take it
        # as-is. It is still recorded: if `chain[0]` is then dropped by the
        # capability filter, this is the only evidence the probe ever failed.
        probe = req.body
        probe_note = [f"probe ({first.alias or first.model}): request translation failed: {err}"]
    chain, dropped = filter_chain(resolution.chain, requirements_of(probe), settings)
    trace = probe_note + [f"{alias}: {reason}" for alias, reason in dropped]

    if not chain:
        return ShuntResult(
            400,
            error_body(
                req.protocol, 400, "no candidate can serve this request: " + "; ".join(trace)
            ),
            None,
            trace,
        )

    deadline = time.monotonic() + TOTAL_DEADLINE
    last_status, last_message = 502, "no candidate answered"

    for candidate in chain:
        label = candidate.alias or candidate.model
        if (candidate.protocol, req.endpoint) not in PATHS:
            trace.append(f"{label}: endpoint not supported")
            continue
        missing = _missing_credential(candidate, settings)
        if missing is not None:
            trace.append(f"{label}: credential {missing} is not set")
            continue
        try:
            payload = _payload(req, candidate, settings)
        except Exception as err:  # noqa: BLE001 - one candidate we cannot render
            # the request for is one candidate skipped, not a dead chain: the
            # next one may speak the client's protocol and need no translation.
            trace.append(f"{label}: request translation failed: {err}")
            continue
        client = pool.get(candidate.provider)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if time.monotonic() > deadline:
                trace.append(f"{label}: total deadline exceeded")
                break
            response, exc = None, None
            try:
                response = await client.post(
                    PATHS[(candidate.protocol, req.endpoint)],
                    json=payload,
                    headers=outbound_headers(req, candidate, settings),
                )
            except httpx.HTTPError as err:
                exc = err
            outcome = classify(
                response.status_code if response else None, exc, _retry_after(response)
            )
            if outcome is Outcome.OK and response is not None:
                try:
                    data = _translate_response(response.json(), req, candidate)
                except Exception:  # noqa: BLE001 - a 2xx we cannot read or cannot
                    # shape is this candidate's failure, not the request's. Not
                    # JSON at all (an HTML page from an interposed gateway, a
                    # truncated response) and JSON of the wrong shape (a list, a
                    # scalar, `null`) fail in different places and mean the same
                    # thing to the caller. The error path was already lenient
                    # about exactly this; the success path was not.
                    last_status = 502
                    last_message = "upstream 2xx body is not a usable JSON object"
                    trace.append(f"{label}: unreadable body (attempt {attempt})")
                    break
                return ShuntResult(
                    response.status_code, data, candidate.model, trace, candidate.provider
                )
            if response is not None:
                last_status, last_message = response.status_code, _error_message(response)
            else:
                last_status, last_message = 502, _exception_text(exc)
            trace.append(f"{label}: {last_status} (attempt {attempt})")
            if outcome is Outcome.SKIP:
                break
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(backoff(attempt))

    message = f"{last_message} - tried: " + "; ".join(trace)
    return ShuntResult(last_status, error_body(req.protocol, last_status, message), None, trace)


# ---------------------------------------------------------------------------
# Streaming
#
# The buffered path above may change its mind at any moment: nothing has
# reached the client until `dispatch` returns. Streaming has no such freedom.
# Once the first byte is on the wire the response is committed -- there is no
# way to retract a `message_start` and try another provider. So the whole
# design is about postponing that commitment for as long as the evidence is
# still ambiguous, and the evidence IS ambiguous: an OpenAI-compatible
# provider answers HTTP 200 and then puts `{"error": ...}` inside the stream,
# and it sends keep-alive comments while the request sits in a queue. Status
# and headers are therefore not enough. The commitment point is the first
# event that parses AND carries real content -- `is_first_valid_event`.
#
# Before that point a failure is a free fallback to the next candidate, and
# the client never learns a provider was tried. After it, a failure can only
# be reported as an error event on the wire.
#
# Three differences from `dispatch`, all deliberate:
#
# - Retry is bounded to the connection attempt. `client.send` is retried like
#   the buffered path does (`classify` + `backoff`, up to `MAX_ATTEMPTS`)
#   because nothing has been emitted yet, so a transient connect error is
#   indistinguishable from the buffered case. Once the response exists there
#   is no retry: a stream cannot be replayed, and one that failed before the
#   first event failed on evidence (an in-band error, a mid-stream drop, or
#   silence past the deadline) a second identical request would very likely
#   reproduce. From there on, streaming falls back; it does not retry.
# - No `TOTAL_DEADLINE`. A long stream is the normal case; the bound that
#   matters here is `FIRST_EVENT_DEADLINE`, on the wait before the first
#   event. Nothing bounds the stream's total LENGTH: what `TIMEOUT.read` in
#   `app/core/upstream.py` bounds is the silence between two reads, which is
#   what covers a provider that answers 200 and then sends nothing at all (it
#   never enters the read loop, so the in-loop deadline check never runs).
# - A well-formed stream that carries no content is a candidate FAILURE, not
#   an empty success. That is a deliberate trade-off: a provider answering
#   200 with a polite empty stream is a provider that did not serve the
#   request, and closing an empty message for it would burn the fallback for
#   nothing. The cost is that a genuinely empty completion is retried on the
#   next candidate instead of being delivered as empty.
# - When both sides speak the same protocol the bytes are forwarded verbatim,
#   with no decoding at all, and the candidate is committed on its 200. There
#   is nothing to inspect without parsing, and parsing a dialect we are not
#   translating would only let us corrupt it.
# ---------------------------------------------------------------------------

PING_INTERVAL = 5.0
DONE = b"data: [DONE]\n\n"


def _now() -> float:
    """Indirection over the clock so a test can pin the deadline and the ping
    interval to exact instants. A test on the real clock would have to sleep
    for twenty seconds and would still pass by coincidence."""
    return time.monotonic()


def is_first_valid_event(data: dict) -> bool:
    """Is this the chunk that commits us to this candidate?

    A keep-alive comment never gets here (it carries no data field). A `ping`,
    an empty delta and an `{"error": ...}` payload do get here and must all
    answer False: each of them is something a provider sends while it has not
    started answering, and treating one as the start would give up the
    fallback for nothing.

    Text, a tool call and RACIOCINIO all close the decision. The third was
    missing and it cost a whole feature: measured against llama.cpp, a
    reasoning model sends dozens of `reasoning_content` chunks before its
    first character of text, and every one of them was dropped here -- before
    the translator ever saw it -- so the client was shown no thinking at all.
    Reasoning is the model's output; a provider emitting it has started
    answering.
    """
    if data.get("error"):
        return False
    choices = data.get("choices") or []
    if not choices:
        return False
    delta = choices[0].get("delta") or {}
    return bool(delta.get("content") or delta.get("tool_calls") or reasoning_of(delta))


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _data(chunk: dict) -> bytes:
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()


def _stream_translator(req: ShuntRequest):
    """Pick the translator from the client's protocol. Equal protocols never
    reach here -- that case forwards raw bytes.

    The id is random per stream, not derived from the model: two concurrent
    streams to the same model must not share a message id.
    """
    requested = req.body.get("model", "")
    if req.protocol == "anthropic":
        return OpenAIStreamToAnthropic(requested, f"msg_{uuid.uuid4().hex}")
    return AnthropicStreamToOpenAI(requested, f"chatcmpl-{uuid.uuid4().hex}")


def _keepalive() -> bytes:
    """Keep the connection alive while we wait for the first valid event --
    the only place this is ever used, since pings stop the moment a stream
    commits (see the read loop below).

    This is ALWAYS a bare SSE comment, never a named event, for both
    protocols. A real Anthropic stream's first event is `message_start`; an
    `event: ping` arriving before it would be an event no strict Anthropic
    client has been shown to tolerate, and nobody here has verified one
    does. A comment line (`: ...`) is not an event at all -- it carries no
    `data:` field, so `SSEDecoder._parse` (and every other SSE reader,
    Anthropic's own included) drops it silently. That keeps the wire warm
    -- bytes still arrive, resetting any read-inactivity timeout on the
    client's socket -- without ever telling an Anthropic client's event
    stream that something happened before `message_start` did. OpenAI
    clients already got exactly this treatment; this makes both sides
    consistent instead of privileging one client's timeout over the
    other's protocol strictness.
    """
    return b": ping\n\n"


def _stream_error(req: ShuntRequest, message: str) -> bytes:
    body = error_body(req.protocol, 502, message)
    if req.protocol == "anthropic":
        return _sse("error", body)
    return _data(body)


def _finish(req: ShuntRequest, translator) -> Iterator[bytes]:
    if req.protocol == "anthropic":
        for name, out in translator.finish():
            yield _sse(name, out)
    else:
        for chunk in translator.finish():
            yield _data(chunk)


@dataclass
class _Reading:
    """Mutable state of one candidate's read loop.

    `started` is the commitment: while it is False nothing has reached the
    client and the next candidate is still available; once it is True the
    only remaining outcome is an error event on the wire.
    """

    started: bool = False
    failed: str | None = None  # abandon this candidate; the next one gets a turn
    committed: bool = False  # this candidate answered; do not try another


@dataclass
class _Tally:
    """O que o streaming descobre enquanto corre, para a linha de log.

    Um gerador nao tem valor de retorno que o chamador possa ler, e o log
    precisa de coisas que so a cadeia sabe -- qual regra casou, qual candidato
    ficou, o que cada tentativa custou. Este objeto e o canal de volta.
    """

    rule: str = "unresolved"
    matched: str | None = None
    candidate: str | None = None
    trace: list[str] = field(default_factory=list)
    first_byte_at: float | None = None
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    provider: str | None = None
    answer_text: str = ""
    tools_called: list[str] = field(default_factory=list)
    thinking_blocks: int = 0
    status: int = 200
    error_type: str | None = None


@dataclass
class _Passthrough:
    """Set when a candidate answered by forwarding raw bytes.

    The provider's own stream already carried whatever terminator its dialect
    uses -- an OpenAI provider ends with `data: [DONE]` of its own -- so
    `dispatch_stream` must not append a second one on top of bytes it never
    decoded. Every other exit (a translated stream, an error, an exhausted
    chain) produced no terminator, and still needs ours.
    """

    happened: bool = False


@dataclass
class _Handled:
    """What one upstream SSE event produced, and what it decided."""

    payloads: list[bytes]
    started: bool
    failed: str | None = None  # abandon this candidate; the next one gets a turn
    fatal: bool = False  # already committed: the error went out, stop here


def _handle_event(req: ShuntRequest, translator, event: SSEEvent, started: bool) -> _Handled:
    try:
        data = json.loads(event.data)
    except json.JSONDecodeError:
        # This is also where OpenAI's `data: [DONE]` sentinel lands, and where
        # it belongs: it is not JSON, it decides nothing, and the close of the
        # stream is already driven by the end of the body. An explicit `[DONE]`
        # branch here would be a second spelling of this same `return` -- a
        # line no test could ever tell apart from this one.
        return _Handled([], started)
    if not isinstance(data, dict):
        # Valid JSON of the wrong shape (a list, a bare string). `.get` on it
        # would raise and kill a stream that is otherwise fine.
        return _Handled([], started)
    if data.get("error"):
        message = json.dumps(data["error"], ensure_ascii=False)
        if not started:
            return _Handled([], started, failed=message)
        return _Handled([_stream_error(req, message)], started, fatal=True)

    payloads: list[bytes] = []
    if req.protocol == "anthropic":
        # The provider speaks OpenAI: the chunk itself is what we judge, and
        # it must NOT reach the translator before the decision -- feeding it
        # would mark `message_start` as emitted while we drop the output.
        if not started:
            if not is_first_valid_event(data):
                return _Handled([], started)
            started = True
        payloads = [_sse(name, out) for name, out in translator.feed(data)]
    else:
        # The provider speaks Anthropic: the criterion is the same, but it
        # applies to the already-translated chunk, since an Anthropic event
        # has no `choices`/`delta` of its own to judge.
        for chunk in translator.feed(event.event or "", data):
            if not started:
                if not is_first_valid_event(chunk):
                    continue
                started = True
            payloads.append(_data(chunk))
    return _Handled(payloads, started)


def _drain(
    req: ShuntRequest, translator, events: list[SSEEvent], state: _Reading
) -> Iterator[bytes]:
    """One place where an event becomes bytes and a decision, so the events
    recovered by `SSEDecoder.flush()` are judged exactly like the rest."""
    for event in events:
        handled = _handle_event(req, translator, event, state.started)
        state.started = handled.started
        yield from handled.payloads
        if handled.fatal:
            state.committed = True
            return
        if handled.failed is not None:
            state.failed = handled.failed
            return


def _absorb(tally: _Tally, translator: object, candidate: Candidate) -> None:
    """Passa para a linha de log o que so o tradutor viu.

    Um repasse cru nao tem tradutor, entao ferramenta e raciocinio ficam
    vazios: o proxy nao leu aqueles bytes e inventar contagem seria pior do
    que nao ter.
    """
    tally.provider = candidate.provider
    tools = getattr(translator, "tools_called", None)
    thinking = getattr(translator, "thinking_blocks", None)
    text = getattr(translator, "answer_text", None)
    if callable(tools):
        tally.tools_called = tools()
    if callable(thinking):
        tally.thinking_blocks = thinking()
    if callable(text):
        tally.answer_text = text()


async def _stream_chain(
    req: ShuntRequest,
    settings: Settings,
    pool: UpstreamPool,
    passthrough: _Passthrough,
    tally: _Tally,
) -> AsyncGenerator[bytes]:
    resolution = resolve(req.body.get("model", ""), settings)
    tally.rule, tally.matched = resolution.rule, resolution.matched
    first = resolution.chain[0]
    probe_note: list[str] = []
    try:
        probe = _payload(req, first, settings)
    except Exception as err:  # noqa: BLE001 - same reasoning as `dispatch`: a
        # probe we cannot render is a worse estimate, not a dead request.
        probe = req.body
        probe_note = [f"probe ({first.alias or first.model}): request translation failed: {err}"]
    chain, dropped = filter_chain(resolution.chain, requirements_of(probe), settings)
    trace = tally.trace
    trace.extend(probe_note)
    trace.extend(f"{alias}: {reason}" for alias, reason in dropped)

    if not chain:
        yield _stream_error(req, "no candidate can serve this request: " + "; ".join(trace))
        return

    last_message = "no candidate answered"

    for candidate in chain:
        label = candidate.alias or candidate.model
        if (candidate.protocol, req.endpoint) not in PATHS:
            trace.append(f"{label}: endpoint not supported")
            continue
        missing = _missing_credential(candidate, settings)
        if missing is not None:
            trace.append(f"{label}: credential {missing} is not set")
            continue
        try:
            payload = _payload(req, candidate, settings)
        except Exception as err:  # noqa: BLE001 - one candidate skipped, not a
            # dead chain: the next may speak the client's protocol untranslated.
            trace.append(f"{label}: request translation failed: {err}")
            continue
        client = pool.get(candidate.provider)
        # Retry belongs here and only here: no byte of this candidate has been
        # emitted, so a transient connect error is exactly the buffered case
        # `classify` already answers RETRY for. Past this point the response
        # exists and a stream cannot be replayed.
        request = client.build_request(
            "POST",
            PATHS[(candidate.protocol, req.endpoint)],
            json=payload,
            headers=outbound_headers(req, candidate, settings),
        )
        response = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await client.send(request, stream=True)
            except httpx.HTTPError as err:
                trace.append(f"{label}: {_exception_text(err)} (attempt {attempt})")
                last_message = _exception_text(err)
                response = None
                if classify(None, err, None) is not Outcome.RETRY:
                    break
                if attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(backoff(attempt))
                continue
            if response.status_code < 400:
                break
            # A 429/503/etc. arriving in the response headers, before any byte
            # of the body reached the client, is exactly the buffered case:
            # `classify` already knows which statuses are worth a retry.
            # `aread()` materialises the streamed body, which is what makes
            # `_error_message` (shared with `dispatch`) usable on it -- and it
            # closes the response on its way out, so no `aclose()` follows.
            outcome = classify(response.status_code, None, _retry_after(response))
            await response.aread()
            message = _error_message(response)
            trace.append(f"{label}: {response.status_code} (attempt {attempt})")
            last_message = message
            if outcome is not Outcome.RETRY or attempt >= MAX_ATTEMPTS:
                break
            await asyncio.sleep(backoff(attempt))
        if response is None:
            continue

        if response.status_code >= 400:
            continue

        if candidate.protocol == req.protocol:
            passthrough.happened = True
            tally.candidate = candidate.model
            tally.provider = candidate.provider
            try:
                async for raw in response.aiter_bytes():
                    yield raw
            finally:
                await response.aclose()
            return

        decoder = SSEDecoder()
        translator = _stream_translator(req)
        state = _Reading()
        started_at = _now()
        last_ping = started_at
        try:
            async for raw in response.aiter_bytes():
                now = _now()
                if not state.started:
                    if now - started_at > FIRST_EVENT_DEADLINE:
                        state.failed = f"no valid event within {FIRST_EVENT_DEADLINE}s"
                        break
                    if now - last_ping > PING_INTERVAL:
                        last_ping = now
                        yield _keepalive()
                for payload_bytes in _drain(req, translator, decoder.feed(raw), state):
                    yield payload_bytes
                if state.committed or state.failed is not None:
                    break
            else:
                # Only when the body ended on its own terms. A provider that
                # closes without the final blank line leaves its last event in
                # the decoder, and dropping it does more than lose an event:
                # `started` would stay False and a working provider would be
                # recorded as having answered nothing.
                for payload_bytes in _drain(req, translator, decoder.flush(), state):
                    yield payload_bytes
        except httpx.HTTPError as err:
            # The response already existed, so the `send` handler above never
            # sees this: a read timeout in mid-generation, a provider dropping
            # the connection. Whether it is a fallback or an error on the wire
            # is decided by the same thing everything else here is decided by.
            if state.started:
                # The error event, and then NOTHING else here: leaving
                # `committed` unset drops through to the normal close below,
                # so a stream cut in mid-flight still gets its `_finish()` and
                # the client's parser releases its buffer. That is the case
                # Task 17 made `finish()` idempotent for. An in-band
                # `{"error": ...}` deliberately gets no such close: that is the
                # provider terminating its own stream in its own protocol
                # (Anthropic's API does exactly that), and a `message_stop`
                # after it would tell the client the message completed.
                yield _stream_error(req, _exception_text(err))
            else:
                state.failed = _exception_text(err)
        finally:
            await response.aclose()

        if state.committed:
            tally.candidate = candidate.model
            tally.usage = translator.usage()
            _absorb(tally, translator, candidate)
            return
        if state.failed is None and not state.started:
            state.failed = "stream ended before the first valid event"
        if state.failed is not None:
            trace.append(f"{label}: {state.failed}")
            last_message = state.failed
            continue

        tally.candidate = candidate.model
        for chunk_bytes in _finish(req, translator):
            yield chunk_bytes
        tally.usage = translator.usage()
        _absorb(tally, translator, candidate)
        return

    tally.status = 502
    tally.error_type = "api_error"
    yield _stream_error(req, f"{last_message} - tried: " + "; ".join(trace))


async def dispatch_stream(
    req: ShuntRequest, settings: Settings, pool: UpstreamPool
) -> AsyncIterator[bytes]:
    """Stream one request, falling back while the decision is still open.

    The `[DONE]` sentinel is OpenAI's, and it terminates every OpenAI-facing
    stream we composed ourselves -- success or error alike, since a client
    waiting for it hangs without it. Two exceptions: an Anthropic client must
    never see it (`message_stop` is its terminator), and a raw passthrough
    already carried the provider's own sentinel, so adding ours would make it
    two. A client that disconnects gets neither, because the `async for` below
    raises out and nothing more is produced.
    """
    passthrough = _Passthrough()
    started = time.monotonic()
    tally = _Tally()
    # `aclosing` is the whole client-disconnect story: when the client goes
    # away this generator is closed, and without it the inner generator would
    # only run its `finally` whenever the garbage collector got to it -- while
    # the provider kept generating tokens nobody reads, on the user's bill.
    #
    # The log line lives in a `finally` for the same reason: a client that
    # walks away mid-stream is exactly the request worth having in the log,
    # and it never reaches the end of this function.
    try:
        async with aclosing(_stream_chain(req, settings, pool, passthrough, tally)) as chain:
            async for chunk in chain:
                if tally.first_byte_at is None:
                    tally.first_byte_at = time.monotonic()
                yield chunk
        if req.protocol != "anthropic" and not passthrough.happened:
            yield DONE
    except GeneratorExit:
        # O cliente foi embora no meio. Isso nao e sucesso nem falha do
        # provedor, e registrar como 200 escondia justamente as requisicoes que
        # o harness cancelou -- que sao as que interessam quando alguem
        # pergunta por que a cadeia parou no meio. 499 e a convencao do nginx
        # para "o cliente fechou antes da resposta".
        tally.status = 499
        tally.error_type = "client_disconnected"
        raise
    finally:
        ttft = tally.first_byte_at
        log_request(
            RequestLog(
                request_id=uuid.uuid4().hex,
                requested_model=req.body.get("model", ""),
                rule=tally.rule,
                matched=tally.matched,
                candidate=tally.candidate,
                attempts=tally.trace,
                input_tokens=tally.usage["input_tokens"],
                output_tokens=tally.usage["output_tokens"],
                ttft_ms=int((ttft - started) * 1000) if ttft is not None else None,
                duration_ms=int((time.monotonic() - started) * 1000),
                translated=tally.candidate is not None,
                route=ROUTES.get(req.endpoint, req.endpoint),
                dialect=req.protocol,
                stream=True,
                status=tally.status,
                error_type=tally.error_type,
                provider=tally.provider,
                tools_offered=tools_offered(req),
                tools_called=tally.tools_called,
                thinking_blocks=tally.thinking_blocks,
                body=bodies.capture(
                    bodies.prompt_text(req.body), tally.answer_text, req.body
                ),
            )
        )
