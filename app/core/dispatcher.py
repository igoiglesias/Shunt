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
import time
from dataclasses import dataclass, field

import httpx

from app.config.settings import Settings
from app.core.attempt import MAX_ATTEMPTS, TOTAL_DEADLINE, Outcome, backoff, classify
from app.core.capabilities import filter_chain, requirements_of
from app.core.resolver import Candidate, resolve
from app.core.upstream import UpstreamPool
from app.translate.to_anthropic import openai_error_to_anthropic, openai_response_to_anthropic
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


async def dispatch(req: ShuntRequest, settings: Settings, pool: UpstreamPool) -> ShuntResult:
    resolution = resolve(req.body.get("model", ""), settings)
    probe_note: list[str] = []
    try:
        probe = _payload(req, resolution.chain[0], settings)
    except Exception as err:  # noqa: BLE001 - a translator that cannot render the
        # probe must not decide the whole request. The untranslated body is a
        # worse estimate, not a fatal one, and a later candidate may take it
        # as-is. It is still recorded: if `chain[0]` is then dropped by the
        # capability filter, this is the only evidence the probe ever failed.
        first = resolution.chain[0]
        probe = req.body
        probe_note = [
            f"probe ({first.alias or first.model}): request translation failed: {err}"
        ]
    chain, dropped = filter_chain(resolution.chain, requirements_of(probe), settings)
    trace = probe_note + [f"{alias}: {reason}" for alias, reason in dropped]

    if not chain:
        return ShuntResult(400, openai_error_to_anthropic(
            400, "no candidate can serve this request: " + "; ".join(trace)), None, trace)

    deadline = time.monotonic() + TOTAL_DEADLINE
    last_status, last_message = 502, "no candidate answered"

    for candidate in chain:
        label = candidate.alias or candidate.model
        if (candidate.protocol, req.endpoint) not in PATHS:
            trace.append(f"{label}: endpoint not supported")
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
                    PATHS[(candidate.protocol, req.endpoint)], json=payload,
                    headers=outbound_headers(req, candidate, settings))
            except httpx.HTTPError as err:
                exc = err
            outcome = classify(
                response.status_code if response else None, exc, _retry_after(response))
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
                return ShuntResult(response.status_code, data, candidate.model, trace)
            if response is not None:
                last_status, last_message = response.status_code, _error_message(response)
            else:
                last_status, last_message = 502, str(exc)
            trace.append(f"{label}: {last_status} (attempt {attempt})")
            if outcome is Outcome.SKIP:
                break
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(backoff(attempt))

    message = f"{last_message} - tried: " + "; ".join(trace)
    return ShuntResult(last_status, openai_error_to_anthropic(last_status, message), None, trace)
