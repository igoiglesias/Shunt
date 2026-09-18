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

TRANSPARENT_DROP = frozenset({"host", "content-length", "accept-encoding"})

DEFAULT_MAX_OUTPUT_TOKENS = 4096


class ChainExhausted(Exception):
    pass


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
        return {k: v for k, v in req.headers.items() if k.lower() not in TRANSPARENT_DROP}
    key = settings.api_key(candidate.provider)
    headers = {"content-type": "application/json"}
    if key and candidate.protocol == "anthropic":
        headers["x-api-key"] = key
        headers["anthropic-version"] = "2023-06-01"
    elif key:
        headers["authorization"] = f"Bearer {key}"
    return headers


def _payload(req: ShuntRequest, candidate: Candidate, settings: Settings) -> dict:
    if candidate.transparent or candidate.protocol == req.protocol:
        return {**req.body, "model": candidate.model}
    cap = (
        settings.models[candidate.alias].max_output_tokens
        if candidate.alias
        else DEFAULT_MAX_OUTPUT_TOKENS
    )
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
            return str((body.get("error") or {}).get("message", response.text))
    return response.text


async def dispatch(req: ShuntRequest, settings: Settings, pool: UpstreamPool) -> ShuntResult:
    resolution = resolve(req.body.get("model", ""), settings)
    probe = _payload(req, resolution.chain[0], settings)
    chain, dropped = filter_chain(resolution.chain, requirements_of(probe), settings)
    trace = [f"{alias}: {reason}" for alias, reason in dropped]

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
        payload = _payload(req, candidate, settings)
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
                data = _translate_response(response.json(), req, candidate)
                return ShuntResult(200, data, candidate.model, trace)
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
