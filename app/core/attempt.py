"""Pure policy for one upstream attempt: retry, skip, or accept.

This module performs no I/O and holds no state. It only classifies the
outcome of a single upstream call; the dispatcher (next task) owns the
retry loop and interprets these decisions.
"""

import random
from enum import Enum

import httpx

from app.config.config import (
    RETRY_AFTER_BUDGET,
)


class Outcome(Enum):
    OK = "ok"
    RETRY = "retry"
    SKIP = "skip"


def classify(status: int | None, exc: Exception | None, retry_after: float | None) -> Outcome:
    if exc is not None:
        # ReadTimeout pula: o servidor aceitou a conexao e nao respondeu dentro
        # do TIMEOUT_READ -- medido num llama.cpp saturado. Repetir no MESMO
        # servidor repete a mesma espera e gasta o prazo total da cadeia.
        if isinstance(exc, httpx.ReadTimeout):
            return Outcome.SKIP
        return Outcome.RETRY if isinstance(exc, httpx.TransportError) else Outcome.SKIP
    if status is None:
        return Outcome.SKIP
    if status < 400:
        return Outcome.OK
    if status == 429:
        if retry_after is not None and retry_after <= RETRY_AFTER_BUDGET:
            return Outcome.RETRY
        return Outcome.SKIP
    if status == 408:
        return Outcome.RETRY
    if status >= 500:
        return Outcome.RETRY
    return Outcome.SKIP


def backoff(attempt: int) -> float:
    base = min(2.0 ** (attempt - 1), 4.0)
    return base + random.uniform(0, base / 2)
