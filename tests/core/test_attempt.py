import httpx

from app.config.config import MAX_ATTEMPTS, RETRY_AFTER_BUDGET
from app.core import attempt
from app.core.attempt import Outcome, backoff, classify


def test_transport_error_is_retried():
    assert classify(None, httpx.ConnectError("recusou"), None) is Outcome.RETRY


def test_server_error_is_retried():
    assert classify(500, None, None) is Outcome.RETRY
    assert classify(529, None, None) is Outcome.RETRY


def test_bad_request_and_payment_skip_to_the_next_candidate():
    assert classify(400, None, None) is Outcome.SKIP
    assert classify(402, None, None) is Outcome.SKIP


def test_rate_limit_waits_once_when_retry_after_is_short():
    assert classify(429, None, 3.0) is Outcome.RETRY


def test_rate_limit_skips_when_retry_after_is_long_or_absent():
    assert classify(429, None, 30.0) is Outcome.SKIP
    assert classify(429, None, None) is Outcome.SKIP


def test_success_is_ok():
    assert classify(200, None, None) is Outcome.OK


def test_backoff_grows_and_stays_bounded():
    assert backoff(1) < backoff(2) < backoff(3)
    assert backoff(MAX_ATTEMPTS) <= 8.0


def test_rate_limit_retries_when_retry_after_exactly_at_budget():
    assert classify(429, None, RETRY_AFTER_BUDGET) is Outcome.RETRY


def test_rate_limit_skips_when_retry_after_is_not_a_number():
    assert classify(429, None, float("nan")) is Outcome.SKIP


def test_other_four_xx_statuses_skip():
    for status in (401, 403, 404, 409):
        assert classify(status, None, None) is Outcome.SKIP


def test_request_timeout_is_retried_like_a_transient_failure():
    assert classify(408, None, None) is Outcome.RETRY


def test_redirect_status_is_ok():
    assert classify(302, None, None) is Outcome.OK


def test_non_transport_exception_skips():
    assert classify(None, ValueError("boom"), None) is Outcome.SKIP


def test_no_status_and_no_exception_skips():
    # `classify` is called from `dispatch` as
    # `classify(response.status_code if response else None, exc, ...)`, where
    # exactly one of `response`/`exc` is ever set. But `classify` is a public,
    # exception-less pure function -- its own contract for the "nothing to
    # go on" input must hold regardless of what its only caller happens to
    # pass. Flipping `status is None: return SKIP` to `return RETRY` should
    # fail here even though it can never fire through `dispatch` today.
    assert classify(None, None, None) is Outcome.SKIP


def test_backoff_stays_bounded_well_past_max_attempts():
    assert backoff(10) <= 6.0


def test_backoff_is_not_deterministic_across_calls():
    samples = {backoff(2) for _ in range(20)}
    assert len(samples) > 1


def test_backoff_floor_is_exactly_the_exponential_base(monkeypatch):
    # The growth/bound tests above only compare backoff() across attempts and
    # against an upper bound; both use real jitter, so they cannot catch a
    # mutation that drops the `base +` term (e.g. `return
    # random.uniform(0, base / 2)`), which shrinks the floor from `base` to 0
    # while still growing and staying bounded. Pinning `random.uniform` to
    # always return 0 makes the exponential base itself the assertion.
    monkeypatch.setattr(attempt.random, "uniform", lambda _lo, _hi: 0.0)
    assert backoff(1) == 1.0
    assert backoff(2) == 2.0
    assert backoff(3) == 4.0
    # The cap: attempt 10 would be 2**9 uncapped, but `min(..., 4.0)` holds it.
    assert backoff(10) == 4.0
