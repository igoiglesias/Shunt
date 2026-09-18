import httpx

from app.core.attempt import MAX_ATTEMPTS, RETRY_AFTER_BUDGET, Outcome, backoff, classify


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


def test_backoff_stays_bounded_well_past_max_attempts():
    assert backoff(10) <= 6.0


def test_backoff_is_not_deterministic_across_calls():
    samples = {backoff(2) for _ in range(20)}
    assert len(samples) > 1
