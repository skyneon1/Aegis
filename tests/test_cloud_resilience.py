"""
Recovery paths for remote providers (aegis/llm.py).

Both behaviours here were found by pointing a real key at a real free tier and
watching a debate die, twice, for two different reasons. Both are reproduced
hermetically: the transport is stubbed, so these run in milliseconds and need
no key.

WHY THESE ARE NOT "NICE TO HAVE"
--------------------------------
A debate is six or more calls. Any per-call failure mode with probability p
fails the RUN at roughly 6p, so a 5%-flaky call is a 26%-flaky debate. On a
free tier — which is oversubscribed by construction and the whole reason the
cloud providers were added — that is the difference between a feature and a
demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from aegis import load_settings  # noqa: E402
from aegis.llm import EmptyCompletionError, LLMResponse, OpenAICompatibleLLM  # noqa: E402
from aegis.llm import _is_transient  # noqa: E402


def _client(**over):
    s = load_settings(provider="groq", **over)
    s.api_key = "test-key"          # never used: _dispatch is stubbed out
    s.stream = False
    return OpenAICompatibleLLM(s), s


class _Boom(Exception):
    """Stands in for an openai SDK error carrying an HTTP status."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Which failures are worth a second attempt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_server_side_failures_are_transient(status):
    assert _is_transient(_Boom(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_request_side_failures_are_not_retried(status):
    """
    A 401 will fail identically forever. Retrying it only delays a clear
    error and triples the time the user waits to be told to fix their key.
    """
    assert _is_transient(_Boom(status)) is False


def test_connection_failures_are_transient_even_without_a_status():
    """These never reach a status code, and are the classic retry case."""
    for name in ("APIConnectionError", "APITimeoutError", "RateLimitError"):
        exc = type(name, (Exception,), {})()
        assert _is_transient(exc) is True, name


# ---------------------------------------------------------------------------
# Transient retry
# ---------------------------------------------------------------------------


def test_an_overloaded_provider_is_retried_and_succeeds(monkeypatch):
    """
    The exact failure that killed round two of a live debate:
    "Upstream error from Nvidia: Service temporarily overloaded".
    """
    llm, _ = _client()
    monkeypatch.setattr("time.sleep", lambda _s: None)   # no real backoff

    calls = []

    def flaky(*a, **kw):
        calls.append(1)
        if len(calls) < 3:
            raise _Boom(503)
        return LLMResponse(text="recovered", model="m", finish_reason="stop")

    monkeypatch.setattr(llm, "_dispatch", flaky)

    out = llm.complete([{"role": "user", "content": "hi"}], model="m")
    assert out.text == "recovered"
    assert len(calls) == 3, "should have retried twice before succeeding"


def test_retries_are_bounded(monkeypatch):
    """
    A provider that is down must not turn one debate into an infinite one.
    """
    llm, _ = _client()
    monkeypatch.setattr("time.sleep", lambda _s: None)

    calls = []

    def always_down(*a, **kw):
        calls.append(1)
        raise _Boom(503)

    monkeypatch.setattr(llm, "_dispatch", always_down)

    with pytest.raises(_Boom):
        llm.complete([{"role": "user", "content": "hi"}], model="m")
    assert len(calls) == llm._MAX_TRANSIENT_RETRIES + 1


def test_a_bad_key_fails_immediately_without_retrying(monkeypatch):
    llm, _ = _client()
    calls = []

    def unauthorised(*a, **kw):
        calls.append(1)
        raise _Boom(401)

    monkeypatch.setattr(llm, "_dispatch", unauthorised)

    with pytest.raises(_Boom):
        llm.complete([{"role": "user", "content": "hi"}], model="m")
    assert len(calls) == 1, "an auth failure must not be retried"


# ---------------------------------------------------------------------------
# Reasoning models that do not advertise themselves
# ---------------------------------------------------------------------------


def test_a_hidden_remote_reasoner_is_detected_and_retried(monkeypatch):
    """
    `nex-agi/nex-n2.5-pro` spent its entire 1200-token budget thinking and
    returned "". Its id contains no marker a name heuristic could catch, and
    a remote /models listing cannot be asked — so the only available evidence
    is the response itself.
    """
    llm, settings = _client()
    settings.max_tokens = 500
    settings.reasoning_max_tokens = 4096

    budgets = []

    def thinks_too_long(*a, **kw):
        budgets.append(kw["max_tokens"])
        if len(budgets) == 1:
            # All budget spent privately; nothing visible came back.
            return LLMResponse(text="", model="m", reasoning="thinking...",
                               finish_reason="length", tokens_out=500)
        return LLMResponse(text="a real answer", model="m", finish_reason="stop")

    monkeypatch.setattr(llm, "_dispatch", thinks_too_long)

    out = llm.complete([{"role": "user", "content": "hi"}], model="m",
                       max_tokens=500, agent="critic")

    assert out.text == "a real answer"
    assert budgets[1] > budgets[0], "the retry must get a bigger budget"


def test_the_discovery_is_remembered_for_later_turns(monkeypatch):
    """
    Paying the failed first call once is acceptable; paying it on every turn
    of every round is not. The model is recorded on the Settings so the next
    turn is sized correctly up front.
    """
    llm, settings = _client()
    settings.max_tokens = 500
    settings.reasoning_max_tokens = 4096
    assert "m" not in settings.reasoning_models

    def thinks_too_long(*a, **kw):
        if kw["max_tokens"] <= 500:
            return LLMResponse(text="", model="m", reasoning="thinking...",
                               finish_reason="length", tokens_out=500)
        return LLMResponse(text="answer", model="m", finish_reason="stop")

    monkeypatch.setattr(llm, "_dispatch", thinks_too_long)
    llm.complete([{"role": "user", "content": "hi"}], model="m", max_tokens=500)

    assert "m" in settings.reasoning_models
    assert settings.budget_for("m") == 4096


def test_an_empty_reply_with_no_reasoning_still_fails_loudly(monkeypatch):
    """
    The retry is for a model that thought too long. A model that simply
    returned nothing has a different problem, and must not be papered over —
    see EmptyCompletionError.
    """
    llm, _ = _client()

    monkeypatch.setattr(llm, "_dispatch", lambda *a, **kw: LLMResponse(
        text="", model="m", reasoning="", finish_reason="stop"))

    with pytest.raises(EmptyCompletionError):
        llm.complete([{"role": "user", "content": "hi"}], model="m")


# ---------------------------------------------------------------------------
# Rate limits come in two kinds
# ---------------------------------------------------------------------------


class _Limited(Exception):
    """A 429 carrying the provider's own explanation."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 429


def test_a_per_minute_rate_limit_is_retried():
    """Backoff is exactly the right response to a short-window limit."""
    assert _is_transient(_Limited("Rate limit exceeded: 20 per minute")) is True


def test_an_exhausted_daily_quota_is_not_retried():
    """
    The failure that prompted this: OpenRouter's free tier answers
    'Rate limit exceeded: free-models-per-day' with X-RateLimit-Remaining: 0.
    Backing off two more times spends two more requests from a balance
    already at zero and delays the provider's own remedy ("wait for the
    daily reset, or purchase credits") by six seconds.
    """
    exc = _Limited("Rate limit exceeded: free-models-per-day. Add 10 credits "
                   "to unlock 1000 free model requests per day")
    assert _is_transient(exc) is False


def test_a_billing_failure_is_not_retried():
    assert _is_transient(_Limited("insufficient_quota: check your billing")) is False


def test_the_carve_out_also_applies_without_a_status_code():
    """
    The SDK's RateLimitError does not always carry status_code, and the
    name-based fallback must not re-open the hole the check above closes.
    """
    exc = type("RateLimitError", (Exception,), {})("exceeded your daily quota")
    assert _is_transient(exc) is False

    fine = type("RateLimitError", (Exception,), {})("slow down")
    assert _is_transient(fine) is True
