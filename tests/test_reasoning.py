"""
Reasoning ("thinking") model support, tested with no model running.

WHY THESE ARE HERMETIC WHEN THE BUG NEEDED A REAL MODEL TO FIND
---------------------------------------------------------------
The failure was discovered live: qwen3.5:4b returned an empty string while
reporting 300 completion tokens, because a thinking model's monologue is
billed against `max_tokens` and never appears in `message.content`. Finding
that needed the real model. PINNING it does not - the behaviour is entirely
determined by the shape of the response, which a stub can reproduce exactly
and in milliseconds.

That split is worth keeping deliberate. `tests/test_critic_calibration.py`
needs a live model because it measures a model's judgement. This file needs
none, because it measures our handling of a response shape. Only the first
kind belongs behind the `live` marker.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aegis import DebateState, load_settings
from aegis.agents import critic_node
from aegis.llm import EmptyCompletionError, OpenAICompatibleLLM, _reasoning_of


# ---------------------------------------------------------------------------
# Stub transport
# ---------------------------------------------------------------------------


def _message(content="", reasoning=None, reasoning_content=None):
    msg = SimpleNamespace(content=content)
    if reasoning is not None:
        msg.reasoning = reasoning
    if reasoning_content is not None:
        msg.reasoning_content = reasoning_content
    return msg


def _blocking_response(content="", reasoning=None, tokens_out=0):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=_message(content, reasoning),
                                 finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=tokens_out),
    )


class _StubClient:
    """Captures the request and replays a canned response."""

    def __init__(self, response, stream_events=None):
        self.response = response
        self.stream_events = stream_events
        self.last_kwargs: dict = {}
        self.calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.last_kwargs = kwargs
                outer.calls.append(kwargs)
                if kwargs.get("stream"):
                    return iter(outer.stream_events or [])
                if isinstance(outer.response, list):
                    return outer.response.pop(0)
                return outer.response

        self.chat = SimpleNamespace(completions=_Completions())


def _llm(response=None, stream_events=None, **overrides):
    settings = load_settings(provider="ollama")
    settings.stream = stream_events is not None
    for k, v in overrides.items():
        setattr(settings, k, v)
    client = OpenAICompatibleLLM(settings)
    client._client = _StubClient(response, stream_events)
    return client


# ---------------------------------------------------------------------------
# Budget selection
# ---------------------------------------------------------------------------


def test_reasoning_model_gets_the_larger_budget():
    settings = load_settings(provider="ollama")
    settings.max_tokens = 500
    settings.reasoning_max_tokens = 2048
    settings.reasoning_models = {"qwen3.5:4b"}
    assert settings.budget_for("gemma2:2b") == 500
    assert settings.budget_for("qwen3.5:4b") == 2048


def test_a_hand_raised_max_tokens_is_never_lowered():
    """
    `budget_for` raises a floor, it does not impose a ceiling. If someone has
    already asked for more than the reasoning default, honour it.
    """
    settings = load_settings(provider="ollama")
    settings.max_tokens = 4000
    settings.reasoning_max_tokens = 2048
    settings.reasoning_models = {"qwen3.5:4b"}
    assert settings.budget_for("qwen3.5:4b") == 4000


def test_the_larger_budget_actually_reaches_the_request():
    """A budget the transport never sees is a setting, not a fix."""
    llm = _llm(_blocking_response(content="hello", tokens_out=5),
               reasoning_models={"qwen3.5:4b"}, max_tokens=500,
               reasoning_max_tokens=2048)
    llm.complete([{"role": "user", "content": "hi"}], model="qwen3.5:4b",
                 max_tokens=500)
    assert llm._client.last_kwargs["max_tokens"] == 2048


# ---------------------------------------------------------------------------
# Failing loudly instead of returning nothing
# ---------------------------------------------------------------------------


def test_all_budget_spent_thinking_raises_an_actionable_error():
    """
    THE REGRESSION THAT MATTERED.

    Returning "" here is far worse than it looks. The empty answer flows into
    the Critic, which correctly cannot approve it, so the debate runs to the
    round cap and reports a confident "contested" verdict about nothing. The
    system looks like it is working. The error message therefore has to name
    both the cause and the fix, because the symptom points at neither.
    """
    # tokens_out == budget: the model genuinely ran out of room. Spelled out
    # rather than left near the limit, because "ran out" and "gave up" now
    # produce different advice and a test should pin one of them, not
    # whichever the thresholds happen to pick.
    llm = _llm(_blocking_response(content="", reasoning="Thinking..." * 40,
                                  tokens_out=300),
               reasoning_models={"qwen3.5:4b"}, reasoning_max_tokens=300,
               max_tokens=300)
    with pytest.raises(EmptyCompletionError) as exc:
        # max_tokens passed explicitly: `complete` takes the LARGER of the
        # caller's request and the model's floor, so leaving it at the 1200
        # default would silently raise the budget above the 300 this case is
        # about.
        llm.complete([{"role": "user", "content": "hi"}], model="qwen3.5:4b",
                     max_tokens=300)
    message = str(exc.value)
    assert "reasoning" in message.lower()
    assert "AEGIS_REASONING_MAX_TOKENS" in message


def test_an_empty_response_with_no_reasoning_reports_a_different_cause():
    """Two causes, two messages. A merged one would misdiagnose both."""
    llm = _llm(_blocking_response(content="", tokens_out=0))
    with pytest.raises(EmptyCompletionError) as exc:
        llm.complete([{"role": "user", "content": "hi"}], model="gemma2:2b")
    assert "AEGIS_REASONING_MAX_TOKENS" not in str(exc.value)


def test_local_reasoning_model_retries_once_without_hidden_thinking():
    """A Qwen-style thought-only turn should recover into a usable reply."""
    first = _len_capped_response("thinking " * 500, 4096)
    retry = _blocking_response(content="VERDICT: REVISE\nREASONS:\n- P1: fix it",
                               tokens_out=18)
    llm = _llm([first, retry], reasoning_models={"qwen3.5:4b"},
               reasoning_max_tokens=4096, max_tokens=500)

    out = llm.complete([{"role": "user", "content": "hi"}],
                       model="qwen3.5:4b", max_tokens=500)

    assert out.text.startswith("VERDICT: REVISE")
    assert out.tokens_out == 4114
    assert len(llm._client.calls) == 2
    assert llm._client.calls[1]["max_tokens"] == 500
    assert llm._client.calls[1]["extra_body"]["reasoning_effort"] == "none"


def test_a_normal_reply_is_untouched():
    llm = _llm(_blocking_response(content="the answer", tokens_out=3))
    out = llm.complete([{"role": "user", "content": "hi"}], model="gemma2:2b")
    assert out.text == "the answer"
    assert out.reasoning == ""


# ---------------------------------------------------------------------------
# Channel separation
# ---------------------------------------------------------------------------


def test_reasoning_never_leaks_into_the_answer_text():
    """
    The Critic's verdict is parsed from the first line of its reply. If a
    paragraph of reasoning were prepended, `parse_verdict` would fail to match
    and default to REVISE - a routing change caused entirely by a rendering
    decision. Hence two channels.
    """
    llm = _llm(_blocking_response(
        content="VERDICT: APPROVE\nREASONS:\n- Fine.",
        reasoning="Let me consider whether this is any good...", tokens_out=20))
    out = llm.complete([{"role": "user", "content": "hi"}], model="qwen3.5:4b")
    assert out.text.startswith("VERDICT: APPROVE")
    assert "Let me consider" not in out.text
    assert "Let me consider" in out.reasoning


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_both_provider_spellings_of_the_thinking_field_are_read(field):
    """Ollama says `reasoning`; others say `reasoning_content`."""
    assert _reasoning_of(_message("x", **{field: "thought"})) == "thought"


def test_a_message_with_no_thinking_channel_yields_empty_string():
    assert _reasoning_of(_message("x")) == ""


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _chunk(content=None, reasoning=None, usage=None):
    delta = SimpleNamespace(content=content)
    if reasoning is not None:
        delta.reasoning = reasoning
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=usage)


def test_streaming_splits_thinking_onto_its_own_channel():
    events = [
        _chunk(reasoning="I should check "),
        _chunk(reasoning="the claim. "),
        _chunk(content="VERDICT: APPROVE"),
        _chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=40)),
    ]
    seen: list[tuple[str, str]] = []
    llm = _llm(stream_events=events)
    out = llm.complete([{"role": "user", "content": "hi"}], model="qwen3.5:4b",
                       agent="critic", on_token=lambda ch, t: seen.append((ch, t)))

    assert out.text == "VERDICT: APPROVE"
    assert out.reasoning == "I should check the claim."
    channels = {ch for ch, _ in seen}
    assert channels == {"critic", "critic:thinking"}


def test_ttft_counts_the_first_thinking_token_not_the_first_word():
    """
    Measuring ttft from the first CONTENT token made a reasoning model report
    237 tok/s: the thinking time landed inside "ttft", so the generation
    window collapsed to the visible reply while the token count still
    included every thinking token. ttft means "when did the model start
    working", and thinking is working.
    """
    events = [
        _chunk(reasoning="thinking..."),
        _chunk(content="answer"),
        _chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1000)),
    ]
    out = _llm(stream_events=events).complete(
        [{"role": "user", "content": "hi"}], model="qwen3.5:4b")
    # 1000 tokens cannot have been produced in the sliver after the reply
    # started; a sane rate proves ttft was taken at the thinking token and
    # that a sub-millisecond window is refused rather than divided by.
    assert out.tokens_per_s < 100_000, "implausible rate from a tiny window"
    assert out.ttft_s <= out.latency_s


# ---------------------------------------------------------------------------
# The turn records it
# ---------------------------------------------------------------------------


def test_a_turn_keeps_the_thinking_that_produced_it():
    settings = load_settings(provider="ollama")
    settings.stream = False
    llm = _llm(_blocking_response(
        content="VERDICT: REVISE\nREASONS:\n- Wrong.",
        reasoning="Weighing it up...", tokens_out=30))
    state = DebateState(topic="t", answer="an answer")
    updates = critic_node(state, llm=llm, settings=settings)
    turn = updates["transcript"][0]
    assert turn.reasoning == "Weighing it up..."
    assert updates["verdict"] == "REVISE"


# ---------------------------------------------------------------------------
# Context sizing
# ---------------------------------------------------------------------------


def test_reasoning_model_gets_a_larger_context_not_just_a_larger_budget():
    """
    num_ctx bounds prompt PLUS generation. Raising only the token budget
    produces a model allowed to write more than it can remember - and the
    intuition that leads there is sizing context from the reply length.
    """
    settings = load_settings(provider="ollama")
    settings.num_ctx = 4096
    settings.reasoning_num_ctx = 8192
    settings.reasoning_models = {"qwen3.5:4b"}
    assert settings.ctx_for("gemma2:2b") == 4096
    assert settings.ctx_for("qwen3.5:4b") == 8192


def test_the_larger_context_reaches_the_request():
    llm = _llm(_blocking_response(content="ok", tokens_out=2),
               reasoning_models={"qwen3.5:4b"}, num_ctx=4096,
               reasoning_num_ctx=8192)
    llm.complete([{"role": "user", "content": "hi"}], model="qwen3.5:4b")
    assert llm._client.last_kwargs["extra_body"] == {"options": {"num_ctx": 8192}}


def test_a_plain_model_keeps_the_ordinary_context():
    llm = _llm(_blocking_response(content="ok", tokens_out=2),
               reasoning_models={"qwen3.5:4b"}, num_ctx=4096,
               reasoning_num_ctx=8192)
    llm.complete([{"role": "user", "content": "hi"}], model="gemma2:2b")
    assert llm._client.last_kwargs["extra_body"] == {"options": {"num_ctx": 4096}}


# ---------------------------------------------------------------------------
# Diagnosing the two ways an empty reply happens
# ---------------------------------------------------------------------------


def _len_capped_response(reasoning, tokens_out):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=_message("", reasoning),
                                 finish_reason="length")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=tokens_out),
    )


def test_running_out_of_budget_advises_raising_the_budget():
    llm = _llm(_len_capped_response("thinking " * 500, 4096),
               reasoning_models={"qwen3.5:4b"}, reasoning_max_tokens=4096)
    with pytest.raises(EmptyCompletionError) as exc:
        llm.complete([{"role": "user", "content": "hi"}], model="qwen3.5:4b")
    message = str(exc.value)
    assert "ran out of room" in message
    assert "AEGIS_REASONING_MAX_TOKENS" in message


def test_stopping_early_does_not_advise_raising_the_budget():
    """
    Observed on qwen3.5:4b: 3208 tokens of pure thought, then finish_reason
    "stop" — 888 tokens SHORT of the limit. The first version of this message
    told the user to raise a budget the model had not reached, which is a
    diagnostic that costs more than none.
    """
    llm = _llm(_blocking_response(content="", reasoning="thinking " * 400,
                                  tokens_out=3208),
               reasoning_models={"qwen3.5:4b"}, reasoning_max_tokens=4096)
    with pytest.raises(EmptyCompletionError) as exc:
        llm.complete([{"role": "user", "content": "hi"}], model="qwen3.5:4b")
    message = str(exc.value)
    assert "stopped on its own" in message
    assert "888 tokens short" in message
    assert "will not help" in message


def test_streaming_reports_the_finish_reason():
    events = [
        _chunk(content="hi"),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None),
                                                finish_reason="length")],
                        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=9)),
    ]
    out = _llm(stream_events=events).complete(
        [{"role": "user", "content": "hi"}], model="gemma2:2b")
    assert out.finish_reason == "length"
