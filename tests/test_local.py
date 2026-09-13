"""
Tests for local-inference support: keyless providers, zero-cost accounting,
the wall-clock guard, the decision log, and declared agent roles.

WHY THESE ARE SEPARATE FROM test_graph.py
-----------------------------------------
test_graph.py proves the debate machine is correct. This file proves the
machine survives a change of SUBSTRATE - inference moving from a metered
API onto local hardware. Those are different claims, and the second one
is where the interesting bugs were: every assumption the cloud path had
baked in silently ("a key always exists", "tokens always cost money")
became a defect the moment the substrate changed.

Run:  .venv/bin/pytest tests/test_local.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aegis import (
    DebateState,
    FakeLLM,
    PROVIDERS,
    build_debate_graph,
    load_settings,
    route_after_critic,
)
from aegis.graph import END
from aegis.llm import OpenAICompatibleLLM

APPROVE = "VERDICT: APPROVE\nREASONS:\n- Fine.\n"
REVISE = "VERDICT: REVISE\nREASONS:\n- Unsupported claim.\n"


# ---------------------------------------------------------------------------
# Provider declarations
# ---------------------------------------------------------------------------


def test_every_provider_declares_whether_it_needs_a_key():
    """
    A missing declaration would silently inherit the cloud assumption and
    make that provider unusable, which is exactly the bug this replaced.
    """
    for name, preset in PROVIDERS.items():
        assert "requires_key" in preset, f"{name} does not declare requires_key"
        assert "local" in preset, f"{name} does not declare local"


def test_ollama_needs_no_credential_and_is_marked_local():
    s = load_settings(provider="ollama")
    assert s.requires_key is False
    assert s.local is True


def test_local_client_constructs_without_a_real_api_key():
    """
    The regression that mattered most: the constructor used to raise
    whenever api_key was empty, making the entire local path unreachable
    before a single token was generated.
    """
    s = load_settings(provider="ollama")
    s.api_key = ""                      # simulate a user with no .env at all
    s.requires_key = False
    OpenAICompatibleLLM(s)              # must not raise


def test_cloud_client_still_demands_a_key():
    """The fix must not weaken the check where it is genuinely needed."""
    s = load_settings(provider="openrouter")
    s.api_key = ""
    with pytest.raises(RuntimeError, match="No API key"):
        OpenAICompatibleLLM(s)


def test_ollama_default_model_is_one_that_fits_this_hardware():
    """
    The preset pointed at qwen2.5:7b-instruct, whose weights alone exceed
    4GB of VRAM. A default that cannot run is not a default.
    """
    s = load_settings(provider="ollama")
    assert "7b" not in s.proposer_model.lower()
    assert "7b" not in s.critic_model.lower()


# ---------------------------------------------------------------------------
# Cost: local inference is free, and saying otherwise breaks a guard
# ---------------------------------------------------------------------------


def test_local_inference_reports_zero_cost():
    s = load_settings(provider="ollama")
    llm = OpenAICompatibleLLM(s)
    assert llm._price(100_000, 100_000) == 0.0


def test_remote_inference_still_estimates_cost():
    s = load_settings(provider="openrouter")
    s.api_key = "sk-test"
    llm = OpenAICompatibleLLM(s)
    assert llm._price(1_000_000, 1_000_000) > 0.0


# ---------------------------------------------------------------------------
# The wall-clock guard
# ---------------------------------------------------------------------------


def test_time_guard_fires_when_elapsed_reaches_the_limit():
    state = DebateState(topic="t", max_rounds=99, max_seconds=10.0)
    state.apply({"verdict": "REVISE", "round": 1, "elapsed_s": 11.0})
    assert state.max_seconds_exceeded() is True
    assert route_after_critic(state) == END


def test_zero_disables_the_time_guard():
    state = DebateState(topic="t", max_seconds=0.0)
    state.apply({"elapsed_s": 99_999.0})
    assert state.max_seconds_exceeded() is False


def test_time_guard_beats_the_arbiter_branch():
    """
    The load-bearing ordering, mirroring
    test_budget_guard_beats_the_arbiter_branch in test_graph.py.

    Escalating to the Arbiter costs a model call, and a model call costs
    TIME. A guard the escalation path can outrun is not a guard. Both
    conditions fire here at once: out of time AND deadlocked. Time must
    win.
    """
    state = DebateState(topic="t", max_rounds=1, verdict="REVISE",
                        max_seconds=5.0)
    state.apply({"round": 1, "elapsed_s": 6.0})
    assert route_after_critic(state, use_arbiter=True) == END
    assert state.stop_reason == "max_rounds"
    assert "Wall-clock" in state.error


def test_approval_still_beats_the_time_guard():
    """
    The happy path stays cheapest. If the critic approved, we are done -
    reporting a timeout on a run that actually succeeded would be a lie
    about the outcome.
    """
    state = DebateState(topic="t", max_rounds=3, verdict="APPROVE",
                        max_seconds=1.0)
    state.apply({"round": 1, "elapsed_s": 99.0})
    assert route_after_critic(state) == END
    assert state.stop_reason == "approved"


def test_elapsed_time_accumulates_across_turns():
    state = DebateState(topic="t")
    state.apply({"elapsed_s": 1.5})
    state.apply({"elapsed_s": 2.5})
    assert state.elapsed_s == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# The decision log — how the UI answers "why did it stop?"
# ---------------------------------------------------------------------------


def _run(script, *, max_rounds=3, use_arbiter=True, max_seconds=0.0,
         latency_s=0.0):
    settings = load_settings(provider="fake")
    settings.max_rounds = max_rounds
    settings.use_arbiter = use_arbiter
    graph = build_debate_graph(FakeLLM(script=script, latency_s=latency_s),
                               settings)
    state = DebateState(topic="t", max_rounds=max_rounds,
                        max_seconds=max_seconds, max_cost_usd=0.0)
    return graph.invoke(state)


def test_every_routing_decision_is_recorded():
    state = _run(["draft", REVISE, "revised", APPROVE], max_rounds=3)
    rules = [d.rule for d in state.decisions]
    assert rules == ["revise", "approved"]


def test_a_decision_records_the_values_it_was_made_on():
    """
    A reason without its inputs is an assertion, not evidence. The
    `observed` dict is what makes a decision auditable six weeks later.
    """
    state = _run(["draft", REVISE, "revised", APPROVE], max_rounds=3)
    first = state.decisions[0]
    assert first.at == "critic"
    assert first.next_node == "proposer"
    assert first.verdict == "REVISE"
    assert first.observed["rounds_remaining"] == 2
    assert first.reason


def test_deadlock_decision_names_the_arbiter_as_the_destination():
    state = _run(["draft", REVISE, "ruling"], max_rounds=1, use_arbiter=True)
    deadlock = [d for d in state.decisions if d.rule == "deadlock"]
    assert len(deadlock) == 1
    assert deadlock[0].next_node == "arbiter"
    assert state.stop_reason == "arbitrated"


def test_timeout_decision_is_distinguishable_from_a_budget_decision():
    """
    Two guards, two rules, deliberately not merged. "Ran out of money" and
    "ran out of time" call for completely different responses from whoever
    reads the transcript.
    """
    # `latency_s` matters here: elapsed_s accumulates from REAL turn
    # latencies, so a zero-latency fake model can never trip a time guard
    # no matter how low the limit is set. Which is correct - and worth
    # knowing, because the first draft of this test set max_seconds to a
    # microsecond against an instant model and asserted a timeout that
    # could not physically happen.
    state = _run(["draft", REVISE, "x", REVISE, "y", REVISE],
                 max_rounds=5, max_seconds=0.05, latency_s=0.03)
    rules = [d.rule for d in state.decisions]
    assert "timeout" in rules
    assert "budget" not in rules
    assert state.stop_reason == "max_rounds"


def test_decisions_survive_serialisation():
    state = _run(["draft", REVISE, "revised", APPROVE])
    payload = state.to_dict()
    assert isinstance(payload["decisions"], list)
    assert payload["decisions"][0]["rule"] == "revise"


# ---------------------------------------------------------------------------
# Declared roles and captured prompts
# ---------------------------------------------------------------------------


def test_nodes_declare_their_role_to_the_llm():
    """
    Three separate bugs in this project came from inferring an agent's
    identity out of prompt text. The caller always knows its own role;
    passing it explicitly removes the entire failure class.
    """
    settings = load_settings(provider="fake")
    settings.max_rounds = 1
    llm = FakeLLM(script=["draft", REVISE, "ruling"])
    graph = build_debate_graph(llm, settings)
    graph.invoke(DebateState(topic="t", max_rounds=1, max_cost_usd=0.0))
    assert [c["agent"] for c in llm.calls] == ["proposer", "critic", "arbiter"]


def test_declared_role_wins_over_prompt_sniffing():
    """
    The old heuristic mistook the proposer for the critic, because the
    proposer's prompt ends with 'A Critic will attack your answer.' With a
    declared role the text becomes irrelevant.
    """
    llm = FakeLLM()
    out = llm.complete(
        [{"role": "system", "content": "YOU ARE THE CRITIC. Judge this."},
         {"role": "user", "content": "hi"}],
        model="anything", agent="proposer",
    )
    assert "VERDICT" not in out.text


def test_every_turn_captures_the_exact_prompt_it_was_given():
    state = _run(["draft", REVISE, "revised", APPROVE])
    for turn in state.transcript:
        assert turn.prompt_system, f"{turn.agent} turn lost its system prompt"
        assert turn.prompt_user, f"{turn.agent} turn lost its user prompt"
    # The revision must actually carry the critique forward - this is the
    # field that catches "the model was sent something other than what I
    # assumed" bugs.
    revision = state.transcript[2]
    # Assert the PROPERTY (the critique reached the reviser), not the exact
    # header, and additionally that the point ledger reached it - that is what
    # makes the second round a reply rather than a fresh attempt.
    assert "Unsupported claim" in revision.prompt_user
    assert "P1" in revision.prompt_user, "open points not carried into the revision"


def test_token_sink_receives_tokens_tagged_with_the_speaking_agent():
    """Live UI output depends on knowing WHO is speaking, not just what."""
    seen: list[tuple[str, str]] = []
    settings = load_settings(provider="fake")
    settings.max_rounds = 1
    graph = build_debate_graph(
        FakeLLM(script=["draft text", APPROVE]), settings,
        lambda agent, token: seen.append((agent, token)),
    )
    graph.invoke(DebateState(topic="t", max_rounds=1, max_cost_usd=0.0))
    agents = {a for a, _ in seen}
    assert agents == {"proposer", "critic"}
    assert "".join(t for a, t in seen if a == "proposer").strip() == "draft text"


# ---------------------------------------------------------------------------
# The offline path must stay reproducible from a bare clone
# ---------------------------------------------------------------------------


def test_fake_provider_ignores_model_env_overrides():
    """
    Regression: a .env naming real local models made OFFLINE runs record
    `gemma2:2b` in their transcripts - a model that generated none of that
    output. A model field that is a guess is worse than an absent one,
    because it will be believed.

    It also quietly disabled a FakeLLM role heuristic keyed on the model
    name containing "critic".
    """
    import os

    os.environ["AEGIS_PROPOSER_MODEL"] = "gemma2:2b"
    os.environ["AEGIS_CRITIC_MODEL"] = "gemma2:2b"
    try:
        fake = load_settings(provider="fake")
        assert fake.proposer_model == "fake-proposer"
        assert fake.critic_model == "fake-critic"
        # ...while a real provider must still honour the override.
        assert load_settings(provider="ollama").proposer_model == "gemma2:2b"
    finally:
        os.environ.pop("AEGIS_PROPOSER_MODEL", None)
        os.environ.pop("AEGIS_CRITIC_MODEL", None)


def test_fake_transcripts_name_the_fake_model():
    from aegis import run_debate

    state = run_debate("t", provider="fake", max_rounds=1)
    assert all("fake" in t.model for t in state.transcript), \
        [t.model for t in state.transcript]
