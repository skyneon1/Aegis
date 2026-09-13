"""
Offline tests for the Aegis orchestration engine.

Every test here runs with a FakeLLM: no network, no API key, no cost,
milliseconds instead of minutes.

WHAT THESE TESTS ARE ACTUALLY FOR
---------------------------------
They do not test whether the answers are GOOD. That is a model-quality
question and needs a different tool (an eval harness over real runs).

They test whether the MACHINE is correct: does the loop terminate, does
the cap hold, does routing follow the verdict, does the transcript
accumulate, does a malformed critic response fail in the safe direction.

Separating "is the orchestration correct" from "is the output good" is
the single most useful discipline in agent engineering. Almost every
stalled agent project is one where those two questions got tangled.

Run:  pytest tests/ -v        (if pytest is installed)
      python tests/test_graph.py   (works with no dependencies at all)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aegis import DebateState, FakeLLM, build_debate_graph, load_settings, run_debate
from aegis.agents import extract_reasons, parse_verdict
from aegis.graph import END, MiniGraph, route_after_critic

APPROVE = "VERDICT: APPROVE\nREASONS:\n- Accurate and complete.\n"
REVISE = "VERDICT: REVISE\nREASONS:\n- Unsupported claim in paragraph two.\n"


def _settings(**kw):
    s = load_settings(provider="fake")
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _run(script, *, max_rounds=3, use_arbiter=False,
         topic="Is REST better than GraphQL?"):
    """
    Run a debate against a scripted model.

    `use_arbiter` defaults to False here so that these tests exercise the
    two-agent core in isolation. Production defaults to True; the Arbiter
    path has its own dedicated section further down.

    Testing the core without the escalation path is deliberate: when a
    test fails you want it to implicate one subsystem, not two.
    """
    settings = _settings(max_rounds=max_rounds, use_arbiter=use_arbiter)
    llm = FakeLLM(script=script)
    graph = build_debate_graph(llm, settings)
    state = DebateState(topic=topic, max_rounds=max_rounds,
                        max_cost_usd=settings.max_cost_usd)
    return graph.invoke(state), llm


# ---------------------------------------------------------------------------
# Verdict parsing — the contract the whole control flow rests on
# ---------------------------------------------------------------------------


def test_parse_verdict_approve():
    assert parse_verdict(APPROVE) == "APPROVE"
    assert parse_verdict("verdict: approve\nreasons: fine") == "APPROVE"


def test_parse_verdict_survives_markdown_decoration():
    """Models bold their structured output whether you ask them to or not.

    This exact case was a real bug: '**VERDICT:** APPROVE' failed to parse,
    fell through to the safe default REVISE, and would have presented as
    'the critic never approves anything' - a reasoning bug in appearance,
    a regex bug in reality.
    """
    for text in [
        "**VERDICT:** APPROVE",
        "## VERDICT - APPROVE",
        "`VERDICT:` APPROVE",
        "**VERDICT: APPROVE**",
        "_VERDICT_: REVISE",
    ]:
        expected = "APPROVE" if "APPROVE" in text else "REVISE"
        assert parse_verdict(text) == expected, text


def test_parse_verdict_revise():
    assert parse_verdict(REVISE) == "REVISE"
    assert parse_verdict("VERDICT - REVISE") == "REVISE"


def test_parse_verdict_fails_safe():
    """Unparseable output must become REVISE, never APPROVE.

    Asymmetric failure: a wrong REVISE costs one round; a wrong APPROVE
    ships unreviewed work while pretending it passed review.
    """
    for junk in ["", "I think it's pretty good honestly", "LGTM!", None or ""]:
        assert parse_verdict(junk) == "REVISE", junk


def test_extract_reasons():
    reasons = extract_reasons(REVISE)
    assert reasons == ["Unsupported claim in paragraph two."]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_approve_on_first_round_stops_immediately():
    state, llm = _run(["First answer.", APPROVE])

    assert state.done is True
    assert state.stop_reason == "approved"
    assert state.verdict == "APPROVE"
    assert state.round == 1
    assert state.answer == "First answer."
    assert len(state.transcript) == 2          # proposer + critic
    assert len(llm.calls) == 2                 # exactly two model calls, no waste


def test_revise_then_approve_runs_two_rounds():
    state, llm = _run(["Draft one.", REVISE, "Draft two.", APPROVE])

    assert state.stop_reason == "approved"
    assert state.round == 2
    assert state.answer == "Draft two."        # final answer is the revision
    assert len(state.transcript) == 4
    assert [t.agent for t in state.transcript] == [
        "proposer", "critic", "proposer", "critic"
    ]


# ---------------------------------------------------------------------------
# The iteration cap — failure mode #1 from the strategy report
# ---------------------------------------------------------------------------


def test_endless_disagreement_is_stopped_by_the_cap():
    """A critic that NEVER approves must not loop forever."""
    script = ["a", REVISE] * 10                # 10 rounds available in the script
    state, llm = _run(script, max_rounds=3)    # but the cap is 3

    assert state.done is True
    assert state.stop_reason == "max_rounds"
    assert state.round == 3
    assert len(state.transcript) == 6          # 3 rounds x 2 agents
    assert len(llm.calls) == 6                 # cap actually saved us 14 calls


def test_cap_still_bounds_the_loop_when_the_arbiter_is_enabled():
    """
    The Arbiter changes the ENDING, never the bound.

    Worth asserting explicitly: adding an escalation path is exactly the
    kind of change that quietly relaxes a safety limit. It must cost one
    extra call, and only one.
    """
    script = ["a", REVISE] * 10
    state, llm = _run(script, max_rounds=3, use_arbiter=True)

    assert state.done is True
    assert state.stop_reason == "arbitrated"
    assert state.round == 3                    # bound unchanged
    assert len(llm.calls) == 7                 # 6 debate + exactly 1 ruling


def test_max_rounds_one_still_produces_an_answer():
    state, _ = _run(["only answer", REVISE], max_rounds=1)

    assert state.round == 1
    assert state.stop_reason == "max_rounds"
    assert state.answer == "only answer"       # unapproved, but present


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------


def test_budget_guard_trips_independently_of_round_count():
    state = DebateState(topic="t", max_rounds=99, max_cost_usd=0.01)
    state.apply({"verdict": "REVISE", "round": 1, "cost_usd": 0.05})

    assert state.max_cost_usd_exceeded() is True
    assert route_after_critic(state) == END
    assert state.stop_reason == "max_rounds"


def test_budget_guard_disabled_when_zero():
    state = DebateState(topic="t", max_cost_usd=0.0)
    state.apply({"cost_usd": 999.0})
    assert state.max_cost_usd_exceeded() is False


# ---------------------------------------------------------------------------
# State mechanics
# ---------------------------------------------------------------------------


def test_transcript_appends_while_scalars_replace():
    state = DebateState(topic="t")
    state.apply({"answer": "one", "transcript": []})
    state.apply({"answer": "two"})
    assert state.answer == "two"               # scalar replaced


def test_cost_fields_accumulate_rather_than_replace():
    state = DebateState(topic="t")
    state.apply({"cost_usd": 0.01, "tokens_in": 100})
    state.apply({"cost_usd": 0.02, "tokens_in": 50})
    assert abs(state.cost_usd - 0.03) < 1e-9
    assert state.tokens_in == 150


def test_writing_an_unknown_field_raises_loudly():
    """Typos in a node's return dict must fail fast, not vanish."""
    state = DebateState(topic="t")
    try:
        state.apply({"anwser": "typo"})
    except AttributeError as exc:
        assert "anwser" in str(exc)
    else:
        raise AssertionError("Expected AttributeError for unknown state field")


# ---------------------------------------------------------------------------
# Graph wiring safety
# ---------------------------------------------------------------------------


def test_graph_rejects_a_node_with_no_exit():
    g = MiniGraph()
    g.add_node("orphan", lambda s: {})
    g.set_entry("orphan")
    try:
        g.validate()
    except ValueError as exc:
        assert "no outgoing edge" in str(exc)
    else:
        raise AssertionError("Expected validate() to catch the dangling node")


def test_graph_rejects_edge_to_unknown_node():
    g = MiniGraph()
    g.add_node("a", lambda s: {})
    g.add_edge("a", "nowhere")
    g.set_entry("a")
    try:
        g.validate()
    except ValueError as exc:
        assert "nowhere" in str(exc)
    else:
        raise AssertionError("Expected validate() to catch the bad edge")


def test_node_exception_is_captured_not_raised():
    """A crashing agent must preserve the partial transcript."""
    def boom(state):
        raise RuntimeError("model exploded")

    g = MiniGraph()
    g.add_node("boom", boom)
    g.add_edge("boom", END)
    g.set_entry("boom")

    state = DebateState(topic="t")
    final = g.invoke(state)
    assert final.done is True
    assert final.stop_reason == "error"
    assert "model exploded" in final.error


def test_structural_step_limit_backstops_a_buggy_router():
    """Second safety net: even a router that never returns END must halt."""
    g = MiniGraph()
    g.add_node("spin", lambda s: {})
    g.add_conditional_edges("spin", lambda s: "spin")   # deliberate infinite loop
    g.set_entry("spin")

    final = g.invoke(DebateState(topic="t"), max_steps=8)
    assert final.done is True
    assert "structural step limit" in final.error


# ---------------------------------------------------------------------------
# FakeLLM role detection — regression tests for two real bugs
# ---------------------------------------------------------------------------


def test_fake_proposer_never_emits_a_verdict():
    """Regression: the proposer's system prompt says 'A Critic will attack
    your answer.' An early FakeLLM sniffed for the substring 'critic' and
    so made the PROPOSER emit critic verdicts. Identity must never be
    inferred from incidental content.
    """
    from aegis.agents import PROPOSER_SYSTEM, REVISER_SYSTEM

    llm = FakeLLM()
    for system in (PROPOSER_SYSTEM, REVISER_SYSTEM):
        out = llm.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": "TOPIC: anything"}],
            model="fake-proposer",
        )
        assert "VERDICT" not in out.text, system[:40]


def test_fake_critic_is_recognised_and_eventually_approves():
    """Regression: the fake critic used to look for 'REVISION' in its own
    prompt - text only the proposer ever receives - so it revised forever.
    """
    from aegis.agents import CRITIC_SYSTEM

    llm = FakeLLM()
    msgs = [{"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": "PROPOSER'S ANSWER: something"}]

    first = llm.complete(msgs, model="fake-critic")
    second = llm.complete(msgs, model="fake-critic")

    assert parse_verdict(first.text) == "REVISE"
    assert parse_verdict(second.text) == "APPROVE"


def test_default_fake_run_ends_approved_not_capped():
    """The out-of-the-box experience should demo the happy path."""
    state = run_debate("What makes code readable?", provider="fake", max_rounds=3)
    assert state.stop_reason == "approved"
    assert state.round == 2
    assert [t.agent for t in state.transcript] == [
        "proposer", "critic", "proposer", "critic"
    ]


# ---------------------------------------------------------------------------
# The Arbiter — third agent, deadlock resolution
# ---------------------------------------------------------------------------

RULING = (
    "RULING:\n- Scope: Critic right.\n\n"
    "FINAL ANSWER:\nThe adjudicated answer.\n\n"
    "REMAINING UNCERTAINTY:\nNone material."
)


def test_deadlock_escalates_to_the_arbiter_instead_of_giving_up():
    """v0.1 handed back an unreviewed draft on deadlock. Now it rules."""
    settings = _settings(max_rounds=2, use_arbiter=True)
    llm = FakeLLM(script=["a", REVISE, "b", REVISE, RULING])
    graph = build_debate_graph(llm, settings)
    state = graph.invoke(DebateState(topic="t", max_rounds=2))

    assert state.stop_reason == "arbitrated"
    assert state.answer == "The adjudicated answer."   # extracted, not raw
    assert state.ruling.startswith("RULING:")          # full ruling retained
    assert [t.agent for t in state.transcript][-1] == "arbiter"


def test_arbiter_can_be_disabled_and_old_behaviour_returns():
    settings = _settings(max_rounds=1, use_arbiter=False)
    llm = FakeLLM(script=["a", REVISE])
    graph = build_debate_graph(llm, settings)
    state = graph.invoke(DebateState(topic="t", max_rounds=1))

    assert state.stop_reason == "max_rounds"
    assert "arbiter" not in [t.agent for t in state.transcript]


def test_arbiter_never_runs_on_the_happy_path():
    """It must cost nothing when the critic approves."""
    settings = _settings(max_rounds=3, use_arbiter=True)
    llm = FakeLLM(script=["a", APPROVE])
    graph = build_debate_graph(llm, settings)
    state = graph.invoke(DebateState(topic="t", max_rounds=3))

    assert state.stop_reason == "approved"
    assert len(llm.calls) == 2          # proposer + critic only
    assert "arbiter" not in [t.agent for t in state.transcript]


def test_arbiter_is_terminal_and_cannot_reenter_the_loop():
    """A judge you can appeal to repeatedly is just another debater."""
    settings = _settings(max_rounds=2, use_arbiter=True)
    llm = FakeLLM(script=["a", REVISE, "b", REVISE, RULING, "SHOULD NEVER RUN"])
    graph = build_debate_graph(llm, settings)
    state = graph.invoke(DebateState(topic="t", max_rounds=2))

    assert [t.agent for t in state.transcript].count("arbiter") == 1
    assert len(llm.calls) == 5          # the 6th script entry was never used


def test_budget_guard_beats_the_arbiter_branch():
    """
    Ordering test. The budget guard is checked BEFORE the deadlock branch,
    so an out-of-money run must NOT spend another call on arbitration.
    A guard the escalation path can bypass is not a guard.
    """
    state = DebateState(topic="t", round=3, max_rounds=3,
                        verdict="REVISE", max_cost_usd=0.01)
    state.apply({"cost_usd": 0.99})

    assert route_after_critic(state, use_arbiter=True) == END
    assert "Budget guard" in state.error


def test_extract_final_answer_falls_back_to_full_text():
    """Presentation-layer parsing fails OPEN, unlike verdict parsing."""
    from aegis.agents import extract_final_answer

    assert extract_final_answer(RULING) == "The adjudicated answer."
    assert extract_final_answer("no markers here") == "no markers here"
    assert extract_final_answer("") == ""


def test_fake_arbiter_is_not_mistaken_for_the_critic():
    """
    Regression for a bug the Arbiter *would* have inherited.

    The Arbiter reuses the critic's MODEL NAME by default. FakeLLM's
    critic check matches on 'critic' in the model name - so without an
    arbiter check running FIRST, the fake Arbiter would emit a verdict
    instead of a ruling. Overlapping role signals need explicit priority.
    """
    from aegis.agents import ARBITER_SYSTEM

    llm = FakeLLM()
    out = llm.complete(
        [{"role": "system", "content": ARBITER_SYSTEM},
         {"role": "user", "content": "the exchange"}],
        model="fake-critic",              # deliberately the critic's model
    )
    assert "VERDICT" not in out.text
    assert "FINAL ANSWER:" in out.text


# ---------------------------------------------------------------------------
# Streaming + top-level API
# ---------------------------------------------------------------------------


def test_stream_yields_every_turn_in_order():
    settings = _settings(max_rounds=2)
    llm = FakeLLM(script=["one", REVISE, "two", APPROVE])
    graph = build_debate_graph(llm, settings)
    state = DebateState(topic="t", max_rounds=2)

    seen = [name for name, _ in graph.stream(state)]
    assert seen == ["proposer", "critic", "proposer", "critic"]


def test_run_debate_works_with_no_api_key_at_all():
    """The fresh-clone experience: it must just run."""
    state = run_debate("What makes code readable?", provider="fake", max_rounds=2)
    assert state.answer
    assert state.done
    assert state.stop_reason in ("approved", "max_rounds")


def test_empty_topic_is_rejected():
    for bad in ["", "   "]:
        try:
            run_debate(bad, provider="fake")
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for empty topic")


# ---------------------------------------------------------------------------
# Dependency-free runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
