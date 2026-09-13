"""
The disagreement ledger: what turns review into a debate.

WHAT IS BEING PINNED
--------------------
Before the ledger, the Critic saw the topic and the current answer and
nothing else - not even its own previous objections. So each round it
answered "what is wrong with this?" (unbounded: there is always something)
rather than "were my objections answered?" (bounded, and able to converge).
The Proposer could also dispute a point and never be answered, which is two
monologues rather than an argument.

These tests pin the properties that fix that, and the fail-safe directions
around them - because every parser here is guessing at a small model's
formatting, and each guess has a cheap direction and an expensive one.

Hermetic: the parsers and ledger logic are pure functions over text.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aegis import DebateState, FakeLLM, build_debate_graph, load_settings
from aegis.agents import (
    apply_responses,
    apply_rulings,
    extract_answer,
    format_points_for_prompt,
    parse_new_points,
    parse_responses,
    parse_rulings,
)
from aegis.state import Point


# ---------------------------------------------------------------------------
# Parsing what a model actually emits
# ---------------------------------------------------------------------------

CRITIC_REPLY = """VERDICT: REVISE
RULINGS:
- P1: RESOLVED - the mechanism is now given.
- **P2:** OPEN — still no support for the 40% figure.
- P3: WITHDRAWN - the Proposer is right.
NEW:
- Invents a deployment-frequency statistic.
"""


def test_rulings_survive_markdown_and_dashes():
    """
    Models decorate structured output no matter how strictly you ask. The
    parser is written defensively even though the prompt is strict, because
    prompts are requests, not guarantees - and a brittle parser produces bugs
    that look like reasoning failures, which are much harder to diagnose.
    """
    rulings = parse_rulings(CRITIC_REPLY)
    assert rulings["P1"][0] == "RESOLVED"
    assert rulings["P2"][0] == "OPEN"          # bold label, em-dash separator
    assert rulings["P3"][0] == "WITHDRAWN"
    assert "40% figure" in rulings["P2"][1]


def test_ruling_synonyms_are_normalised():
    parsed = parse_rulings("- P1: ADDRESSED - fine\n- P2: STILL OPEN - nope")
    assert parsed["P1"][0] == "RESOLVED"
    assert parsed["P2"][0] == "OPEN"


def test_new_points_exclude_rulings_on_existing_ones():
    """A ruling line is not a new point, even though both are bullets."""
    new = parse_new_points(CRITIC_REPLY)
    assert new == ["Invents a deployment-frequency statistic."]


def test_first_round_reasons_are_read_as_points():
    """Round one has no ledger, so a plain REASONS list must still work."""
    reply = ("VERDICT: REVISE\nREASONS:\n"
             "- Unsupported claim about scaling.\n- Wrong about Git internals.\n")
    assert len(parse_new_points(reply)) == 2


def test_proposer_stances_and_answer_are_separated():
    reply = ("RESPONSES:\n- P1: FIXED - added the mechanism.\n"
             "- P2: DISPUTED - the figure is standard.\n\n"
             "ANSWER:\nThe standalone answer.")
    stances = parse_responses(reply)
    assert stances["P1"] == ("FIXED", "added the mechanism.")
    assert stances["P2"][0] == "DISPUTED"
    # The next Critic must judge the ANSWER, not the commentary about the
    # previous round.
    assert extract_answer(reply) == "The standalone answer."


def test_answer_extraction_fails_open():
    """
    Presentation, so it fails OPEN - the opposite direction to parse_verdict,
    which drives control flow and fails SAFE. Showing slightly too much is
    cosmetic; showing nothing because a header was missing is not.
    """
    assert extract_answer("no headers at all, just prose") == \
        "no headers at all, just prose"


# ---------------------------------------------------------------------------
# Fail-safe: nobody may close a point by accident
# ---------------------------------------------------------------------------


def test_a_point_the_critic_forgot_to_rule_on_stays_open():
    """
    The load-bearing fail-safe. A formatting slip must never resolve an
    objection: the cost of leaving it open is one more round, the cost of
    closing it wrongly is shipping unreviewed work as reviewed.
    """
    points = [Point("P1", "a", 1), Point("P2", "b", 1)]
    out = apply_rulings(points, {"P1": ("RESOLVED", "")}, [], 2)
    assert out[0].status == "resolved"
    assert out[1].status == "open", "an unmentioned point was silently closed"


def test_the_proposer_cannot_close_its_own_objection():
    """
    DISPUTED is not RESOLVED. Only the Critic's ruling can close a point -
    otherwise the party being reviewed decides when review is finished.
    """
    points = [Point("P1", "a", 1)]
    out = apply_responses(points, {"P1": ("DISPUTED", "the claim is standard")})
    assert out[0].status == "disputed"
    assert out[0].is_open is True
    assert out[0].proposer_stance == "DISPUTED"


def test_a_disputed_point_can_be_withdrawn_by_the_critic():
    """The rebuttal has to be able to WIN, or disputing is theatre."""
    points = [Point("P1", "a", 1, status="disputed", proposer_stance="DISPUTED")]
    out = apply_rulings(points, {"P1": ("WITHDRAWN", "fair point")}, [], 2)
    assert out[0].status == "withdrawn"
    assert out[0].is_open is False
    assert out[0].round_closed == 2


def test_ledger_functions_never_mutate_their_input():
    """Nodes return updates; they do not modify state. Same rule here."""
    points = [Point("P1", "a", 1)]
    apply_rulings(points, {"P1": ("RESOLVED", "")}, ["new"], 2)
    apply_responses(points, {"P1": ("FIXED", "")})
    assert points[0].status == "open" and len(points) == 1


# ---------------------------------------------------------------------------
# Deduplication: the debate has to be able to end
# ---------------------------------------------------------------------------


def test_a_resolved_point_cannot_come_back_as_new():
    """
    Observed on a real run: the Critic closed P2 as RESOLVED and in the same
    reply raised P3 with almost identical wording, so the debate deadlocked on
    something it had just agreed was fixed. The prompt asks it not to; asking
    is not enough, because a stateless critic re-derives the same objection
    from the same answer. The ledger enforces it.
    """
    resolved = [Point("P1", "The answer needs to address the specific mechanisms "
                            "supporting claims about Kubernetes advantages", 1,
                      status="resolved")]
    out = apply_rulings(resolved, {}, [
        "The answer must address the specific mechanisms that support its "
        "claims about Kubernetes advantages",
    ], 2)
    assert len(out) == 1, "a re-raised point was added as new"


def test_a_genuinely_different_point_is_still_added():
    """Dedup must not become a gag. Different objection, different point."""
    existing = [Point("P1", "no mechanism for the scaling claim", 1,
                      status="resolved")]
    out = apply_rulings(existing, {}, ["invents a statistic about deploys"], 2)
    assert [p.id for p in out] == ["P1", "P2"]


def test_an_empty_new_point_is_dropped():
    out = apply_rulings([], {}, ["", "   "], 1)
    assert out == []


# ---------------------------------------------------------------------------
# What the agents are shown
# ---------------------------------------------------------------------------


def test_only_open_points_are_put_in_front_of_an_agent():
    points = [Point("P1", "closed one", 1, status="resolved"),
              Point("P2", "live one", 1)]
    rendered = format_points_for_prompt(points)
    assert "live one" in rendered
    assert "closed one" not in rendered


def test_a_disputed_point_shows_the_rebuttal_to_the_critic():
    """The Critic has to see WHAT it is being asked to concede."""
    points = [Point("P2", "the 40% claim", 1, status="disputed",
                    proposer_stance="DISPUTED",
                    proposer_note="the figure is standard")]
    rendered = format_points_for_prompt(points)
    assert "DISPUTED" in rendered and "figure is standard" in rendered


# ---------------------------------------------------------------------------
# End to end through the graph
# ---------------------------------------------------------------------------


def test_a_full_debate_builds_and_closes_the_ledger():
    settings = load_settings(provider="fake")
    settings.max_rounds = 3
    settings.use_arbiter = False
    llm = FakeLLM(script=[
        "First answer.",
        "VERDICT: REVISE\nREASONS:\n- Unsupported claim about scaling.\n",
        "RESPONSES:\n- P1: FIXED - added support.\n\nANSWER:\nBetter answer.",
        "VERDICT: APPROVE\nRULINGS:\n- P1: RESOLVED - support is now given.\n",
    ])
    state = build_debate_graph(llm, settings).invoke(
        DebateState(topic="t", max_rounds=3, max_cost_usd=0.0))

    assert state.stop_reason == "approved"
    assert len(state.points) == 1
    point = state.points[0]
    assert point.status == "resolved"
    assert point.proposer_stance == "FIXED"
    assert state.open_points == []
    # The answer handed on is the ANSWER section, not the rebuttal preamble.
    assert state.answer == "Better answer."


def test_an_approval_that_contradicts_the_ledger_is_recorded():
    """
    Routing still branches on the VERDICT alone - the ledger informs, it does
    not decide. But approving while your own objections stand is incoherent,
    and recording it makes a badly-behaved critic countable instead of
    invisible.
    """
    settings = load_settings(provider="fake")
    settings.max_rounds = 3
    settings.use_arbiter = False
    llm = FakeLLM(script=[
        "First answer.",
        "VERDICT: REVISE\nREASONS:\n- Unsupported claim about scaling.\n",
        "ANSWER:\nSecond answer.",
        "VERDICT: APPROVE\nREASONS:\n- Looks fine now.\n",   # P1 never ruled on
    ])
    state = build_debate_graph(llm, settings).invoke(
        DebateState(topic="t", max_rounds=3, max_cost_usd=0.0))

    assert state.stop_reason == "approved"
    approved = [d for d in state.decisions if d.rule == "approved"][0]
    assert approved.observed["open_points"] == ["P1"]
    assert "still open" in approved.reason


def test_approval_reasons_are_not_read_as_new_objections():
    """
    REAL BUG. `VERDICT: APPROVE / REASONS: - Looks fine now.` had its
    justification parsed as a fresh objection, so approving a debate INVENTED
    a point and then reported the approval as contradicting the ledger.

    The bullets under REASONS mean opposite things depending on the verdict -
    grounds for rejecting, or grounds for accepting - so the same text cannot
    be parsed the same way in both cases.
    """
    reply = "VERDICT: APPROVE\nREASONS:\n- Looks fine now.\n- Claims are hedged.\n"
    assert parse_new_points(reply, strict=True) == []
    # ...but a real NEW section is still honoured, so a genuinely
    # self-contradicting critic stays visible rather than being smoothed over.
    both = ("VERDICT: APPROVE\nREASONS:\n- Fine.\nNEW:\n"
            "- Still invents a statistic.\n")
    assert parse_new_points(both, strict=True) == ["Still invents a statistic."]


def test_rejection_reasons_are_still_read_as_objections():
    """The non-strict path must keep working, or round one has no points."""
    reply = "VERDICT: REVISE\nREASONS:\n- Unsupported claim.\n"
    assert parse_new_points(reply) == ["Unsupported claim."]


# The exact strings from a real run, kept verbatim. A regression test built
# from observed output is worth more than one built from an invented example:
# these two scored 0.68 on the original intersection/min measure and slipped
# under its 0.7 bar, which is how the flaw was found.
_REAL_P2 = ("The answer provides some good advice on how to learn Kubernetes, "
            "but it could be more specific about which steps are most "
            "beneficial for a two-person startup. For example, the answer could "
            "mention that starting with a hands-on project using Minikube or "
            "Kind is particularly helpful for beginners.")
_REAL_P3 = ("The answer could benefit from more specific advice on which steps "
            "are most beneficial for a two-person startup. For example, it "
            "mentions hands-on projects using Minikube or Kind, but doesn't "
            "elaborate on why these are particularly helpful for beginners in "
            "this context.")


def test_real_observed_paraphrase_is_caught():
    out = apply_rulings([Point("P1", _REAL_P2, 2)], {}, [_REAL_P3], 3)
    assert len(out) == 1, "the observed paraphrase was added as a new point"


def test_distinct_objections_about_the_same_subject_stay_distinct():
    """
    The counterweight to the test above. Dedup must not merge two different
    complaints that happen to share a subject - that would silently suppress
    real findings, which is worse than a duplicate.
    """
    existing = [Point("P1", "the 40% figure is given with no source at all", 1)]
    out = apply_rulings(
        existing, {},
        ["the 40% figure directly contradicts the study the answer cites"], 2)
    assert len(out) == 2, "two distinct objections were merged"


def test_no_rulings_format_is_shown_when_there_are_no_prior_points():
    """
    REAL BUG, found in calibration output rather than by design.

    With the RULINGS format shown unconditionally, both gemma2:2b and
    llama3.2:3b invented rulings on a FIRST-round answer - "P1: RESOLVED ...
    P2: OPEN ..." citing objections nobody had made - and the fabricated
    "P2: OPEN" drove the verdict to REVISE. A good answer was rejected on the
    strength of an imaginary objection.

    The engine was already robust (a ruling for an unknown id is ignored, and
    P-prefixed lines never become new points) so the ledger stayed clean; the
    VERDICT did not. A format slot is an instruction: show a model a section
    and it will fill it in whether or not it has anything to put there.
    """
    settings = load_settings(provider="fake")
    settings.max_rounds = 3
    settings.use_arbiter = False
    llm = FakeLLM(script=[
        "First answer.",
        "VERDICT: REVISE\nREASONS:\n- Unsupported claim about scaling.\n",
        "RESPONSES:\n- P1: FIXED - added support.\n\nANSWER:\nBetter answer.",
        "VERDICT: APPROVE\nRULINGS:\n- P1: RESOLVED - support is now given.\n",
    ])
    build_debate_graph(llm, settings).invoke(
        DebateState(topic="t", max_rounds=3, max_cost_usd=0.0))

    critic_calls = [c for c in llm.calls if c["agent"] == "critic"]
    first, second = critic_calls[0], critic_calls[1]

    def system_of(call):
        return next(m["content"] for m in call["messages"] if m["role"] == "system")

    # Round 1: nothing to rule on, so no RULINGS machinery at all.
    assert "RULINGS:" not in system_of(first)
    assert "CLOSE YOUR OWN POINTS" not in system_of(first)
    # Round 2: P1 is open, so the clause appears.
    assert "RULINGS:" in system_of(second)
    # Substring chosen to sit inside one line of the prompt: the source wraps
    # "Do not invent / rulings on points", so the obvious phrase does not
    # exist as a contiguous string.
    assert "rulings on points that were never raised" in system_of(second)


def test_a_ruling_on_a_point_that_was_never_raised_is_ignored():
    """Defence in depth: even if a model invents P9, the ledger ignores it."""
    out = apply_rulings([Point("P1", "real point", 1)],
                        {"P1": ("OPEN", ""), "P9": ("RESOLVED", "imaginary")},
                        [], 2)
    assert [p.id for p in out] == ["P1"]
    assert out[0].status == "open"


# ---------------------------------------------------------------------------
# The instructions are not objections
# ---------------------------------------------------------------------------


def test_a_verbatim_quote_of_the_prompt_is_not_a_point():
    """
    OBSERVED IN THE UI, not caught by any earlier test: llama3.2:3b turned the
    Critic's own pre-approval checklist into ledger entries. Three of four
    points in that run were quotes of the instructions, which inflates the
    ledger, leaves fake points OPEN, and makes an approval look as though it
    contradicted itself.
    """
    from aegis.agents import CRITIC_SYSTEM, _is_prompt_echo

    quotes = [line.strip(" -*•")
              for line in CRITIC_SYSTEM.splitlines()
              if len(line.split()) >= 8][:5]
    assert quotes, "no long prompt lines to test against"
    for quote in quotes:
        assert _is_prompt_echo(quote), f"prompt line not detected: {quote[:60]}"


def test_real_objections_are_never_filtered_as_prompt_echoes():
    """
    THE COUNTERWEIGHT, and the more important half.

    The first version of this filter used intersection/min at 0.6 and dropped
    a genuine objection - "The answer fails to address the question that was
    asked..." overlapped heavily with the prompt line describing that very
    defect category. Of course it did: the prompt names the category and a
    real point instantiates it, so they share vocabulary by design.

    Silently dropping a real finding is far worse than keeping a junk one, so
    the filter is near-verbatim only. If this test ever fails, loosen the
    filter rather than the test.
    """
    from aegis.agents import _is_prompt_echo

    real = [
        "The answer fails to address the question that was asked. While the "
        "proposer discusses benefits and drawbacks, they never directly answer "
        "whether Kubernetes is necessary for such a setup.",
        "Claims a Gartner study estimates 100000 per year but gives no way to "
        "check that figure.",
        "States that Git stores full snapshots so repository size is free, "
        "which is false.",
        "The 47% time-to-merge figure has no source a reader could check.",
    ]
    for point in real:
        assert not _is_prompt_echo(point), f"real objection filtered: {point[:60]}"


def test_echoed_checklist_items_never_reach_the_ledger():
    """End to end through apply_rulings, which is where the filter is applied."""
    from aegis.agents import CRITIC_SYSTEM

    quote = next(line.strip(" -*•") for line in CRITIC_SYSTEM.splitlines()
                 if len(line.split()) >= 8)
    out = apply_rulings([], {}, [quote, "Invents a statistic about deploys."], 1)
    assert len(out) == 1
    assert out[0].text == "Invents a statistic about deploys."
