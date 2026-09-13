"""
aegis.state — the single source of truth for a debate run.

WHY THIS FILE EXISTS AT ALL
---------------------------
The #1 thing that separates a real multi-agent system from a pile of
chained prompt calls is that the system has ONE explicit, inspectable
state object. Every agent reads from it and writes to it. Nothing is
hidden in a closure, a global, or a Streamlit session variable.

Once state is explicit you get, almost for free:
  * replay      - re-run a debate from a saved state
  * checkpoint  - pause after round 2, resume tomorrow
  * evaluation  - diff two runs on the same topic
  * debugging   - print the state, see exactly why it routed the way it did

If you remember one idea from this whole project, make it this one.

CONTRACT WITH THE GRAPH
-----------------------
Nodes (agent functions) NEVER mutate the state in place. They return a
plain dict of the fields they want to change, e.g.

    return {"answer": "...", "round": state.round + 1}

The graph runner merges that dict into the state. This is the same
contract LangGraph uses, which is why our MiniGraph fallback and
LangGraph are drop-in swappable in graph.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal

# `tools` imports nothing from this package, so this direction is safe. The
# alternative - defining Source here and having tools import it - would put a
# transport-shaped type in the state module; retrieved evidence belongs to the
# run, but the SHAPE of a search result belongs to retrieval.
from .tools import Source

# A critic must answer with exactly one of these. Never free-form prose.
# Deterministic routing depends on this being a closed set.
Verdict = Literal["APPROVE", "REVISE"]

# Why the run stopped. Being able to distinguish these after the fact is
# the difference between "the system works" and "I think the system works".
StopReason = Literal[
    "",                  # still running
    "approved",          # critic was satisfied - the happy path
    "arbitrated",        # deadlocked, then resolved by the Arbiter
    "max_rounds",        # hit the cap with no arbiter - weakest outcome
    "error",             # something threw
    "cancelled",         # user stopped it from the UI
]


@dataclass
class Turn:
    """One agent speaking once. The atomic unit of the transcript."""

    round: int
    agent: str                      # "proposer" | "critic" | future: "researcher", ...
    content: str
    verdict: str | None = None      # only the critic sets this
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    model: str = ""

    # Generation telemetry. ttft and tokens_per_s are kept apart because
    # locally they diagnose different faults - see LLMResponse.
    ttft_s: float = 0.0
    tokens_per_s: float = 0.0

    # A reasoning model's internal monologue for this turn. Kept apart from
    # `content` on purpose: `content` is what the next agent sees and what
    # the verdict parser reads, and folding thought into it would put a
    # paragraph of musing ahead of the VERDICT line. Stored so the UI can
    # show the model's reasoning without any of it reaching the machinery.
    reasoning: str = ""

    # THE EXACT PROMPT THIS TURN WAS GIVEN.
    #
    # Stored on the turn, not reconstructed later, and this is the single
    # highest-value debugging field in the project. Nearly every "the model
    # is being stupid" bug turns out to be "the model was sent something
    # other than what I believed it was sent" - a stale critique, a missing
    # revision counter, the wrong system prompt for the role. You cannot
    # diagnose that from the output alone, and rebuilding the prompt after
    # the fact just reproduces your assumption about what it was.
    prompt_system: str = ""
    prompt_user: str = ""
    at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Point:
    """
    One contested claim, tracked across the whole debate.

    WHY A LEDGER TURNS REVIEW INTO A DEBATE
    ---------------------------------------
    Before this existed, the Critic saw the topic and the current answer and
    nothing else - not even its own previous objections. So every round it
    answered a fresh question ("what is wrong with this?") instead of the
    one a debate actually turns on ("were my objections answered?").

    Two consequences followed, and neither was a prompt problem:

      * Convergence was impossible by construction. There is always another
        fault to find, so a critic with no memory of what it already asked
        for can go on asking forever. The round cap was doing the work that
        agreement should have been doing.
      * The Proposer could dispute a point and simply never be answered. It
        was allowed to push back, and nothing in the system ever ruled on
        the pushback. That is not an argument; it is two monologues.

    A point has a life: raised, answered or disputed, then RULED ON. The
    Critic must close its own points before opening new ones, so "APPROVE"
    comes to mean "nothing I raised is still open" rather than "I failed to
    think of anything this time".

    status:
      open      - raised, not yet resolved
      resolved  - the Critic accepts it was addressed
      withdrawn - the Critic accepts the Proposer's rebuttal; it was wrong
      disputed  - the Proposer pushed back and the Critic has not yet ruled
    """

    id: str                          # "P1", stable for the whole debate
    text: str
    round_raised: int
    status: str = "open"
    # What the Proposer said about it: FIXED (changed the answer) or
    # DISPUTED (defended the original). Recorded separately from the
    # Critic's ruling, because "the Proposer thinks it is fixed" and "the
    # Critic agrees" are different facts and collapsing them would let the
    # Proposer close its own objections.
    proposer_stance: str = ""        # "FIXED" | "DISPUTED" | ""
    proposer_note: str = ""
    critic_note: str = ""
    round_closed: int = 0

    @property
    def is_open(self) -> bool:
        return self.status in ("open", "disputed")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    """
    One routing decision, recorded as it is made.

    WHY THE ENGINE RECORDS THIS INSTEAD OF THE UI RECONSTRUCTING IT
    --------------------------------------------------------------
    Invariant: frontends contain zero orchestration logic. But "show me
    why this debate stopped" is exactly the question a user of an agent
    system most wants answered, and a UI can only answer it two ways:
    re-implement the router's conditions (breaking the invariant, and
    guaranteeing the copy drifts from the original), or read a record the
    engine wrote. The second is strictly better, so the engine writes one.

    The wider point: an agent system's control flow is its least visible
    and most consequential part. Log the decision at the moment it is
    taken, with the values it was taken on. Reconstructing it later from
    outcomes is guesswork dressed as telemetry.

    rule   - machine-readable name of the condition that fired. Stable
             enough to count across runs in an eval.
    reason - the human sentence, for the UI.
    """

    at: str                     # node the decision was made after
    round: int
    rule: str                   # "approved" | "budget" | "timeout" | "deadlock" | "revise"
    next_node: str
    reason: str
    verdict: str = ""
    observed: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DebateState:
    """
    Everything a debate run knows about itself.

    Field-by-field reasoning:

    topic       - immutable input. Never changed by any node.
    max_rounds  - the hard iteration cap. The report lists runaway loops
                  as failure mode #1 (Proposer and Critic disagreeing
                  forever, burning tokens). A cap is not a nice-to-have;
                  it is the primary safety mechanism in this design.
    round       - which cycle we are on. Incremented by the critic node,
                  because a "round" is only complete once both agents
                  have spoken.
    answer      - the CURRENT best answer. Overwritten each revision.
                  We keep history in `transcript`, so overwriting here
                  is safe and keeps the prompt short.
    critique    - the latest critique text, fed into the next revision.
    verdict     - parsed APPROVE/REVISE. The graph routes on THIS FIELD
                  ONLY - never on the raw text of the critique.
    transcript  - append-only log of every Turn. This is what the UI
                  renders and what transcript.py persists.
    done        - set True by whichever node decides the run is over.
    stop_reason - WHY it is done. Critical for evaluation later.
    """

    topic: str
    max_rounds: int = 3

    round: int = 0
    answer: str = ""
    critique: str = ""
    verdict: str = ""

    # The Arbiter's full ruling, kept separate from `answer`. `answer`
    # holds only the extracted FINAL ANSWER section, because that is what
    # the user asked for; the reasoning behind the ruling is available
    # here for anyone who wants to audit the decision.
    ruling: str = ""

    transcript: list[Turn] = field(default_factory=list)

    # Append-only log of every routing decision. Written by the router,
    # read by the UI and by anyone asking "why did it stop?".
    decisions: list["Decision"] = field(default_factory=list)

    # Retrieved evidence, gathered once by the Researcher before the debate
    # starts. Shown to BOTH agents: the Proposer so it can support a claim,
    # the Critic so an objection can cite a source instead of merely noting
    # that the answer did not. Lives on the state like everything else, so a
    # saved transcript records what the argument was actually grounded in -
    # a debate you cannot re-check is an anecdote.
    sources: list["Source"] = field(default_factory=list)

    # The exact query the Researcher issued. Kept because "why did it find
    # nothing useful?" is almost always answered by the query, not the engine.
    research_query: str = ""

    # The disagreement ledger. REPLACE semantics, not append: a round both
    # adds new points and changes the status of old ones, and a node that
    # could only append would have no way to record that P2 was resolved.
    # Nodes build the new list from `state.points` and return it, so they
    # still never mutate state in place.
    points: list["Point"] = field(default_factory=list)

    done: bool = False
    stop_reason: str = ""
    error: str = ""

    # Cost accounting. You are on a $5 OpenRouter budget - if you cannot
    # see spend accumulating, you will blow through it without noticing.
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    # Hard spend ceiling for this run. 0.0 disables the guard.
    # Lives on the STATE, not just in Settings, because it must be
    # visible in a saved transcript: when you look at a run six weeks
    # later you need to know what limits it was operating under.
    max_cost_usd: float = 0.50

    # Seconds of inference spent so far, accumulated from turn latencies.
    elapsed_s: float = 0.0

    # Wall-clock ceiling. The guard that actually binds when inference is
    # local and therefore free. Same reasoning as max_cost_usd living
    # here: a limit invisible in the transcript cannot explain the run it
    # terminated. 0.0 disables.
    max_seconds: float = 0.0

    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    run_id: str = ""

    # ---- helpers -------------------------------------------------------

    def apply(self, updates: dict[str, Any]) -> "DebateState":
        """
        Merge a node's returned dict into this state.

        `transcript` is special-cased: it APPENDS rather than replaces.
        In LangGraph this behaviour is called a "reducer" and is declared
        with Annotated[list, operator.add]. We do it explicitly here so
        you can see that there is no magic involved.
        """
        for key, value in updates.items():
            if not hasattr(self, key):
                raise AttributeError(
                    f"Node tried to write unknown state field {key!r}. "
                    f"Add it to DebateState or fix the typo."
                )
            if key in ("transcript", "decisions"):
                getattr(self, key).extend(value)
            elif key in ("tokens_in", "tokens_out", "cost_usd", "elapsed_s"):
                setattr(self, key, getattr(self, key) + value)  # accumulate
            else:
                setattr(self, key, value)
        return self

    @property
    def rounds_remaining(self) -> int:
        return max(0, self.max_rounds - self.round)

    def max_cost_usd_exceeded(self) -> bool:
        """
        Budget guard, checked by the router each cycle.

        Separate from the round cap on purpose: round count and spend are
        not proportional. One round with a very long context can cost
        more than five short ones, so bounding rounds alone does not
        bound money. Two independent limits, two independent risks.
        """
        if self.max_cost_usd <= 0:
            return False
        return self.cost_usd >= self.max_cost_usd

    def max_seconds_exceeded(self) -> bool:
        """
        Wall-clock guard - the local-inference twin of the budget guard.

        Deliberately a separate method with the same shape, rather than one
        generalised "resource exceeded" check. The router has to be able to
        say WHICH limit stopped a run: "ran out of money" and "ran out of
        time" call for entirely different responses from whoever reads the
        transcript, and a merged predicate would erase that distinction at
        exactly the moment it matters.
        """
        if self.max_seconds <= 0:
            return False
        return self.elapsed_s >= self.max_seconds

    @property
    def open_points(self) -> list["Point"]:
        return [p for p in self.points if p.is_open]

    def points_summary(self) -> str:
        """One line, for a CLI footer or a decision reason."""
        if not self.points:
            return "no points raised"
        closed = len(self.points) - len(self.open_points)
        withdrawn = sum(1 for p in self.points if p.status == "withdrawn")
        return (f"{len(self.points)} raised · {closed} closed"
                f"{f' ({withdrawn} withdrawn)' if withdrawn else ''} · "
                f"{len(self.open_points)} open")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["transcript"] = [t.to_dict() for t in self.transcript]
        data["decisions"] = [d.to_dict() for d in self.decisions]
        data["points"] = [p.to_dict() for p in self.points]
        data["sources"] = [s.to_dict() for s in self.sources]
        return data
