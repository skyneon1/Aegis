"""
aegis.graph — the orchestration engine and the debate wiring.

This file contains two things:

  1. MiniGraph  - a ~70-line state-machine runner I wrote so you can see
                  that a "graph framework" is not magic.
  2. build_debate_graph() - the actual Proposer/Critic wiring.

WHY WRITE MY OWN RUNNER INSTEAD OF USING LANGGRAPH IMMEDIATELY
--------------------------------------------------------------
Two honest reasons:

  * It runs today with zero installs. You can see the system work before
    committing to a dependency tree.
  * Frameworks stop being intimidating once you have read the small
    version. LangGraph does much more than this (durable checkpoints,
    parallel branches, interrupts, distributed execution) - but its core
    loop is genuinely this shape. When you later read the LangGraph
    docs, you will recognise every concept.

MiniGraph and LangGraph share the same contract, so swapping is
mechanical: nodes take state and return dicts of updates; edges are
either static or decided by a router function.

THE GRAPH WE ARE BUILDING
-------------------------

        entry
          |
          v
     [proposer] <------------------+
          |                        |
          v                        | REVISE, and rounds
      [critic]                     | and budget remain
          |                        |
          +--- route_after_critic -+
                |     |
    APPROVE, or |     | deadlock: cap reached
    budget hit  |     | without agreement
                |     v
                |  [arbiter]  (runs at most once, terminal)
                |     |
                v     v
                  END

Two things worth noticing:

* The back-edge from critic to proposer is the entire reason this is a
  graph and not a pipeline. Cycles are what make it an agent system.
* The arbiter has no edge back into the loop. It is terminal by
  construction, so adding a third agent did not add a way to fail to
  terminate. That property was designed in, not discovered afterwards.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

from .agents import arbiter_node, critic_node, proposer_node, researcher_node
from .config import Settings
from .llm import LLM, TokenSink
from .tools import SearchProvider
from .state import DebateState, Decision

END = "__end__"

Node = Callable[[DebateState], dict[str, Any]]
Router = Callable[[DebateState], str]


# ---------------------------------------------------------------------------
# MiniGraph
# ---------------------------------------------------------------------------


class MiniGraph:
    """
    A minimal directed graph with cycles and conditional routing.

    Deliberately tiny. Read it once and you will understand the mental
    model behind every agent framework on the market.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, str] = {}          # static: after A, always go to B
        self._routers: dict[str, Router] = {}     # dynamic: after A, ask a function
        self._entry: str | None = None

    def add_node(self, name: str, fn: Node) -> "MiniGraph":
        if name == END:
            raise ValueError("'__end__' is reserved.")
        self._nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> "MiniGraph":
        self._edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: Router) -> "MiniGraph":
        """The router returns the NAME of the next node, or END."""
        self._routers[src] = router
        return self

    def set_entry(self, name: str) -> "MiniGraph":
        self._entry = name
        return self

    def validate(self) -> None:
        """Catch wiring mistakes at build time, not three rounds into a run."""
        if self._entry is None:
            raise ValueError("Graph has no entry point.")
        if self._entry not in self._nodes:
            raise ValueError(f"Entry point {self._entry!r} is not a node.")
        for src, dst in self._edges.items():
            if src not in self._nodes:
                raise ValueError(f"Edge from unknown node {src!r}.")
            if dst != END and dst not in self._nodes:
                raise ValueError(f"Edge to unknown node {dst!r}.")
        for name in self._nodes:
            if name not in self._edges and name not in self._routers:
                raise ValueError(
                    f"Node {name!r} has no outgoing edge - the run would "
                    f"stall there. Add an edge, or route it to END."
                )

    def stream(
        self, state: DebateState, *, max_steps: int = 50
    ) -> Iterator[tuple[str, DebateState]]:
        """
        Execute the graph, yielding (node_name, state) after every step.

        Streaming rather than returning a final value is what lets the
        Streamlit UI render each agent turn as it happens instead of
        staring at a spinner for 40 seconds. Same generator drives the
        CLI's live output.

        `max_steps` is a second, structural safety net underneath the
        semantic iteration cap in route_after_critic(). If a future
        router has a bug and ping-pongs forever, this stops it. Two
        independent guards, because loops are the failure mode that
        actually bites people.
        """
        self.validate()
        current = self._entry
        steps = 0

        while current != END:
            if steps >= max_steps:
                state.apply({"done": True, "stop_reason": "max_rounds",
                             "error": f"MiniGraph hit structural step limit ({max_steps})."})
                yield ("__guard__", state)
                return

            node_fn = self._nodes[current]
            try:
                updates = node_fn(state)
            except Exception as exc:  # keep partial transcript on failure
                # "Type: message", not repr(). repr escapes newlines, so a
                # carefully written multi-line diagnostic - "raise
                # AEGIS_REASONING_MAX_TOKENS, currently 2048" - arrives as one
                # unreadable line with literal \n in it. An error message is
                # part of the interface; mangling it wastes the effort that
                # went into making it actionable.
                state.apply({"done": True, "stop_reason": "error",
                             "error": f"{type(exc).__name__}: {exc}"})
                yield ("__error__", state)
                return

            state.apply(updates)
            steps += 1
            yield (current, state)

            if current in self._routers:
                current = self._routers[current](state)
            else:
                current = self._edges[current]

        if not state.done:
            state.apply({"done": True, "stop_reason": state.stop_reason or "approved"})

    def invoke(self, state: DebateState, *, max_steps: int = 50) -> DebateState:
        """Run to completion, discarding intermediate yields."""
        for _name, current_state in self.stream(state, max_steps=max_steps):
            state = current_state
        return state


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_critic(state: DebateState, *, use_arbiter: bool = True) -> str:
    """
    The only decision point in the graph. Read this function whenever you
    wonder "why did the debate stop?" - it is, by design, the complete
    answer.

    Conditions are checked in PRIORITY ORDER, and that order is load-
    bearing. Note especially that the budget guard is checked BEFORE the
    round cap routes to the Arbiter: the Arbiter costs a model call, and
    a guard that can be bypassed by the very path it is meant to catch is
    not a guard. When you have several stop conditions, always ask which
    one must win when two fire at once.
    """
    def _record(rule: str, next_node: str, reason: str, **extra: Any) -> None:
        """
        Write down the decision as it is taken.

        Every branch below records one of these before returning, so the
        transcript can always answer "why did it go there?" without anyone
        re-deriving the conditions. The `observed` dict carries the values
        the decision was actually made on - a reason without its inputs is
        an assertion, not evidence.
        """
        state.apply({"decisions": [Decision(
            at="critic",
            round=state.round,
            rule=rule,
            next_node=next_node,
            reason=reason,
            verdict=state.verdict,
            observed=extra,
        )]})

    # 1. Happy path. The critic is satisfied. Cheapest possible exit.
    if state.verdict == "APPROVE":
        # Routing still branches on the VERDICT alone - the ledger informs, it
        # does not decide (invariant: never route on anything but the parsed
        # enum). But an APPROVE with the Critic's own objections still open is
        # incoherent, and worth recording rather than smoothing over: it is a
        # measurable symptom of a critic that is not tracking its own points.
        still_open = [p.id for p in state.open_points]
        _record("approved", END,
                f"Critic approved on round {state.round}. "
                + (f"Note: {len(still_open)} of its own points were still open "
                   f"({', '.join(still_open)}) - an approval that contradicts "
                   f"its own ledger."
                   if still_open else "All points closed."),
                round=state.round, open_points=still_open,
                ledger=state.points_summary())
        state.apply({"done": True, "stop_reason": "approved"})
        return END

    # 2. Budget guard. Deliberately ahead of the arbiter branch below -
    #    if we are out of money we must not spend more, even to produce a
    #    nicer ending. Independent of round count, because one long-
    #    context round can cost more than five short ones.
    if state.max_cost_usd_exceeded():
        _record("budget", END,
                f"Spend ${state.cost_usd:.4f} reached the ${state.max_cost_usd:.2f} "
                f"ceiling. Stopped before escalating, because the Arbiter costs "
                f"a model call.",
                cost_usd=state.cost_usd, max_cost_usd=state.max_cost_usd)
        state.apply({"done": True, "stop_reason": "max_rounds",
                     "error": "Budget guard tripped."})
        return END

    # 3. Wall-clock guard. Sits beside the budget guard, and ABOVE the
    #    arbiter branch for precisely the same reason: escalation costs a
    #    model call, so a guard the escalation path can outrun is not a
    #    guard. This one is what actually protects a LOCAL run, where
    #    tokens are free and the scarce resource is your only GPU.
    if state.max_seconds_exceeded():
        _record("timeout", END,
                f"Inference time {state.elapsed_s:.1f}s reached the "
                f"{state.max_seconds:.0f}s ceiling. Local tokens are free, so "
                f"time is the limit that binds here.",
                elapsed_s=state.elapsed_s, max_seconds=state.max_seconds)
        state.apply({"done": True, "stop_reason": "max_rounds",
                     "error": "Wall-clock guard tripped."})
        return END

    # 4. Deadlock. Proposer and Critic did not converge within the cap
    #    (failure mode #1 from the strategy report). Rather than hand back
    #    an unreviewed draft, escalate to the Arbiter for a final ruling.
    if state.round >= state.max_rounds:
        if use_arbiter:
            _record("deadlock", "arbiter",
                    f"No agreement after {state.round} of {state.max_rounds} "
                    f"rounds. Escalating to the Arbiter for a binding ruling. "
                    f"Still contested: "
                    f"{', '.join(p.id for p in state.open_points) or 'nothing named'}.",
                    round=state.round, max_rounds=state.max_rounds,
                    open_points=[p.id for p in state.open_points])
            return "arbiter"
        _record("deadlock", END,
                f"Hit the {state.max_rounds}-round cap with the Arbiter "
                f"disabled. Returning the last unreviewed revision - the "
                f"weakest output this system produces.",
                round=state.round, max_rounds=state.max_rounds)
        state.apply({"done": True, "stop_reason": "max_rounds"})
        return END

    # Otherwise: back to the Proposer for revision. This is the cycle,
    # and the cycle is what makes this an agent system.
    _record("revise", "proposer",
            f"Critic requested changes on round {state.round}. "
            f"{state.max_rounds - state.round} round(s) left, so back to the "
            f"Proposer. Ledger: {state.points_summary()}.",
            round=state.round, rounds_remaining=state.rounds_remaining,
            open_points=[p.id for p in state.open_points])
    return "proposer"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_debate_graph(
    llm: LLM,
    settings: Settings,
    on_token: TokenSink | None = None,
    search: SearchProvider | None = None,
) -> MiniGraph:
    """
    Wire the Proposer/Critic debate.

    Note the closures: the nodes stored in the graph are zero-argument-
    beyond-state functions, with `llm` and `settings` already bound. This
    keeps the graph's Node type simple (state -> updates) while still
    letting each node reach its dependencies. It is also why tests can
    build a graph around a FakeLLM without touching global state.

    `on_token` rides along the same way, and that is the whole reason live
    output does not leak orchestration into the UI. The alternative - the
    frontend reaching into the LLM client to attach a listener - would put
    the UI on the far side of the one boundary this design exists to keep.
    A callback bound at graph-build time keeps the Node contract intact
    (state -> updates), so nothing about the engine changes shape just
    because someone wants to watch it work.
    """

    def _researcher(state: DebateState) -> dict[str, Any]:
        return researcher_node(state, search=search, settings=settings,
                               on_token=on_token)

    def _proposer(state: DebateState) -> dict[str, Any]:
        return proposer_node(state, llm=llm, settings=settings, on_token=on_token)

    def _critic(state: DebateState) -> dict[str, Any]:
        return critic_node(state, llm=llm, settings=settings, on_token=on_token)

    def _arbiter(state: DebateState) -> dict[str, Any]:
        return arbiter_node(state, llm=llm, settings=settings, on_token=on_token)

    graph = MiniGraph()
    graph.add_node("proposer", _proposer)
    graph.add_node("critic", _critic)
    graph.add_edge("proposer", "critic")

    # The Researcher, when enabled, is the ENTRY node with a single static
    # edge to the Proposer. Deliberately not a loop and not conditional:
    # evidence is gathered once, before the argument, so the rounds argue
    # over a fixed set of facts. A retrieval step inside the cycle would mean
    # an improvement between rounds could be a better argument or just better
    # search, with no way to tell which - and it would spend the context
    # budget the answer needs.
    if getattr(settings, "use_researcher", False) and search is not None:
        graph.add_node("researcher", _researcher)
        graph.add_edge("researcher", "proposer")
        graph.set_entry("researcher")
    else:
        graph.set_entry("proposer")

    use_arbiter = getattr(settings, "use_arbiter", True)
    if use_arbiter:
        graph.add_node("arbiter", _arbiter)
        # Terminal by construction. The Arbiter rules once and the debate
        # is over - there is deliberately no edge back into the loop.
        # A judge who can be appealed to repeatedly is just another
        # debater, and would reintroduce the non-termination we spent
        # the whole design avoiding.
        graph.add_edge("arbiter", END)

    graph.add_conditional_edges(
        "critic", lambda s: route_after_critic(s, use_arbiter=use_arbiter)
    )
    graph.validate()
    return graph
