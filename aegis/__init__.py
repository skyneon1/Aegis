"""
Aegis — a multi-agent orchestration platform.

v1: two agents (Proposer and Critic) debate a topic until the Critic
approves or the iteration cap is reached.

PUBLIC API
----------
    from aegis import run_debate, stream_debate

    state = run_debate("Should small teams use microservices?")
    print(state.answer)

Everything the frontends (cli.py, app.py) need is exported from here.
Neither frontend imports from aegis.graph or aegis.agents directly -
that keeps the UI decoupled from the internals, so we can restructure
the graph without touching a line of UI code.
"""

from __future__ import annotations

from typing import Any, Iterator

from . import catalog, credentials, hostinfo, providers, tools
from .config import (
    PROVIDERS,
    Settings,
    forget_api_key,
    keystore_path,
    load_settings,
    save_api_key,
)
from .graph import END, MiniGraph, build_debate_graph, route_after_critic
from .agents import extract_final_answer, extract_reasons, parse_verdict
from .llm import FakeLLM, LLM, LLMResponse, OpenAICompatibleLLM, TokenSink, build_llm
from .state import DebateState, Decision, Turn
from .tools import (
    FakeSearch,
    SearchProvider,
    SearchResult,
    Source,
    TinyFishSearch,
    build_search,
    format_sources,
)
from .transcript import new_run_id, save_run, to_markdown
from .evaluation import (
    DEFAULT_TOPICS,
    EvalCase,
    EvalReport,
    EvalResult,
    compare_reports,
    load_cases,
    load_report,
    run_eval,
    save_report,
)

__version__ = "0.3.0"

__all__ = [
    "catalog",
    "credentials",
    "hostinfo",
    "providers",
    "tools",
    "forget_api_key",
    "keystore_path",
    "save_api_key",
    "FakeSearch",
    "SearchProvider",
    "SearchResult",
    "Source",
    "TinyFishSearch",
    "build_search",
    "format_sources",
    "run_debate",
    "stream_debate",
    "prepare_debate",
    "DebateState",
    "Decision",
    "Turn",
    "TokenSink",
    "Settings",
    "load_settings",
    "PROVIDERS",
    "build_llm",
    "FakeLLM",
    "LLM",
    "LLMResponse",
    "OpenAICompatibleLLM",
    "build_debate_graph",
    "route_after_critic",
    "parse_verdict",
    "extract_reasons",
    "extract_final_answer",
    "MiniGraph",
    "END",
    "save_run",
    "to_markdown",
    "new_run_id",
    # Phase 0.3 — eval harness
    "DEFAULT_TOPICS",
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "run_eval",
    "load_cases",
    "save_report",
    "load_report",
    "compare_reports",
]


def _prepare(
    topic: str,
    *,
    settings: Settings | None = None,
    llm: LLM | None = None,
    search: SearchProvider | None = None,
    max_rounds: int | None = None,
    provider: str | None = None,
    on_token: TokenSink | None = None,
    **overrides: Any,
) -> tuple[MiniGraph, DebateState, Settings]:
    """Shared setup for both the streaming and blocking entry points."""
    if not topic or not topic.strip():
        raise ValueError("A debate needs a topic.")

    settings = settings or load_settings(provider=provider, **overrides)
    if max_rounds is not None:
        settings.max_rounds = max_rounds

    llm = llm or build_llm(settings)
    search = search or (build_search(settings) if settings.use_researcher else None)

    state = DebateState(
        topic=topic.strip(),
        max_rounds=settings.max_rounds,
        max_cost_usd=settings.max_cost_usd,
        max_seconds=settings.max_seconds,
        run_id=new_run_id(),
    )
    return build_debate_graph(llm, settings, on_token, search), state, settings


def stream_debate(topic: str, **kwargs: Any) -> Iterator[tuple[str, DebateState]]:
    """
    Run a debate, yielding (node_name, state) after each agent turn.

    Use this when you want live output - the Streamlit UI and the CLI's
    default mode both consume this. Streaming is not a cosmetic nicety
    here: a three-round debate against a real model takes 30-90 seconds,
    and watching the agents argue is most of the value of building this.
    """
    graph, state, _settings = _prepare(topic, **kwargs)
    yield from graph.stream(state)


def prepare_debate(topic: str, **kwargs: Any) -> tuple[MiniGraph, DebateState, Settings]:
    """
    Build a run without starting it: (graph, state, resolved settings).

    Exposed because a UI needs to show what it is ABOUT to do - the models
    it resolved, the guards in force - before a single token is generated.
    Without this, a frontend has to either duplicate load_settings' merge
    order or display its own guess about the configuration, and a
    configuration display that is a guess is worse than none.
    """
    return _prepare(topic, **kwargs)


def run_debate(topic: str, **kwargs: Any) -> DebateState:
    """
    Run a debate to completion and return the final state.

    Blocking counterpart to stream_debate(). Use in scripts, tests, and
    the future batch evaluator.
    """
    graph, state, _settings = _prepare(topic, **kwargs)
    return graph.invoke(state)
