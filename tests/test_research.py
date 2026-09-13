"""
Retrieval and the Researcher node — tested with no network and no key.

WHY THESE ARE HERMETIC
----------------------
Same split as everywhere else in this suite. Whether a search engine returns
anything USEFUL is a question about the engine, and answering it needs the
real one. Whether evidence reaches the prompts, survives a failed search, and
lands in the transcript is a question about our plumbing, and a fake answers
it in milliseconds and identically on every machine.

`FakeSearch` exists for the same reason `FakeLLM` does, and it is not a
testing nicety: it lets the grounding machinery be built and debugged
separately from the retrieval backend. Mixing the two is how you end up
debugging someone else's API while convinced you have a prompt bug.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aegis import (
    DebateState,
    FakeLLM,
    FakeSearch,
    Source,
    TinyFishSearch,
    build_debate_graph,
    build_search,
    format_sources,
    load_settings,
)
from aegis.agents import build_research_query, researcher_node

APPROVE = "VERDICT: APPROVE\nREASONS:\n- Fine.\n"
REVISE = "VERDICT: REVISE\nREASONS:\n- Unsupported claim.\n"


def _settings(**kw):
    s = load_settings(provider="fake")
    s.use_researcher = True
    s.max_cost_usd = 0.0
    for k, v in kw.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def test_fake_search_needs_no_key_and_no_network():
    """A fresh clone must ground a debate before it asks for a credential."""
    result = FakeSearch().search("anything")
    assert result.ok and result.sources and result.provider == "fake"


def test_build_search_falls_back_to_fake_without_a_key():
    """
    Grounding is an enrichment, not a dependency. A missing optional
    credential must not stop a debate that would otherwise run.
    """
    settings = load_settings(provider="fake")
    settings.tinyfish_api_key = ""
    assert build_search(settings).name == "fake"


def test_build_search_uses_tinyfish_when_a_key_exists():
    settings = load_settings(provider="fake")
    settings.tinyfish_api_key = "sk-test"
    assert build_search(settings).name == "tinyfish"


def test_research_defaults_off_without_a_key():
    """
    Keyed on the CREDENTIAL, not a separate flag, and the two are not
    independent. Enabling research with no key would ground the debate in
    FakeSearch's canned snippets - agents citing sources that do not exist,
    which is far worse than not grounding at all.
    """
    import os
    saved = os.environ.pop("TINYFISH_API_KEY", None)
    try:
        assert load_settings(provider="ollama").use_researcher is False
    finally:
        if saved is not None:
            os.environ["TINYFISH_API_KEY"] = saved


def test_a_bad_key_returns_an_error_rather_than_raising():
    """
    Observability and enrichment must never crash the thing they serve.
    A dead search engine degrades the debate; it must not end it.
    """
    result = TinyFishSearch("", base_url="https://example.invalid").search("x")
    assert result.ok is False and result.error


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


def test_the_research_query_is_built_without_a_model_call():
    """
    A model asked to write a search query for its own topic mostly
    paraphrases the topic, so the extra turn buys a rewording for 5-10s of
    local inference. Agent systems accumulate ceremony exactly here: a node
    exists, therefore it must call a model. It must not.
    """
    assert build_research_query("  Should a  team use K8s? ") == \
        "Should a team use K8s?"


def test_a_very_long_topic_is_truncated():
    assert len(build_research_query("word " * 500)) <= 240


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------


def test_the_researcher_records_a_turn_and_the_sources():
    state = DebateState(topic="Should a two-person startup use Kubernetes?")
    updates = researcher_node(state, search=FakeSearch(), settings=_settings())

    assert len(updates["sources"]) == 2
    assert updates["research_query"].startswith("Should a two-person")
    turn = updates["transcript"][0]
    assert turn.agent == "researcher"
    # Round 0: it speaks before the argument begins.
    assert turn.round == 0
    # A step that touches the outside world and leaves no trace is the one
    # you will wish you could see later.
    assert "[S1]" in turn.content


def test_a_failed_search_says_so_instead_of_looking_grounded(caplog):
    """
    A run with no citations must not be indistinguishable from a run where
    retrieval broke. The transcript has to record which happened.
    """
    state = DebateState(topic="t")
    updates = researcher_node(state, search=FakeSearch(error="engine down"),
                              settings=_settings())
    assert updates["sources"] == []
    body = updates["transcript"][0].content
    assert "No evidence retrieved" in body and "engine down" in body
    assert "ungrounded" in body


def test_sources_survive_serialisation():
    state = DebateState(topic="t")
    state.apply(researcher_node(state, search=FakeSearch(), settings=_settings()))
    assert len(state.to_dict()["sources"]) == 2


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_the_researcher_is_the_entry_node_when_enabled():
    graph = build_debate_graph(FakeLLM(), _settings(), None, FakeSearch())
    assert graph._entry == "researcher"
    # One static edge onward, and no edge back: evidence is gathered once so
    # the rounds argue over a fixed set of facts.
    assert graph._edges["researcher"] == "proposer"
    assert "researcher" not in graph._routers


def test_without_research_the_proposer_is_still_the_entry():
    graph = build_debate_graph(FakeLLM(), _settings(use_researcher=False))
    assert graph._entry == "proposer"
    assert "researcher" not in graph._nodes


def test_evidence_reaches_both_agents():
    """
    The Critic needs it more than the Proposer: without evidence its only
    possible objection is that a claim is UNSUPPORTED, never that it is
    WRONG. A reviewer with no access to sources can audit form but never
    substance.
    """
    settings = _settings(max_rounds=1, use_arbiter=False)
    llm = FakeLLM(script=["draft", APPROVE])
    graph = build_debate_graph(llm, settings, None, FakeSearch())
    graph.invoke(DebateState(topic="t", max_rounds=1, max_cost_usd=0.0))

    # calls[0] is the proposer, calls[1] the critic; the researcher makes none.
    assert len(llm.calls) == 2, "the researcher must not call a model"
    for call in llm.calls:
        prompts = " ".join(m["content"] for m in call["messages"])
        assert "[S1]" in prompts
        assert "Never invent a citation" in prompts


def test_an_ungrounded_debate_carries_no_evidence_clause():
    """The citation contract must not appear when there is nothing to cite."""
    settings = _settings(max_rounds=1, use_arbiter=False, use_researcher=False)
    llm = FakeLLM(script=["draft", APPROVE])
    build_debate_graph(llm, settings).invoke(
        DebateState(topic="t", max_rounds=1, max_cost_usd=0.0))
    for call in llm.calls:
        prompts = " ".join(m["content"] for m in call["messages"])
        assert "[S1]" not in prompts and "Never invent a citation" not in prompts


def test_a_failed_search_still_lets_the_debate_finish():
    """Degradation, not failure. The whole point of never raising."""
    settings = _settings(max_rounds=1, use_arbiter=False)
    llm = FakeLLM(script=["draft", APPROVE])
    graph = build_debate_graph(llm, settings, None, FakeSearch(error="down"))
    state = graph.invoke(DebateState(topic="t", max_rounds=1, max_cost_usd=0.0))
    assert state.stop_reason == "approved"
    assert state.sources == []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_sources_are_numbered_for_citation():
    """
    Numbered handles rather than bare URLs: a small model asked to reproduce
    a long URL inside prose will mangle it, and a mangled citation is worse
    than none because it still looks checkable.
    """
    rendered = format_sources([
        Source("T1", "https://a.example/very/long/path", "snip one", "a.example", 1),
        Source("T2", "https://b.example", "snip two", "b.example", 2),
    ])
    assert "[S1]" in rendered and "[S2]" in rendered
    assert "snip one" in rendered


def test_a_source_without_a_site_falls_back_to_its_host():
    assert Source("t", "https://host.example/x", "s").label == "host.example"
