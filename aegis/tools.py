"""
aegis.tools — retrieval, so the agents can check a claim instead of asserting.

WHY THIS EXISTS, AND WHY NOW
----------------------------
The measured failure of this system is invented facts. `aegis/calibration.py`
grades the Critic on exactly that: an answer stating "requiring two approvals
increases time-to-merge by 47%" is a plausible sentence and a fabricated
number, and the hardest calibration tier is full of them.

But look at what the Critic could actually say about it. Its best objection
was "the answer gives no support for the 40% figure" - a complaint about
ABSENCE, because absence was the only thing it could detect. It had no way to
say "the figure is wrong, here is the real one". A reviewer with no access to
evidence can only ever audit form, never substance, and that is a ceiling no
prompt reaches past.

So: retrieval. Evidence goes into the state, both agents see it, and an
objection can cite a source. This does not make a small model clever, and it
is not claimed to. It changes what the argument is ABOUT - from "does this
sound supported?" to "does this match what the sources say?" - which is the
difference between two models sharing a hunch and a debate with facts in it.

WHY SNIPPETS AND NOT FULL PAGES
-------------------------------
TinyFish also exposes a Fetch endpoint that returns whole pages, and on this
hardware that is the wrong tool. The local context window is 4096 tokens
(see Settings.num_ctx - on a 4GB card it is a VRAM dial, not a preference).
One fetched article would consume the entire budget the Proposer needs for
the topic, its previous answer, the open points, and the critique.

Search snippets are ~200 characters and already carry the checkable part -
the number, the date, the claim. The constraint picks the granularity here,
and it happens to pick well.

DESIGN RULES
------------
* Provider-agnostic, for the same reason `llm.py` is: the retrieval backend
  must be a config change. TinyFish today, Apify or Brave or SerpAPI later.
* Never raises. Retrieval is an enrichment, not a dependency - a debate with
  no evidence must still run. Every failure degrades to "no results", the
  same discipline as `hostinfo.py`.
* A FAKE provider that works with no key and no network, so the offline path
  and the tests stay hermetic. Same fake-first principle as `llm.FakeLLM`;
  the moment retrieval becomes mandatory, a fresh clone stops running.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

_TIMEOUT_S = 25.0


@dataclass
class Source:
    """One retrieved result. Deliberately small - see WHY SNIPPETS above."""

    title: str
    url: str
    snippet: str
    site: str = ""
    position: int = 0

    @property
    def label(self) -> str:
        """Short citation handle an agent can refer to in prose."""
        return self.site or urllib.parse.urlparse(self.url).netloc or "source"

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet,
                "site": self.site, "position": self.position}


@dataclass
class SearchResult:
    query: str
    sources: list[Source] = field(default_factory=list)
    latency_s: float = 0.0
    provider: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.sources)

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "provider": self.provider,
                "latency_s": self.latency_s, "error": self.error,
                "sources": [s.to_dict() for s in self.sources]}


class SearchProvider(Protocol):
    """The entire surface area an agent may depend on."""

    name: str

    def search(self, query: str, *, limit: int = 5,
               purpose: str = "") -> SearchResult: ...


# ---------------------------------------------------------------------------
# TinyFish
# ---------------------------------------------------------------------------


class TinyFishSearch:
    """
    TinyFish Search: GET https://api.search.tinyfish.ai with an X-API-Key.

    Note the header name. It is not `Authorization: Bearer`, which is what
    every other credential in this project uses - a detail worth stating in
    code rather than discovering from a 401, since the failure looks like a
    bad key rather than a bad header.
    """

    name = "tinyfish"

    def __init__(self, api_key: str,
                 base_url: str = "https://api.search.tinyfish.ai") -> None:
        self._key = api_key
        self._base = base_url.rstrip("/")

    def search(self, query: str, *, limit: int = 5,
               purpose: str = "") -> SearchResult:
        started = time.perf_counter()
        if not self._key:
            return SearchResult(query=query, provider=self.name,
                                error="no TINYFISH_API_KEY set")

        params = {"query": query, "domain_type": "web"}
        if purpose:
            params["purpose"] = purpose[:2000]      # documented cap
        url = f"{self._base}?{urllib.parse.urlencode(params)}"

        try:
            request = urllib.request.Request(
                url, headers={"X-API-Key": self._key,
                              "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as resp:
                payload = json.loads(resp.read())
        except Exception as exc:                    # noqa: BLE001 - see DESIGN RULES
            return SearchResult(
                query=query, provider=self.name,
                latency_s=round(time.perf_counter() - started, 3),
                error=f"{type(exc).__name__}: {exc}")

        sources = [
            Source(title=str(r.get("title") or "").strip(),
                   url=str(r.get("url") or "").strip(),
                   snippet=" ".join(str(r.get("snippet") or "").split()),
                   site=str(r.get("site_name") or "").strip(),
                   position=int(r.get("position") or 0))
            for r in (payload.get("results") or [])
        ]
        # Drop results with nothing checkable in them. A title alone is not
        # evidence, and padding the prompt with empty snippets spends context
        # the local model does not have.
        sources = [s for s in sources if s.snippet][:limit]

        return SearchResult(query=query, sources=sources, provider=self.name,
                            latency_s=round(time.perf_counter() - started, 3))


# ---------------------------------------------------------------------------
# Fake
# ---------------------------------------------------------------------------


class FakeSearch:
    """
    Scripted retrieval. No network, no key, no cost.

    Exists for the same reason FakeLLM does, and it is not a testing nicety:
    it lets the grounding PLUMBING - state, prompts, citation rendering, the
    graph edge - be developed and tested separately from whether a search
    engine returns anything useful. Two different categories of problem, and
    mixing them is how you end up debugging someone's API while believing you
    have a prompt bug.
    """

    name = "fake"

    def __init__(self, sources: list[Source] | None = None,
                 error: str = "") -> None:
        self._sources = sources if sources is not None else [
            Source(title="Kubernetes for Small Teams: Is It Worth the Complexity?",
                   url="https://example.com/k8s-small-teams",
                   snippet="EKS charges $0.10/hour for the control plane, and a "
                           "reliable production cluster typically needs four "
                           "engineers to maintain.",
                   site="example.com", position=1),
            Source(title="Why Kubernetes Is Overkill for Small Teams",
                   url="https://example.com/overkill",
                   snippet="Unless you are running 50+ microservices with "
                           "complex scaling needs, managed platforms cut "
                           "deployment issues substantially.",
                   site="example.com", position=2),
        ]
        self._error = error
        self.calls: list[str] = []

    def search(self, query: str, *, limit: int = 5,
               purpose: str = "") -> SearchResult:
        self.calls.append(query)
        if self._error:
            return SearchResult(query=query, provider=self.name,
                                error=self._error)
        return SearchResult(query=query, sources=self._sources[:limit],
                            provider=self.name, latency_s=0.0)


# ---------------------------------------------------------------------------
# Rendering evidence into a prompt
# ---------------------------------------------------------------------------


def format_sources(sources: list[Source]) -> str:
    """
    Render sources for an agent, numbered so they can be cited as [S1].

    Numbered handles rather than bare URLs on purpose: a small model asked to
    reproduce a long URL inside prose will mangle it, and a mangled citation
    is worse than none because it looks checkable and is not.
    """
    lines = []
    for n, s in enumerate(sources, start=1):
        lines.append(f"[S{n}] {s.label} — {s.title}\n     {s.snippet}")
    return "\n".join(lines)


def build_search(settings: Any) -> SearchProvider:
    """
    Factory. The graph asks for a provider and does not care which it gets.

    Falls back to FakeSearch when no key is configured rather than raising:
    grounding is an enrichment, and a missing optional credential must not
    stop a debate that would otherwise run.
    """
    key = getattr(settings, "tinyfish_api_key", "")
    if key:
        return TinyFishSearch(key)
    return FakeSearch()
