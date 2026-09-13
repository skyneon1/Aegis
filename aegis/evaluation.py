"""
aegis.evaluation — Phase 0.3: run N topics, score outcomes, compare
prompt/config versions with data.

WHY THIS FILE EXISTS
---------------------
Every debate already produces a `stop_reason` - approved / arbitrated /
max_rounds - and the README is explicit that these are not equally
trustworthy. Reading that field on ONE run tells you something about ONE
topic. Reading it on fifty tells you something about the SYSTEM: does
this prompt version tend to rubber-stamp? Does raising max_rounds
actually change the outcome mix, or just the cost? Those are the only
questions that let you change a prompt on evidence instead of vibes -
precisely the gap transcript.py's own docstring names: "you cannot
evaluate runs you did not record."

This module is a thin CONSUMER of aegis's public API (`run_debate`). It
adds zero new orchestration of its own - no verdict parsing, no routing,
no LLM calls. It runs the existing engine N times and counts what came
back. If you ever find yourself calling an LLM directly in here, that
call belongs in llm.py instead.

WHAT THIS DELIBERATELY DOES NOT DO
-----------------------------------
It does not grade answer QUALITY (no "was this a good answer" LLM
judge). That is a real and useful thing to add later, but it is a
different kind of measurement from what lives here: process metrics
(which of the three outcomes did we get, how many rounds, what did it
cost) versus content metrics (was the answer actually good). Conflating
them is the same mistake tests/test_graph.py already warns against -
"is the orchestration correct" and "is the output good" are different
questions and tangling them is why agent projects stall. Bolt a judge on
top of this later; do not smuggle one into the counting logic now.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import Settings, load_settings
from .state import DebateState
from .agents import prompt_fingerprint
from .transcript import new_run_id

# A handful of topics spanning different kinds of disagreement - technical
# tradeoff, empirical claim, values question, process question - so that
# `python3 evaluate.py --provider fake` exercises more than one shape of
# debate out of the box. Mirrors the single-run CLI's "no API key, no
# installs" promise: a fresh clone can eval before it can pay for tokens.
DEFAULT_TOPICS: list[str] = [
    "Should a two-person startup use Kubernetes?",
    "Is premature optimization always bad?",
    "Should junior engineers be allowed to use AI coding assistants unsupervised?",
    "Is a monorepo better than a polyrepo for a 10-person engineering team?",
    "Should a startup write its own auth system instead of using a vendor?",
    "Is 100% unit test coverage a meaningful goal?",
    "Should code review require two approvals instead of one?",
    "Is it better to rewrite a legacy system from scratch or refactor it incrementally?",
]


def _slug(text: str, max_len: int = 40) -> str:
    """Turn a topic into a filesystem/JSON-friendly id when none is given."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "case"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Cases and results
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """One topic to run through the debate graph."""

    topic: str
    case_id: str = ""  # defaults to a slug of the topic if left blank

    def __post_init__(self) -> None:
        if not self.case_id:
            self.case_id = _slug(self.topic)


@dataclass
class EvalResult:
    """What one case produced, boiled down to the numbers that matter."""

    case_id: str
    topic: str
    stop_reason: str
    round: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_s: float  # sum of every turn's latency in this run
    error: str
    run_id: str

    @classmethod
    def from_state(cls, case: "EvalCase", state: DebateState) -> "EvalResult":
        return cls(
            case_id=case.case_id,
            topic=case.topic,
            stop_reason=state.stop_reason,
            round=state.round,
            tokens_in=state.tokens_in,
            tokens_out=state.tokens_out,
            cost_usd=state.cost_usd,
            latency_s=round(sum(t.latency_s for t in state.transcript), 3),
            error=state.error,
            run_id=state.run_id,
        )


@dataclass
class EvalReport:
    """
    Aggregate over every case in one eval run.

    Kept as a plain dataclass (not a loose bag of dict/list locals) for
    the same reason DebateState is: it is one explicit object you can
    print, save, reload, and hand to compare_reports().
    """

    eval_id: str
    settings_snapshot: dict[str, Any]
    results: list[EvalResult] = field(default_factory=list)
    started_at: str = ""
    duration_s: float = 0.0

    # Which prompt set produced these numbers. The settings snapshot records
    # the model and the round cap but not the prompts - and prompts are the
    # thing most likely to differ between two reports you are comparing. An
    # empty value means "recorded before fingerprinting existed", not
    # "identical to yours".
    prompt_fingerprint: str = ""

    # ---- derived metrics ---------------------------------------------

    @property
    def outcome_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            key = r.stop_reason or "error"
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def total_cost_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.results), 6)

    @property
    def avg_cost_usd(self) -> float:
        return round(self.total_cost_usd / len(self.results), 6) if self.results else 0.0

    @property
    def avg_rounds(self) -> float:
        return round(sum(r.round for r in self.results) / len(self.results), 2) if self.results else 0.0

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "settings": self.settings_snapshot,
            "prompt_fingerprint": self.prompt_fingerprint,
            "n_cases": len(self.results),
            "outcome_counts": self.outcome_counts,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_usd": self.avg_cost_usd,
            "avg_rounds": self.avg_rounds,
            "error_count": self.error_count,
            "results": [asdict(r) for r in self.results],
        }

    def summary(self, label: str = "") -> str:
        """Human-readable table for the terminal / a saved report.md."""
        n = len(self.results)
        lines = [f"Eval {self.eval_id}" + (f"  ({label})" if label else "")]
        lines.append(f"  cases: {n}   duration: {self.duration_s:.1f}s")
        lines.append("  outcome distribution:")
        # Canonical outcomes first (matches state.StopReason), in the same
        # order the README's outcome table uses; anything unexpected still
        # shows up afterward rather than being silently dropped.
        seen = set()
        for outcome in ("approved", "arbitrated", "max_rounds", "error"):
            seen.add(outcome)
            if outcome not in self.outcome_counts:
                continue
            count = self.outcome_counts[outcome]
            pct = (count / n * 100) if n else 0.0
            lines.append(f"    {outcome:<12} {count:>3}/{n}  ({pct:5.1f}%)")
        for outcome, count in self.outcome_counts.items():
            if outcome in seen:
                continue
            lines.append(f"    {outcome:<12} {count:>3}/{n}")
        lines.append(
            f"  avg rounds: {self.avg_rounds}   avg cost: ${self.avg_cost_usd:.4f}"
            f"   total cost: ${self.total_cost_usd:.4f}"
        )
        if self.error_count:
            lines.append(f"  ERRORS: {self.error_count}/{n} cases raised an error")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading cases
# ---------------------------------------------------------------------------


def load_cases(path: str | Path) -> list[EvalCase]:
    """
    Read topics from a file.

    Two formats, chosen by extension:
      .txt   - one topic per line. Blank lines and lines starting with
               '#' are ignored. Simple, greppable, diffable in git - the
               same philosophy as keeping prompts as module constants
               instead of burying them in config.
      .jsonl - one JSON object per line, at least {"topic": "..."}.
               Use this when you want a STABLE case_id independent of the
               topic wording, so lightly rewording a topic doesn't start
               a new series in compare_reports's per-case diff.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix == ".jsonl":
        cases: list[EvalCase] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(EvalCase(topic=obj["topic"], case_id=obj.get("id", "")))
        return cases

    cases = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(EvalCase(topic=line))
    return cases


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_eval(
    cases: Iterable[EvalCase] | None = None,
    *,
    settings: Settings | None = None,
    provider: str | None = None,
    workers: int = 1,
    on_result: Callable[[EvalCase, DebateState], None] | None = None,
    **overrides: Any,
) -> EvalReport:
    """
    Run every case through the debate graph and collect results.

    Each case gets its OWN fresh DebateState and its OWN fresh LLM client,
    exactly like a standalone `python cli.py` invocation (both go through
    aegis.run_debate). That is deliberate, not an unoptimised shortcut: it
    means eval results do not depend on case ORDER, and it means FakeLLM's
    heuristic mode (which counts calls on the instance - see llm.py)
    behaves IDENTICALLY here and in a one-off run. What you saw for one
    topic in the CLI is what you'll see as row one of an eval. Sharing a
    single LLM/graph across cases would save a little object-construction
    overhead and cost exactly that guarantee.

    `workers > 1` runs cases concurrently via a thread pool. Safe because
    every case is fully independent (own state, own client) and the work
    is I/O-bound - waiting on the network - which is exactly the situation
    threads help with in Python despite the GIL. Mind provider rate limits
    before turning this up against a real API.
    """
    cases = list(cases) if cases is not None else [EvalCase(topic=t) for t in DEFAULT_TOPICS]
    if not cases:
        raise ValueError("No eval cases given.")

    base_settings = settings or load_settings(provider=provider, **overrides)

    # Local import: aegis/__init__.py defines run_debate AFTER importing
    # this module, so importing it at module load time would be circular.
    # By the time run_eval() actually executes, aegis is fully initialised.
    from . import run_debate

    def _run_one(case: EvalCase) -> EvalResult:
        try:
            state = run_debate(case.topic, settings=base_settings)
        except Exception as exc:
            # One bad case (empty topic, a transient client error building
            # the LLM, ...) must not take the whole batch down - that
            # would turn an eval run into an all-or-nothing gamble.
            state = DebateState(
                topic=case.topic, done=True, stop_reason="error",
                error=f"{type(exc).__name__}: {exc}",
            )
        if on_result:
            on_result(case, state)
        return EvalResult.from_state(case, state)

    report = EvalReport(
        eval_id=new_run_id(),
        settings_snapshot=base_settings.redacted(),
        started_at=_now_iso(),
        prompt_fingerprint=prompt_fingerprint(),
    )

    started = time.perf_counter()
    if workers <= 1:
        for case in cases:
            report.results.append(_run_one(case))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, case): case for case in cases}
            for future in as_completed(futures):
                report.results.append(future.result())
        # Restore input order so a saved report reads the same regardless
        # of which thread happened to finish first - reproducible to read
        # and to diff, even though execution itself was concurrent.
        order = {c.case_id: i for i, c in enumerate(cases)}
        report.results.sort(key=lambda r: order.get(r.case_id, 0))

    report.duration_s = round(time.perf_counter() - started, 2)
    return report


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_report(report: EvalReport, directory: str | Path = "runs/evals") -> Path:
    """
    Write <directory>/<eval_id>/report.json (+ report.md).

    Deliberately separate from transcript.save_run: a report is about the
    BATCH. Individual per-case transcripts are a different, optional
    artifact - use run_eval's `on_result` hook (with transcript.save_run)
    if you want those too.
    """
    out_dir = Path(directory) / report.eval_id
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "report.json"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(_to_markdown(report), encoding="utf-8")
    return json_path


def load_report(path: str | Path) -> EvalReport:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results = [EvalResult(**r) for r in data["results"]]
    return EvalReport(
        eval_id=data["eval_id"],
        settings_snapshot=data.get("settings", {}),
        results=results,
        started_at=data.get("started_at", ""),
        duration_s=data.get("duration_s", 0.0),
        prompt_fingerprint=data.get("prompt_fingerprint", ""),
    )


def _to_markdown(report: EvalReport) -> str:
    lines = [
        f"# Aegis eval — {report.eval_id}",
        "",
        report.summary().replace("\n", "  \n"),
        "",
        "| case | outcome | rounds | cost | tokens | topic |",
        "|---|---|---|---|---|---|",
    ]
    for r in report.results:
        tokens = r.tokens_in + r.tokens_out
        lines.append(
            f"| `{r.case_id}` | {r.stop_reason or 'error'} | {r.round} | "
            f"${r.cost_usd:.4f} | {tokens:,} | {r.topic[:60]} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparing two reports - Phase 0.3's other stated goal: "compare prompt
# versions with data" instead of by feel.
# ---------------------------------------------------------------------------


def compare_reports(a: EvalReport, b: EvalReport, label_a: str = "A", label_b: str = "B") -> str:
    """
    Render a side-by-side diff of two eval reports.

    Intended use: run the SAME case file through two Settings (a reworded
    prompt, a different model, a different max_rounds) and see whether the
    outcome mix and cost actually moved - instead of eyeballing a handful
    of transcripts and guessing whether a change helped.
    """
    lines = [f"Comparing {label_a} ({a.eval_id}) vs {label_b} ({b.eval_id})", ""]

    # State the provenance BEFORE the numbers. A comparison of two reports
    # that share a prompt fingerprint and a model is measuring noise, and
    # you want to know that before you start attributing a delta to a
    # change you did not actually make.
    fa = a.prompt_fingerprint or "unrecorded"
    fb = b.prompt_fingerprint or "unrecorded"
    lines.append(f"{'prompts':<12} {fa:>10} {fb:>10}   "
                 f"{'CHANGED' if fa != fb else 'identical'}")
    ma = a.settings_snapshot.get("critic_model", "?")
    mb = b.settings_snapshot.get("critic_model", "?")
    lines.append(f"{'critic model':<12} {ma:>10} {mb:>10}   "
                 f"{'CHANGED' if ma != mb else 'identical'}")
    if fa == fb and ma == mb:
        lines.append("  ! prompts and model both identical - any delta below "
                     "is run-to-run variance, not evidence.")
    lines.append("")

    outcomes = sorted(set(a.outcome_counts) | set(b.outcome_counts))
    lines.append(f"{'outcome':<12} {label_a:>10} {label_b:>10}   delta")
    for outcome in outcomes:
        na = a.outcome_counts.get(outcome, 0)
        nb = b.outcome_counts.get(outcome, 0)
        lines.append(f"{outcome:<12} {na:>10} {nb:>10}   {nb - na:+d}")

    lines.append("")
    lines.append(
        f"avg rounds     {a.avg_rounds:>10} {b.avg_rounds:>10}   "
        f"{b.avg_rounds - a.avg_rounds:+.2f}"
    )
    lines.append(
        f"avg cost ($)   {a.avg_cost_usd:>10.4f} {b.avg_cost_usd:>10.4f}   "
        f"{b.avg_cost_usd - a.avg_cost_usd:+.4f}"
    )
    lines.append(
        f"total cost ($) {a.total_cost_usd:>10.4f} {b.total_cost_usd:>10.4f}   "
        f"{b.total_cost_usd - a.total_cost_usd:+.4f}"
    )

    if a.error_count or b.error_count:
        lines.append(f"errors         {a.error_count:>10} {b.error_count:>10}")

    # Per-case flips are the most actionable signal here: same topic,
    # different outcome, means the change actually altered a real
    # decision - not just shifted an aggregate percentage around.
    by_id_a = {r.case_id: r for r in a.results}
    by_id_b = {r.case_id: r for r in b.results}
    flips = [
        (cid, ra.stop_reason, by_id_b[cid].stop_reason)
        for cid, ra in by_id_a.items()
        if cid in by_id_b and ra.stop_reason != by_id_b[cid].stop_reason
    ]
    if flips:
        lines.append("")
        lines.append(f"per-case outcome changes ({len(flips)}):")
        for cid, ra_outcome, rb_outcome in flips:
            lines.append(f"  {cid}: {ra_outcome} -> {rb_outcome}")

    return "\n".join(lines)
