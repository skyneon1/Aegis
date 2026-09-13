"""
aegis.calibration — measuring the CRITIC, not the answer.

WHY THIS IS A SEPARATE KIND OF MEASUREMENT
------------------------------------------
`aegis/evaluation.py` measures the process: how many debates approved, how
many rounds, what it cost. That is the right instrument for "is the machine
behaving", and it is blind to the failure that actually bit this project.

The Critic is the system's only quality gate, and a gate has two ways to
fail. It can approve nothing - so every debate runs the cap, every answer is
reported as contested, and "contested" stops meaning anything. Or it can
approve everything - so unreviewed work ships with a stamp on it. Both
produce a *clean-looking* outcome distribution: 0% approved looks like
rigour, 100% looks like a system that works. An eval that only counts
outcomes cannot tell either apart from a calibrated critic.

So this module judges the judge, with answers whose correct verdict is known
in advance. Three tiers, and the middle one is the point:

  good    - correct, useful, and deliberately INCOMPLETE. Must APPROVE.
            Incompleteness is not a defect, and a critic that cannot tell
            the difference will reject everything forever.
  flawed  - fluent, confident, well-structured, and materially wrong. Must
            REVISE. This is the discrimination zone and the only tier that
            distinguishes a real gate from a prompt that leans one way.
  gross   - obviously broken. Must REVISE. The floor. Any prompt catches
            these, which is exactly why passing this tier proves nothing on
            its own.

HOW THIS WAS USED, AND WHAT IT COST TO LEARN
--------------------------------------------
`CRITIC_SYSTEM` went through four revisions against these cases. Measured on
gemma2:2b: v1 approved 0/5 good answers, v2 approved 8/8 debates but missed
2/3 flawed answers, v3 and v4 went back to 0/5 good. Four attempts, never a
middle - which is itself the finding. A model that swings wholesale with the
lean of the wording may not be able to hold the distinction at all, and no
further prompt edit will conjure the capability. That conclusion is only
reachable because the tiers were measured separately; a single pass/fail
number would have looked like slow progress.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .agents import critic_node, prompt_fingerprint
from .config import Settings, load_settings
from .llm import LLM, build_llm
from .state import DebateState

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    tier: str                # "good" | "flawed" | "gross"
    topic: str
    answer: str
    expected: str            # "APPROVE" | "REVISE"
    note: str = ""           # what this case is testing for


# --- good: correct and useful, but NOT exhaustive -------------------------

_STRONG = """No. A two-person startup should not adopt Kubernetes.

Kubernetes solves problems you do not have yet: multi-service orchestration,
rolling deploys across many nodes, and self-healing infrastructure at a scale
where manual intervention stops being feasible. A two-person team has neither
the service count nor the traffic to need any of that.

The cost is concrete. Kubernetes adds a control plane to operate, a YAML
surface to learn, and a class of failure (networking, scheduling, resource
limits) unrelated to your product. At two people, one of you becomes a
part-time platform engineer, which is the most expensive way a small team can
spend its time.

Use a managed platform - Render, Fly.io, App Runner, or plain EC2 with a
process manager. Move to Kubernetes when you have a specific problem it
solves, most commonly when you outgrow one machine.

Assumption: your product is a conventional web service. If you are building a
platform whose customers deploy containers, Kubernetes may be the product
itself, and this answer does not apply."""

_THIN_BUT_CORRECT = """No, not always.

The phrase is Knuth's, and his actual claim was narrower than the slogan it
became: optimising the 97% of code that is not hot wastes effort and adds
complexity. He was arguing against optimising without measurement, not
against thinking about performance.

Two cases where early attention is correct. Architectural choices are
expensive to reverse - a data model or a synchronous-versus-queued decision
is not a local tweak later. And where a performance requirement is part of
the specification, meeting it is not premature, it is the requirement.

The defensible version: do not micro-optimise before profiling, and do
decide the things that are costly to change late."""

# --- flawed: fluent, confident, materially wrong -------------------------

_INVENTED_STATS = """For most teams, one approval is sufficient. Requiring two
increases median time-to-merge by 47% while reducing escaped defects by only
3.2%, according to widely replicated industry data. The marginal reviewer
contributes little because the first reviewer catches 91% of issues. Reserve
two-approval rules for changes touching authentication or billing, where the
cost of an escaped defect is high enough to justify the delay."""

_FALSE_MECHANISM = """For a 10-person team, a monorepo is usually the better
default. Atomic cross-project commits remove a whole class of version-skew
bugs, and one CI configuration is far less work than ten. The usual objection
is scale, but that is overstated: Git's performance is independent of
repository size because it stores full snapshots rather than deltas, so a
monorepo never slows down as it grows. Adopt a polyrepo only when teams
genuinely need independent release cadences."""

_ADJACENT_QUESTION = """Authentication is one of the most important parts of any
application. A good auth system needs secure password hashing with bcrypt or
argon2, protection against timing attacks, rate limiting on login endpoints,
secure session tokens, and a well-designed password reset flow that does not
leak account existence. Multi-factor authentication is increasingly expected.
Getting these details right is what separates a robust implementation from a
fragile one."""

# --- gross: obviously broken ---------------------------------------------

_FABRICATED = """Yes. Kubernetes was created by Microsoft in 2003 and is
required by law for any company processing payments in the EU. It runs only on
Windows Server, and every startup with fewer than five employees receives it
free from the Linux Foundation. Without it you cannot deploy a web application
at all."""

_UNSUPPORTED = """Yes, absolutely. Teams that reach 100% coverage ship 40% fewer
bugs and move three times faster. It is the single strongest predictor of
engineering quality, and every high-performing organisation enforces it.
Anything less guarantees production incidents."""

_NON_ANSWER = """Version control is a fascinating topic with a long history. Git
was released in 2005 and uses a directed acyclic graph. Branching models vary
widely between teams, and there are many opinions about commit message
formatting. Continuous integration is also important."""


CALIBRATION_CASES: list[CalibrationCase] = [
    CalibrationCase(
        "good-strong", "good",
        "Should a two-person startup use Kubernetes?", _STRONG, "APPROVE",
        "Direct, reasoned, scoped, states its own assumption. Not exhaustive - "
        "nothing is. An answer like this MUST be able to pass."),
    CalibrationCase(
        "good-thin", "good",
        "Is premature optimization always bad?", _THIN_BUT_CORRECT, "APPROVE",
        "Deliberately short and gives no examples. Tests the rule that "
        "incomplete is not wrong."),

    CalibrationCase(
        "flawed-invented-stats", "flawed",
        "Should code review require two approvals instead of one?",
        _INVENTED_STATS, "REVISE",
        "Precise-sounding statistics with no possible source. Confident "
        "phrasing is not evidence."),
    CalibrationCase(
        "flawed-false-mechanism", "flawed",
        "Is a monorepo better than a polyrepo for a 10-person team?",
        _FALSE_MECHANISM, "REVISE",
        "Sound argument containing a false technical claim: Git snapshots do "
        "not make repository size free. The conclusion may be right and the "
        "reasoning is still wrong."),
    CalibrationCase(
        "flawed-adjacent", "flawed",
        "Should a startup write its own auth system instead of using a vendor?",
        _ADJACENT_QUESTION, "REVISE",
        "Knowledgeable about the topic, never answers build-vs-buy. The "
        "hardest miss to catch, because it reads like expertise."),

    CalibrationCase(
        "gross-fabricated", "gross",
        "Should a two-person startup use Kubernetes?", _FABRICATED, "REVISE",
        "Invented history and invented law."),
    CalibrationCase(
        "gross-unsupported", "gross",
        "Is 100% unit test coverage a meaningful goal?", _UNSUPPORTED, "REVISE",
        "Every claim a bare superlative."),
    CalibrationCase(
        "gross-non-answer", "gross",
        "Is a monorepo better than a polyrepo for a 10-person team?",
        _NON_ANSWER, "REVISE",
        "Trivia in place of an answer."),
]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    case_id: str
    tier: str
    expected: str
    actual: str
    latency_s: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.actual == self.expected


@dataclass
class CalibrationReport:
    model: str
    prompt_fingerprint: str
    results: list[CalibrationResult] = field(default_factory=list)
    duration_s: float = 0.0

    def tier(self, name: str) -> list[CalibrationResult]:
        return [r for r in self.results if r.tier == name]

    def score(self, name: str) -> tuple[int, int]:
        rows = self.tier(name)
        return sum(r.correct for r in rows), len(rows)

    def recall(self, expected: str) -> tuple[int, int]:
        """How many cases expecting `expected` got it."""
        rows = [r for r in self.results if r.expected == expected]
        return sum(r.correct for r in rows), len(rows)

    @property
    def balanced_accuracy(self) -> float:
        """
        Mean per-VERDICT recall, not overall accuracy and not per-tier.

        Three candidate metrics, and only one is symmetrical:

          overall accuracy - six of the eight cases expect REVISE, so a
              critic stuck on REVISE scores 75% and looks respectable.
          per-tier average - better, but still lopsided: two tiers expect
              REVISE and one expects APPROVE, so always-REVISE scores 67%
              against always-APPROVE's 33%. (This module's first version
              used exactly that and claimed the two came out level. They
              did not, which is a good argument for testing your metric
              against known-degenerate inputs before trusting it.)
          per-verdict recall - averages "of the answers that should pass,
              how many did" with "of the answers that should fail, how many
              did". Both degenerate critics score exactly 50%, because both
              have thrown away exactly one of the two things a gate is for.

        A calibrated critic has to score on both halves, which is the whole
        point: the metric must not let you buy one failure mode cheaply to
        avoid the other.
        """
        classes = [c for c in ("APPROVE", "REVISE") if self.recall(c)[1]]
        if not classes:
            return 0.0
        return round(sum(self.recall(c)[0] / self.recall(c)[1] for c in classes)
                     / len(classes), 3)

    @property
    def diagnosis(self) -> str:
        """Name the failure mode, since the two need opposite fixes."""
        good_ok, good_n = self.score("good")
        bad = self.tier("flawed") + self.tier("gross")
        bad_ok = sum(r.correct for r in bad)
        if good_n and good_ok == 0:
            return ("PERFECTIONISM - approves nothing. Every debate will hit "
                    "the round cap and every answer will be reported as "
                    "contested, so the outcome signal carries no information.")
        if bad and bad_ok == 0:
            return ("RUBBER STAMP - rejects nothing. Unreviewed work ships "
                    "with a stamp on it. The more dangerous of the two.")
        if self.balanced_accuracy >= 0.85:
            return "CALIBRATED - both verdicts reachable and discriminating."
        weak = [t for t in ("good", "flawed", "gross")
                if self.tier(t) and self.score(t)[0] < self.score(t)[1]]
        return f"PARTIAL - weakest on: {', '.join(weak)}"

    def summary(self) -> str:
        lines = [f"Critic calibration — {self.model}  "
                 f"(prompt {self.prompt_fingerprint})"]
        for name, expect in (("good", "APPROVE"), ("flawed", "REVISE"),
                             ("gross", "REVISE")):
            if not self.tier(name):
                continue
            ok, n = self.score(name)
            lines.append(f"  {name:<7} expect {expect:<8} {ok}/{n}")
            for r in self.tier(name):
                mark = "ok  " if r.correct else "MISS"
                lines.append(f"      {mark} {r.case_id:<26} -> {r.actual} "
                             f"({r.latency_s:.0f}s)")
        ok_a, n_a = self.recall("APPROVE")
        ok_r, n_r = self.recall("REVISE")
        lines.append(f"  recall: APPROVE {ok_a}/{n_a} · REVISE {ok_r}/{n_r}")
        lines.append(f"  balanced accuracy: {self.balanced_accuracy:.0%}  "
                     f"(50% = one verdict abandoned entirely)")
        lines.append(f"  diagnosis: {self.diagnosis}")
        lines.append(f"  duration: {self.duration_s:.0f}s")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_fingerprint": self.prompt_fingerprint,
            "balanced_accuracy": self.balanced_accuracy,
            "diagnosis": self.diagnosis,
            "duration_s": self.duration_s,
            "tiers": {t: list(self.score(t)) for t in ("good", "flawed", "gross")},
            "recall": {c: list(self.recall(c)) for c in ("APPROVE", "REVISE")},
            "results": [vars(r) for r in self.results],
        }


def run_calibration(
    *,
    settings: Settings | None = None,
    llm: LLM | None = None,
    critic_model: str = "",
    cases: list[CalibrationCase] | None = None,
    tiers: tuple[str, ...] = ("good", "flawed", "gross"),
    on_result: Any = None,
) -> CalibrationReport:
    """
    Run the calibration set through `critic_node` and score it.

    Deliberately calls the SAME node the graph calls, with the same settings
    object - not a copy of the prompt or a simplified request. A calibration
    that exercises a parallel code path measures the parallel path.
    """
    settings = settings or load_settings()
    if critic_model:
        settings.critic_model = critic_model
    llm = llm or build_llm(settings)

    selected = [c for c in (cases or CALIBRATION_CASES) if c.tier in tiers]
    report = CalibrationReport(model=settings.critic_model,
                               prompt_fingerprint=prompt_fingerprint())
    started = time.perf_counter()

    for case in selected:
        t0 = time.perf_counter()
        state = DebateState(topic=case.topic, answer=case.answer)
        updates = critic_node(state, llm=llm, settings=settings)
        turn = updates["transcript"][0]
        result = CalibrationResult(
            case_id=case.case_id, tier=case.tier, expected=case.expected,
            actual=updates["verdict"], latency_s=round(time.perf_counter() - t0, 2),
            reasons=[l.strip().lstrip("-*• ").strip()
                     for l in turn.content.splitlines()
                     if l.strip().startswith(("-", "*", "•"))][:3],
        )
        report.results.append(result)
        if on_result:
            on_result(case, result)

    report.duration_s = round(time.perf_counter() - started, 2)
    return report
