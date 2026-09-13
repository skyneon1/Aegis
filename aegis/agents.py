"""
aegis.agents — the two debating agents, plus the verdict parser.

An "agent" here is deliberately unglamorous: a function that takes the
current DebateState, makes exactly one LLM call, and returns a dict of
state updates. No classes, no inheritance, no hidden memory.

    node(state, llm=..., settings=...) -> dict[str, Any]

Keeping nodes as pure-ish functions is what makes them testable in
isolation and trivially reorderable in the graph. When we add a
Researcher or a Coder later, they will have this exact same shape.

THE CENTRAL DESIGN DECISION IN THIS FILE
----------------------------------------
The Critic must emit a MACHINE-READABLE VERDICT.

If the critic replies "this is pretty good but maybe reconsider the
second point", the graph has to guess whether that means approve or
revise. Guessing means your control flow is now a vibe. So we:

  1. Instruct the critic to start with `VERDICT: APPROVE` or
     `VERDICT: REVISE` on the very first line.
  2. Parse that line with parse_verdict().
  3. Route ONLY on the parsed enum, never on the prose.
  4. Default to REVISE when parsing fails - the safe direction, since
     an extra round costs a few cents while a wrongly-approved bad
     answer is the actual failure we care about.

This is the single most important habit to carry into every agent you
ever build: constrain the model's output where the machine needs to make
a decision, and let it be free-form everywhere else.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .config import Settings
from .llm import LLM, TokenSink
from .tools import SearchProvider, format_sources
from dataclasses import replace

from .state import DebateState, Point, Turn

# ---------------------------------------------------------------------------
# Prompts. Kept as module-level constants so they are easy to find, diff in
# git, and A/B test later. Never inline a prompt at a call site.
# ---------------------------------------------------------------------------

# Appended to the Critic's prompt ONLY when there are prior points to rule on.
#
# WHY THIS IS CONDITIONAL, discovered from calibration data rather than
# designed: when the RULINGS format was shown unconditionally, both gemma2:2b
# and llama3.2:3b invented rulings on a FIRST-round answer that had no prior
# points - "P1: RESOLVED ... P2: OPEN ..." referring to objections nobody had
# made. The fabricated "P2: OPEN" then drove the verdict to REVISE, so a good
# answer was rejected on the strength of an imaginary objection.
#
# The engine was already robust to it (a ruling for an unknown id is ignored,
# and P-prefixed lines are excluded from new points), so the ledger never
# corrupted - but the VERDICT did. A format slot is an instruction: show a
# model a section and it will fill it in, whether or not it has anything to
# put there. Do not present a slot that cannot legitimately be filled.
CRITIC_ADVERSARIAL_CLAUSE = """

ADVERSARIAL REVIEW IS IN FORCE.

The bar for APPROVE is raised. In addition to everything above, each of
these is now MATERIAL and must be raised as a point:

  * a central claim resting on an assumption the answer never states. Not
    "it should say more" - the assumption is doing load-bearing work and a
    reader cannot see it,
  * a number, probability or comparison given without the derivation that
    produced it, where seeing that derivation would change how much weight
    the claim deserves,
  * a recommendation that never says under what conditions it would be the
    wrong recommendation.

And before you may answer APPROVE you must state, in one sentence, the
strongest argument AGAINST the answer's conclusion and why it does not
overturn it. If you cannot construct that argument, you have not finished
reviewing and the verdict is not yet APPROVE.

WHAT HAS NOT CHANGED: everything the rules above call a note is still a
note. "Could also mention", "lacks examples", "too vague", "would be
stronger if" remain forbidden as points in this mode too. This clause
raises the standard of RIGOUR, not of completeness - an answer is still
allowed to be short, and incomplete is still not wrong."""


CRITIC_RULINGS_CLAUSE = """

YOUR FIRST JOB IS TO CLOSE YOUR OWN POINTS.

Previous points are listed below. Rule on EVERY one of them before you raise
anything new, using this format after the VERDICT line:

RULINGS:
- P1: RESOLVED - <why it is now addressed>
- P2: OPEN - <what is still missing>
- P3: WITHDRAWN - <why the Proposer's rebuttal was right>

Exactly one of:
  RESOLVED  - the revision addressed it.
  WITHDRAWN - the Proposer disputed it and was right. Say so plainly; being
              wrong about an objection is normal, and withdrawing it is a
              working gate, not a loss.
  OPEN      - genuinely not addressed. Say what is still missing.

You are ruling on YOUR OWN previous objections, not hunting for new ones. A
point the Proposer disputed must be answered: either concede it (WITHDRAWN)
or explain why the rebuttal fails (OPEN).

Never re-raise a point you just closed. If you resolved P2, do not list the
same complaint again under NEW - it is settled, and reopening it means the
debate can never end.

Refer only to point numbers that appear in the list below. Do not invent
rulings on points that were never raised."""


# Appended to the Proposer and Critic prompts only when evidence exists.
# Kept as its own constant rather than inlined into both, so the citation
# contract is stated once and cannot drift between the agent that cites and
# the agent that checks the citation.
EVIDENCE_CLAUSE = """

You have been given SOURCES, each with a handle like [S1]. Use them:
- Cite the handle inline when a claim rests on a source, e.g. "roughly $70/month
  for a managed control plane [S1]".
- Never invent a citation. If no source supports a claim, say the claim is
  your own reasoning rather than attaching a handle to it.
- The sources are search snippets, not the full articles. Treat them as
  evidence about what is claimed elsewhere, not as proof, and do not pretend
  to more certainty than a snippet supports.
- A source that contradicts you is the most useful thing on the list. Say so."""

PROPOSER_SYSTEM = """You are the PROPOSER in a two-agent reasoning system.

Your job: give the best possible answer to the user's topic.

Rules:
- Lead with a direct answer. Do not open with throat-clearing.
- Support claims with reasoning the reader can check.
- State your assumptions explicitly rather than hiding them.
- Where you are uncertain, say so and say why. Confident wrongness is
  the failure mode you are being judged against.
- Be substantive but tight. No filler, no restating the question.

A Critic will attack your answer. Write something that survives that."""

REVISER_SYSTEM = """You are the PROPOSER in a two-agent reasoning system.

The Critic has raised numbered points against your answer. You must answer
EVERY open point by name, and then give the improved answer.

For each point choose exactly one stance:
  FIXED    - you accept it and changed the answer accordingly.
  DISPUTED - you think the point is wrong, and you say why. Defend your
             original position; do not change the answer to appease it.

Disputing is legitimate and expected. A Proposer that concedes every point
is not reasoning, it is capitulating - and the Critic is required to rule on
your disputes, so a good rebuttal can get a point withdrawn.

Reply in exactly this format:

RESPONSES:
- P1: FIXED - added the mechanism connecting X to Y.
- P2: DISPUTED - the figure is standard and the answer cites where it comes
  from; the objection asks for a source the claim does not need.

ANSWER:
<the complete revised answer, standalone>

Rules:
- One RESPONSES line per open point, using the Critic's own P-number.
- The ANSWER section must stand alone. No diff, no changelog, no thanks.
- Do not lose correct material from the previous version while fixing the
  flawed parts.
- If you FIXED a point, the change must actually be present in the ANSWER."""

# TUNED IN THREE MEASURED STEPS in session 4. The history matters, because
# each step fixed the previous step's failure and the endpoints are both
# useless:
#
#   v1  approved 1/8 debates, 0/5 on a deliberately strong answer. Every
#       objection was "lacks nuance" / "would be stronger if" - incompleteness
#       treated as defect. Cause: the rules listed "a missing counter-argument"
#       as grounds for REVISE, a condition every answer satisfies.
#
#   v2  approved 8/8, all on round 1, avg 1.12 rounds. Over-corrected into a
#       rubber stamp: it still caught cartoonish errors (fabricated history,
#       pure non-sequiturs) but missed a confident technical falsehood buried
#       in a sound argument, and missed an answer that discussed the topic
#       without answering the question - which its own rules call material.
#       Cause: thumb-on-scale wording ("most serviceable answers meet it",
#       "if you write 'strong, but' that is an APPROVE") let it approve on
#       fluency instead of inspection.
#
#   v3  swung back: 3/3 in the discrimination zone but 0/5 on a strong
#       answer, objecting that it "lacks concrete examples" - a phrase sitting
#       in its own exclusion list. The mandatory checks were stated first and
#       forcefully, and a 2B model followed the most recent, most emphatic
#       instruction rather than reconciling it with an earlier one.
#
#   v5  adds the RULINGS mechanic. The critic now receives its OWN previous
#       points and must close each one before raising anything new, which
#       changes the question it is answering from "what is wrong with this?"
#       (unbounded - there is always something) to "were my objections
#       answered?" (bounded, and therefore capable of converging). The
#       prompt-only versions below could not fix that, because it was never
#       only a wording problem: a critic with no memory of its own demands
#       cannot recognise that they were met. See state.Point.
#
#   v4  same checks, but the exclusion list is re-applied AFTER them
#       as a mechanical audit of the critic's own written reasons: name the
#       banned phrasings, delete any reason that uses them, and if nothing
#       survives, approve. Placing the counterweight last is the whole fix -
#       with a small model, ORDER is part of the instruction, not just
#       content. Keeps v2's exclusion list, which is what cured the
#       perfectionism, and removes v2's thumb on the scale. Adds a mandatory
#       three-question check that must be RUN before APPROVE is allowed, so
#       approval is earned by inspection rather than granted for reading well.
#       Every question in that check is about correctness or responsiveness -
#       none is about completeness, which is what keeps v1 from returning.
#
# Both degenerate outcomes look like success from one side: 100% approval
# looks like a system that works, 0% looks like rigour. Neither carries any
# information, and a distribution is the only way to see which you have.
CRITIC_SYSTEM = """You are the CRITIC in a two-agent reasoning system.

You are the only quality gate here, and a gate has TWO ways to fail.
Approving weak work lets it through as if it had passed review. Never
approving is equally a failure: your verdict stops carrying information,
every question burns the full round budget, and every answer gets marked
contested whether it is or not. You are useful only if both verdicts are
genuinely reachable.

Reply in exactly this format, VERDICT on the very first line:

VERDICT: REVISE
NEW:
- <the specific defect, in your own words about THIS answer>

BEFORE you may answer APPROVE, silently satisfy yourself of three things:
responsiveness (it answers the question asked, not a neighbouring one),
factual accuracy (every number, date, attribution and mechanism it states as
fact is true - confident phrasing is not evidence), and safety (a reader
acting on it would not be misled).

These are checks for you to RUN, not text to repeat. Never list them, quote
them, or turn them into points - a point must describe a specific defect in
THIS answer, in your own words. If all three hold and no point is OPEN, the
verdict is APPROVE.

Raise a NEW point only for a MATERIAL defect - one of:
  * a factual error, including an invented statistic or a wrong description
    of how something works. If a SOURCE contradicts the answer, cite it -
    "claims 47%, but [S2] reports 12%" is the strongest objection you can
    make, and far better than noting that a figure is unsupported,
  * a central claim the answer gives no support for and a reader could not
    check,
  * a failure to answer the question that was asked.

These are NOT grounds for a point. They are notes, and notes do not block:
  * "it could also mention X"
  * "it lacks specific examples"
  * "it is too vague" / "it lacks nuance"
  * "it would be stronger if..."
  * a counter-argument that exists but would not change the conclusion
  * a preference about structure, length, or tone

An answer is allowed to be incomplete. Incomplete is not wrong.

LAST STEP - audit your own points before you send them. Read each one and
ask: does it name something the answer SAYS that is wrong, or something the
answer DOES NOT SAY? If it contains "lacks", "does not mention", "could
include", "needs more", "no examples", "too vague", or "would be stronger",
it is about something absent. Absence is not a defect. Delete it.

If deleting leaves nothing OPEN and nothing NEW, the verdict is APPROVE.
Say so plainly. An answer you could not find a real fault in has passed.

Do not rewrite the answer yourself. That is the Proposer's job."""


ARBITER_SYSTEM = """You are the ARBITER in a multi-agent reasoning system.

A Proposer and a Critic have argued and failed to reach agreement within
the round limit. You are reading the full exchange and issuing the final
ruling. There is no appeal and no further round.

Your job is NOT to split the difference. Diplomatic hedging is the one
outcome that helps nobody here. Decide who was right, point by point.

Produce:

1. A short ruling on the substantive disagreements - for each open point,
   say whether the Critic's objection stands or the Proposer's position
   holds, and why.
2. The FINAL ANSWER to the original topic, written to stand alone. Fold in
   the Critic's objections you judged valid; ignore the ones you did not.
3. A brief, honest statement of remaining uncertainty - what would have to
   be true for this answer to be wrong.

Format:

RULING:
- <point>: <who was right and why>

FINAL ANSWER:
<the complete standalone answer>

REMAINING UNCERTAINTY:
<what is still genuinely unresolved, or "none material">

Rules:
- Do not restate the argument. Rule on it.
- If the Critic's objections were pedantic, say so plainly and keep the
  Proposer's answer.
- If the Proposer never fixed a real problem, fix it yourself."""


# ---------------------------------------------------------------------------
# Prompt identity
# ---------------------------------------------------------------------------

PROMPTS: dict[str, str] = {
    "proposer": PROPOSER_SYSTEM,
    "reviser": REVISER_SYSTEM,
    "critic": CRITIC_SYSTEM,
    "arbiter": ARBITER_SYSTEM,
}


def prompt_fingerprint() -> str:
    """
    A short hash of every system prompt in this module.

    WHY A REPORT NEEDS THIS
    -----------------------
    The settings snapshot saved beside each run records the model, the
    temperatures, and the round cap - on the stated principle that a
    transcript without the configuration that produced it is an anecdote
    rather than data. But it recorded everything EXCEPT the prompts, which
    are the single thing most likely to differ between two runs you are
    comparing.

    So `evaluate.py --compare A B` could tell you the outcome mix moved
    without being able to tell you that the prompts were what moved it -
    or worse, could compare two reports produced by the same prompt and
    invite you to attribute the difference to a change you had not made.

    A fingerprint is enough. The full prompt text lives in git; what a
    report needs is a stable identity it can be matched against.
    """
    blob = "\n\x00".join(f"{k}:{v}" for k, v in sorted(PROMPTS.items()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

# Tolerant on purpose. Models routinely decorate structured output with
# markdown even when told not to: "**VERDICT:** APPROVE", "## VERDICT - REVISE",
# "`VERDICT:` APPROVE". Treat *, _, `, #, :, - and whitespace between the
# label and the value as ignorable noise.
#
# Lesson worth internalising: when you ask a model for structure, write the
# PARSER defensively even though the PROMPT is strict. Prompts are requests,
# not guarantees. A parser that is brittle about formatting produces bugs
# that look like reasoning failures, which are far harder to diagnose.
_VERDICT_RE = re.compile(
    r"\bVERDICT\b[\s:\-*_`#]*(APPROVE|REVISE)\b", re.IGNORECASE
)


def parse_verdict(text: str) -> str:
    """
    Extract APPROVE / REVISE from critic output.

    Fails SAFE: anything unparseable becomes REVISE.

    Rationale for that default: the two error directions are not
    symmetric. Wrongly revising costs one extra round (a few cents and
    some seconds). Wrongly approving ships a bad answer as if it had
    passed quality control - which silently destroys the entire value of
    having a critic. When failure modes are asymmetric, always default
    toward the cheap one.
    """
    if not text:
        return "REVISE"

    match = _VERDICT_RE.search(text)
    if match:
        return match.group(1).upper()

    # Fallback: a bare APPROVE on its own line, tolerating models that
    # drop the label despite instructions.
    for line in text.strip().splitlines()[:3]:
        stripped = line.strip().strip("*#- ").upper()
        if stripped in ("APPROVE", "APPROVED"):
            return "APPROVE"
        if stripped in ("REVISE", "CHANGES_REQUESTED"):
            return "REVISE"

    return "REVISE"


# ---------------------------------------------------------------------------
# The point ledger: parsing the structured halves of the exchange
# ---------------------------------------------------------------------------

# "- P2: RESOLVED - the mechanism is now given."  Tolerant of markdown noise
# and of an em-dash, a hyphen, or nothing separating verdict from note.
_RULING_RE = re.compile(
    r"^[\s*\-•]*\**\s*(P\d+)\**\s*[:\-]\s*\**\s*"
    r"(RESOLVED|WITHDRAWN|OPEN|ADDRESSED|STILL\s+OPEN)\**"
    r"\s*[-–—:]?\s*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)

_RESPONSE_RE = re.compile(
    r"^[\s*\-•]*\**\s*(P\d+)\**\s*[:\-]\s*\**\s*"
    r"(FIXED|DISPUTED|ACCEPTED|REJECTED)\**"
    r"\s*[-–—:]?\s*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# Section headers we split on. Kept loose because models decorate headers.
_SECTION_RE = re.compile(
    r"^[\s*#>]*\**\s*(RULINGS?|NEW(?:\s+POINTS?)?|RESPONSES?|ANSWER|"
    r"VERDICT|REASONS?)\b\**\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Normalise the synonyms a model reaches for when it ignores the enum.
_RULING_ALIASES = {"ADDRESSED": "RESOLVED", "STILL OPEN": "OPEN"}
_STANCE_ALIASES = {"ACCEPTED": "FIXED", "REJECTED": "DISPUTED"}


def _sections(text: str) -> dict[str, str]:
    """Split structured output into its labelled sections."""
    if not text:
        return {}
    matches = list(_SECTION_RE.finditer(text))
    out: dict[str, str] = {}
    for n, match in enumerate(matches):
        name = re.sub(r"\s+", " ", match.group(1).upper())
        name = {"RULING": "RULINGS", "NEW POINTS": "NEW", "NEW POINT": "NEW",
                "RESPONSE": "RESPONSES", "REASON": "REASONS"}.get(name, name)
        end = matches[n + 1].start() if n + 1 < len(matches) else len(text)
        out[name] = text[match.end():end].strip()
    return out


def parse_rulings(text: str) -> dict[str, tuple[str, str]]:
    """
    Pull the Critic's rulings on existing points: {"P1": ("RESOLVED", note)}.

    Points the Critic did not mention are deliberately ABSENT from the result
    rather than defaulted. `apply_rulings` then leaves them open, which is the
    safe direction: a parse failure or a forgetful critic must never close an
    objection. Same asymmetry as parse_verdict - the cheap error is one extra
    round, the expensive one is shipping something as reviewed.
    """
    rulings: dict[str, tuple[str, str]] = {}
    for match in _RULING_RE.finditer(text or ""):
        verdict = re.sub(r"\s+", " ", match.group(2).upper())
        verdict = _RULING_ALIASES.get(verdict, verdict)
        rulings[match.group(1).upper()] = (verdict, match.group(3).strip())
    return rulings


def parse_new_points(text: str, *, strict: bool = False) -> list[str]:
    """
    The Critic's newly raised points.

    Read from the NEW section when there is one. Falling back to REASONS (and
    then to bare bullets) keeps the first round working, where there are no
    previous points and a model naturally just lists faults.

    `strict` disables that fallback, and exists because of a real bug: on
    `VERDICT: APPROVE\nREASONS:\n- Looks fine now.` the fallback read the
    approval's own justification as a fresh objection and invented a point
    from it. The bullets under REASONS mean opposite things depending on the
    verdict - grounds for rejecting, or grounds for accepting - so the same
    text cannot be parsed the same way in both cases. Callers pass strict=True
    when the verdict is APPROVE.
    """
    parts = _sections(text or "")
    body = parts.get("NEW") or ("" if strict else parts.get("REASONS") or "")
    if strict:
        return [line for line in extract_reasons(body)
                if not re.match(r"^\**\s*P\d+\b", line, re.IGNORECASE)]
    if not body:
        # No recognisable sections: treat top-level bullets as points, but
        # skip anything that is actually a ruling on an existing point.
        body = text or ""
        return [line for line in extract_reasons(body)
                if not re.match(r"^\**\s*P\d+\b", line, re.IGNORECASE)]
    return [line for line in extract_reasons(body)
            if not re.match(r"^\**\s*P\d+\b", line, re.IGNORECASE)]


def parse_responses(text: str) -> dict[str, tuple[str, str]]:
    """The Proposer's stance per point: {"P1": ("FIXED", note)}."""
    out: dict[str, tuple[str, str]] = {}
    for match in _RESPONSE_RE.finditer(text or ""):
        stance = match.group(2).upper()
        out[match.group(1).upper()] = (_STANCE_ALIASES.get(stance, stance),
                                       match.group(3).strip())
    return out


def extract_answer(text: str) -> str:
    """
    The ANSWER section of a Proposer reply, or the whole text if unlabelled.

    Fails OPEN, like extract_final_answer and for the same reason: this drives
    what the user reads and what the next agent is shown, not control flow.
    Showing slightly too much is cosmetic; showing nothing because a regex
    missed a header is not.

    The RESPONSES block is stripped when a real ANSWER section exists, because
    the answer must stand alone - the next Critic should judge the answer, not
    the Proposer's commentary about the last round.
    """
    if not text:
        return ""
    parts = _sections(text)
    if parts.get("ANSWER"):
        return parts["ANSWER"].strip()
    return text.strip()


def apply_responses(
    points: list[Point], responses: dict[str, tuple[str, str]]
) -> list[Point]:
    """
    Record the Proposer's stance on each open point. Returns a NEW list.

    A disputed point becomes `disputed`, not `resolved`: the Proposer does not
    get to close an objection against it. Only the Critic's ruling can do
    that, which is the whole reason stance and ruling are separate fields.
    """
    updated: list[Point] = []
    for point in points:
        stance, note = responses.get(point.id, ("", ""))
        if point.is_open and stance:
            updated.append(replace(
                point, proposer_stance=stance, proposer_note=note,
                status="disputed" if stance == "DISPUTED" else point.status,
            ))
        else:
            updated.append(point)
    return updated


def _normalise(text: str) -> set[str]:
    """Content words, for comparing whether two points say the same thing."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "that",
            "this", "for", "on", "with", "as", "be", "are", "no", "not", "but",
            "answer", "needs", "should", "does", "give", "gives", "given"}
    return {w for w in words if w not in stop and len(w) > 2}


def _prompt_phrases() -> list[set[str]]:
    """
    Normalised phrases from the Critic's own instructions.

    DERIVED from the prompt constants rather than hard-coded, so the filter
    cannot drift out of sync when the prompt is reworded - the failure mode a
    hand-copied blocklist guarantees eventually.
    """
    lines = []
    for block in (CRITIC_SYSTEM, CRITIC_ADVERSARIAL_CLAUSE,
                  CRITIC_RULINGS_CLAUSE, EVIDENCE_CLAUSE):
        for raw in block.splitlines():
            line = raw.strip(" -*•\t")
            if len(line.split()) >= 6:          # skip headers and fragments
                lines.append(_normalise(line))
    return [w for w in lines if len(w) >= 4]


_PROMPT_PHRASES: list[set[str]] | None = None


def _is_prompt_echo(text: str, similarity: float = 0.8) -> bool:
    """
    Is this "point" a near-verbatim quote of the instructions?

    OBSERVED IN THE UI, not caught by any test: llama3.2:3b turned the
    Critic's own pre-approval checklist into ledger entries - "Is every claim
    presented as fact actually true? Check numbers, statistics, dates..." was
    filed as P3. Three of four points in that run were echoes, which inflates
    the ledger, leaves fake points OPEN, and makes an approval look as though
    it contradicted itself. Third instance in this project of "show a model a
    shape and it produces that shape".

    WHY THE THRESHOLD IS HIGH, AND JACCARD
    --------------------------------------
    The first version used intersection/min at 0.6 and filtered a GENUINE
    objection: "The answer fails to address the question that was asked...
    they never directly answer whether Kubernetes is necessary" overlapped
    heavily with the prompt line describing that very defect category. Of
    course it did - the prompt names the category and a real point
    instantiates it, so they share vocabulary BY DESIGN.

    Silently dropping a real finding is far worse than keeping a junk one, so
    this filter is deliberately near-verbatim only: Jaccard (symmetric, not
    inflated by one side being short) at 0.8. That catches a quote and leaves
    paraphrase alone. The prompt rewording is the actual fix; this is a
    backstop, and a backstop that eats real evidence is not worth having.

    (Third time an intersection/min similarity measure has been wrong in this
    project, always in the direction that flattered it. Use Jaccard.)
    """
    global _PROMPT_PHRASES
    if _PROMPT_PHRASES is None:
        _PROMPT_PHRASES = _prompt_phrases()
    incoming = _normalise(text)
    if len(incoming) < 4:
        return False
    for phrase in _PROMPT_PHRASES:
        if len(incoming & phrase) / len(incoming | phrase) >= similarity:
            return True
    return False


def _is_duplicate(
    text: str,
    existing: list[Point],
    similarity: float = 0.45,
    containment: float = 0.85,
) -> bool:
    """
    Is this "new" point really one already on the ledger?

    WHY THE ENGINE HAS TO GUARD THIS
    --------------------------------
    Observed on a real run: the Critic closed P2 as RESOLVED and, in the very
    same reply, raised P3 with almost the same wording. The debate then
    deadlocked on a point it had just agreed was fixed. The prompt asks it not
    to; asking is not enough, because a model with no memory of its own
    reasoning will re-derive the same objection from the same answer. So the
    ledger enforces the property rather than hoping for it.

    TWO MEASURES, BECAUSE ONE WAS WRONG
    -----------------------------------
    The first version used intersection / min(len) with a 0.7 bar. Two real
    paraphrases of the same objection scored 0.68 and slipped through - but
    the fix is not simply a lower bar. `intersection / min` is inflated
    whenever one text is short and mostly contained in a longer one, so
    lowering it would start merging genuinely distinct objections about the
    same subject ("the figure has no source" vs "the figure contradicts the
    cited study").

    So: Jaccard (intersection / union) as the primary measure, which is
    symmetric and does not reward brevity - the two real paraphrases score
    0.50 on it. Containment is kept as a separate, much stricter test, for
    the case it is genuinely good at: a short point wholly restating a
    longer existing one.
    """
    incoming = _normalise(text)
    if not incoming:
        return True                      # nothing to add
    for point in existing:
        prior = _normalise(point.text)
        if not prior:
            continue
        shared = len(incoming & prior)
        if shared / len(incoming | prior) >= similarity:
            return True
        if shared / min(len(incoming), len(prior)) >= containment:
            return True
    return False


def apply_rulings(
    points: list[Point],
    rulings: dict[str, tuple[str, str]],
    new_texts: list[str],
    round_number: int,
) -> list[Point]:
    """
    Fold the Critic's rulings and new points into the ledger. Returns a NEW
    list; the caller's list is never mutated.

    Unmentioned points stay OPEN. That is not laziness in the parser - it is
    the same fail-safe direction as parse_verdict. If a critic forgets to rule
    on P2, the honest state of P2 is "still open", and the cost is one more
    round. Quietly resolving it would let a formatting slip pass unreviewed
    work.
    """
    updated: list[Point] = []
    for point in points:
        verdict, note = rulings.get(point.id, ("", ""))
        if point.is_open and verdict in ("RESOLVED", "WITHDRAWN"):
            updated.append(replace(point, status=verdict.lower(),
                                   critic_note=note, round_closed=round_number))
        elif point.is_open and verdict == "OPEN":
            updated.append(replace(point, status="open", critic_note=note))
        else:
            updated.append(point)

    # New ids continue the sequence so a P-number means one thing for the
    # whole debate, even across rounds and in a saved transcript.
    #
    # Deduplication compares against EVERY point, closed ones included. A
    # point the Critic already resolved must not come back as new - that is
    # the specific failure this guards, and checking only open points would
    # miss it entirely.
    next_id = len(points) + 1
    for text in new_texts:
        if _is_prompt_echo(text) or _is_duplicate(text, updated):
            continue
        updated.append(Point(id=f"P{next_id}", text=text,
                             round_raised=round_number))
        next_id += 1
    return updated


def format_points_for_prompt(points: list[Point]) -> str:
    """Render open points for whichever agent needs to see them."""
    lines: list[str] = []
    for point in points:
        if not point.is_open:
            continue
        lines.append(f"- {point.id}: {point.text}")
        if point.proposer_stance:
            lines.append(f"    Proposer says {point.proposer_stance}: "
                         f"{point.proposer_note or '(no reason given)'}")
    return "\n".join(lines)


def extract_reasons(text: str) -> list[str]:
    """Pull the bulleted reasons out of a critique, for UI display."""
    reasons: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "• ")):
            reasons.append(stripped[2:].strip())
        elif re.match(r"^\d+[.)]\s+", stripped):
            reasons.append(re.sub(r"^\d+[.)]\s+", "", stripped))
    return reasons


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


RESEARCHER_QUERY_MAX = 240


def build_research_query(topic: str) -> str:
    """
    Turn a debate topic into a search query.

    Deliberately NOT an LLM call. A model asked to write a query for its own
    topic mostly paraphrases the topic, so the extra turn buys a rewording at
    the cost of one more inference - which on this hardware is 5-10 seconds
    of the user's time for no measurable gain. Search engines handle
    natural-language questions well; the honest move is to pass the question
    through and spend the round budget on the argument instead.

    This is the sort of place agent systems accumulate expensive ceremony:
    a node exists, therefore it must call a model. It must not.
    """
    query = " ".join((topic or "").split())
    return query[:RESEARCHER_QUERY_MAX]


def researcher_node(
    state: DebateState,
    *,
    search: SearchProvider,
    settings: Settings,
    on_token: TokenSink | None = None,
) -> dict[str, Any]:
    """
    Gather evidence once, before the argument starts.

    ONCE, and not per round, for two reasons. The context budget is 4096
    tokens locally and evidence competes with the answer for it, so more
    retrieval is not free. And a fixed evidence set makes the rounds
    comparable: if the sources changed under the debate, an improvement
    between round 1 and round 3 could be better argument or just better
    search, and there would be no way to tell which.

    Note this node makes NO model call. It is a tool-using node, and its
    turn is recorded in the transcript like any other so the run stays
    auditable - a step that touches the outside world and leaves no trace is
    the one you will wish you could see later.
    """
    query = build_research_query(state.topic)
    result = search.search(query, limit=settings.research_results,
                           purpose=f"Find checkable evidence for a debate on: "
                                   f"{state.topic}")

    if result.ok:
        body = (f"Retrieved {len(result.sources)} source(s) via "
                f"{result.provider} in {result.latency_s:.1f}s.\n\n"
                + format_sources(result.sources))
    else:
        # A failed search is reported, not hidden. The debate continues
        # ungrounded, and the transcript says so - otherwise a run with no
        # citations looks identical to a run where retrieval broke.
        body = (f"No evidence retrieved via {result.provider}"
                + (f" — {result.error}" if result.error else "")
                + ".\n\nThe debate proceeds ungrounded: treat every "
                  "factual claim below as unverified.")

    if on_token is not None:
        on_token("researcher", body)

    turn = Turn(
        round=0,                     # round 0: before the argument begins
        agent="researcher",
        content=body,
        model=f"{result.provider} (search)",
        latency_s=result.latency_s,
        prompt_system="(tool node - no model call)",
        prompt_user=query,
    )

    return {
        "sources": result.sources,
        "research_query": query,
        "transcript": [turn],
        "elapsed_s": result.latency_s,
    }


def proposer_node(
    state: DebateState,
    *,
    llm: LLM,
    settings: Settings,
    on_token: TokenSink | None = None,
) -> dict[str, Any]:
    """
    Produce an answer. Handles BOTH the first draft and every revision -
    the only difference is which system prompt and context it gets.

    Note what is NOT here: the proposer never sees the full transcript.
    It sees its own last answer plus the latest critique. That is a
    deliberate context-budget decision (your report flags context
    overflow as a top failure mode) and it also keeps the model focused
    on the current problem instead of relitigating round one.
    """
    is_revision = bool(state.critique)

    if is_revision:
        system = REVISER_SYSTEM
        open_points = format_points_for_prompt(state.points)
        user = (
            f"TOPIC:\n{state.topic}\n\n"
            f"YOUR PREVIOUS ANSWER:\n{state.answer}\n\n"
            f"OPEN POINTS AGAINST IT:\n"
            f"{open_points or '(none listed - see the review below)'}\n\n"
            f"CRITIC'S FULL REVIEW:\n{state.critique}\n\n"
            f"Answer every open point by its P-number, then give the improved "
            f"standalone answer. (This is REVISION {state.round}.)"
        )
    else:
        system = PROPOSER_SYSTEM
        user = f"TOPIC:\n{state.topic}\n\nGive your best answer."

    if state.sources:
        system += EVIDENCE_CLAUSE
        user = (f"SOURCES:\n{format_sources(state.sources)}\n\n" + user)

    response = llm.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=settings.proposer_model,
        temperature=settings.proposer_temperature,
        max_tokens=settings.max_tokens,
        agent="proposer",
        on_token=on_token,
    )

    turn = Turn(
        round=state.round + 1,
        agent="proposer",
        content=response.text,
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        latency_s=response.latency_s,
        ttft_s=response.ttft_s,
        tokens_per_s=response.tokens_per_s,
        model=response.model,
        reasoning=response.reasoning,
        prompt_system=system,
        prompt_user=user,
    )

    # `answer` is the ANSWER section only; `content` on the turn keeps the
    # whole reply including the rebuttals. The next Critic must judge the
    # answer, not the Proposer's commentary about the previous round - but a
    # reader (and the UI) wants to see the rebuttals, so both are kept.
    return {
        "answer": extract_answer(response.text) if is_revision else response.text,
        "points": apply_responses(state.points, parse_responses(response.text)),
        "transcript": [turn],
        "tokens_in": response.tokens_in,
        "tokens_out": response.tokens_out,
        "cost_usd": response.cost_usd,
        "elapsed_s": response.latency_s,
    }


def critic_node(
    state: DebateState,
    *,
    llm: LLM,
    settings: Settings,
    on_token: TokenSink | None = None,
) -> dict[str, Any]:
    """
    Judge the current answer and emit a structured verdict.

    The critic owns the round counter. A round is only complete once
    BOTH agents have spoken, so incrementing here (rather than in the
    proposer) keeps `state.round` meaning something honest.
    """
    # The Critic sees its OWN open points. That is the change that makes this
    # a debate rather than a sequence of unrelated reviews: without them it
    # answers "what is wrong with this?" every round, which has no fixed
    # point, instead of "were my objections answered?", which does.
    prior = format_points_for_prompt(state.points)
    # Assemble only the sections that apply. Every clause a model is shown is
    # a slot it will try to fill, so an inapplicable one is not neutral.
    system = (CRITIC_SYSTEM
              + (CRITIC_ADVERSARIAL_CLAUSE
                 if settings.critic_strictness == "adversarial" else "")
              + (CRITIC_RULINGS_CLAUSE if prior else "")
              + (EVIDENCE_CLAUSE if state.sources else ""))
    user = (
        (f"SOURCES:\n{format_sources(state.sources)}\n\n" if state.sources else "")
        + f"TOPIC:\n{state.topic}\n\n"
        f"PROPOSER'S ANSWER:\n{state.answer}\n\n"
        + (f"YOUR PREVIOUS POINTS, STILL OPEN "
           f"(rule on every one before raising anything new):\n{prior}\n\n"
           if prior else "")
        + "Review it. Start your reply with the VERDICT line."
    )

    response = llm.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=settings.critic_model,
        temperature=settings.critic_temperature,
        max_tokens=settings.max_tokens,
        agent="critic",
        on_token=on_token,
    )

    verdict = parse_verdict(response.text)
    next_round = state.round + 1

    turn = Turn(
        round=next_round,
        agent="critic",
        content=response.text,
        verdict=verdict,
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        latency_s=response.latency_s,
        ttft_s=response.ttft_s,
        tokens_per_s=response.tokens_per_s,
        model=response.model,
        reasoning=response.reasoning,
        prompt_system=system,
        prompt_user=user,
    )

    return {
        "critique": response.text,
        "verdict": verdict,
        "round": next_round,
        # On APPROVE the REASONS bullets justify the approval; parsing them
        # as objections would manufacture points from an agreement.
        "points": apply_rulings(
            state.points, parse_rulings(response.text),
            parse_new_points(response.text, strict=(verdict == "APPROVE")),
            next_round),
        "transcript": [turn],
        "tokens_in": response.tokens_in,
        "tokens_out": response.tokens_out,
        "cost_usd": response.cost_usd,
        "elapsed_s": response.latency_s,
    }


# Marks the boundary between the Arbiter's reasoning and its final answer.
# The Arbiter's output is structured, so - exactly like the Critic's
# verdict - we parse it rather than hoping. Same discipline, same reason.
_FINAL_ANSWER_RE = re.compile(
    r"FINAL\s*ANSWER\s*[:\-]?\s*\n?(.*?)(?=\n\s*REMAINING\s*UNCERTAINTY|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def extract_final_answer(text: str) -> str:
    """
    Pull the FINAL ANSWER section out of an Arbiter ruling.

    Falls back to the whole text if the section marker is missing. That
    fallback is the RIGHT default here, and note how it differs from
    parse_verdict's: a verdict drives control flow, so ambiguity must
    resolve to the cautious branch. This is presentation - showing the
    user slightly more than intended is a cosmetic problem, whereas
    showing them nothing because a regex missed is a real one.

    Same principle both times: pick the failure you can live with.
    """
    if not text:
        return ""
    match = _FINAL_ANSWER_RE.search(text)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return text.strip()


def arbiter_node(
    state: DebateState,
    *,
    llm: LLM,
    settings: Settings,
    on_token: TokenSink | None = None,
) -> dict[str, Any]:
    """
    Break a deadlock. Runs at most ONCE per debate, and only when the
    Proposer and Critic failed to converge within the round cap.

    WHY THIS AGENT EXISTS
    ---------------------
    In v0.1, hitting the cap produced the system's weakest possible
    output: the last revision, never approved, handed over with a
    warning. That is precisely the wrong time to give up - a deadlock
    means the question was genuinely hard, which is when careful
    adjudication is worth the most.

    WHY IT SEES THE FULL TRANSCRIPT
    -------------------------------
    Unlike the Proposer (which sees only its last answer plus the latest
    critique, to protect the context budget), the Arbiter needs the whole
    argument. It is judging a disagreement, and you cannot judge an
    argument you have only seen the last line of. The cost is acceptable
    because this node runs at most once.
    """
    exchange: list[str] = []
    for turn in state.transcript:
        label = turn.agent.upper()
        if turn.verdict:
            label += f" ({turn.verdict})"
        exchange.append(f"--- ROUND {turn.round} · {label} ---\n{turn.content}")

    # Name the still-contested points explicitly. The Arbiter previously got
    # the raw transcript and had to work out what was actually in dispute -
    # which is exactly the inference the ledger now makes unnecessary, and the
    # one a small model is worst at.
    contested = format_points_for_prompt(state.points)
    user = (
        f"ORIGINAL TOPIC:\n{state.topic}\n\n"
        + (f"POINTS STILL CONTESTED (rule on each):\n{contested}\n\n"
           if contested else "")
        + f"FULL EXCHANGE ({state.round} rounds, ended without agreement):\n\n"
        + "\n\n".join(exchange)
        + "\n\nIssue your ruling and the final answer now."
    )

    response = llm.complete(
        [{"role": "system", "content": ARBITER_SYSTEM}, {"role": "user", "content": user}],
        model=settings.arbiter_model or settings.critic_model,
        temperature=settings.arbiter_temperature,
        max_tokens=settings.max_tokens,
        agent="arbiter",
        on_token=on_token,
    )

    turn = Turn(
        round=state.round,
        agent="arbiter",
        content=response.text,
        verdict="RULED",
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        latency_s=response.latency_s,
        ttft_s=response.ttft_s,
        tokens_per_s=response.tokens_per_s,
        model=response.model,
        reasoning=response.reasoning,
        prompt_system=ARBITER_SYSTEM,
        prompt_user=user,
    )

    return {
        "answer": extract_final_answer(response.text),
        "ruling": response.text,
        "done": True,
        "stop_reason": "arbitrated",
        "transcript": [turn],
        "tokens_in": response.tokens_in,
        "tokens_out": response.tokens_out,
        "cost_usd": response.cost_usd,
        "elapsed_s": response.latency_s,
    }
