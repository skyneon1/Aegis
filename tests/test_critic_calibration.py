"""
Is the Critic a working gate? Measured against a real model.

HOW THIS FILE DIFFERS FROM EVERY OTHER TEST HERE
------------------------------------------------
It reaches a real model on purpose, and `pytest.ini` DESELECTS it by default.

Everywhere else that is forbidden — `tests/test_app.py` says so in capitals,
after two of its cases accidentally started driving live inference and became
slow and flaky. That rule stands. This file is the deliberate exception, and
the distinction is what is under test: those cases test ORCHESTRATION, which a
model can only make slower and less deterministic. These test a MODEL'S
JUDGEMENT against a prompt, which cannot be tested without one.

    ./dev.sh test        # hermetic, ~5s, this file not run
    ./dev.sh test-live   # this file

Cases and scoring live in `aegis/calibration.py`, not here. They were
duplicated in both places for exactly one session before the two copies
started to differ, which is the usual lifespan of duplicated fixtures.

WHAT THE ASSERTIONS ENCODE
--------------------------
Two claims, and only one of them currently holds on gemma2:2b:

  * The gate CLOSES on materially wrong answers — asserted normally.
  * The gate OPENS on good ones — asserted as an EXPECTED FAILURE, because
    four prompt revisions established that gemma2:2b cannot do it (see the
    version table in aegis/agents.py). Marking it xfail rather than deleting
    it keeps the requirement visible and, being non-strict, turns a future
    improvement into an XPASS that says "this changed, update the docs"
    instead of silence.

A test suite that passed while the gate was broken would be worse than no
suite; so would one that failed forever and got ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aegis import build_llm, hostinfo, load_settings
from aegis.calibration import CALIBRATION_CASES, run_calibration

SETTINGS = load_settings(provider="ollama")
_models = hostinfo.available_models(SETTINGS.base_url)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        SETTINGS.critic_model not in _models,
        reason=(f"local model {SETTINGS.critic_model!r} not available "
                f"(found: {_models or 'ollama unreachable'})"),
    ),
]


@pytest.fixture(scope="module")
def report():
    """One calibration run, shared. Each case is a real model call."""
    settings = load_settings(provider="ollama")
    settings.reasoning_models = hostinfo.reasoning_models(_models, settings.base_url)
    return run_calibration(settings=settings, llm=build_llm(settings))


# ---------------------------------------------------------------------------
# The gate must close
# ---------------------------------------------------------------------------


def test_catches_obviously_broken_answers(report):
    """The floor. Any prompt passes this, which is why it proves little alone."""
    ok, n = report.score("gross")
    assert ok == n, [r.case_id for r in report.tier("gross") if not r.correct]


def test_catches_fluent_but_materially_wrong_answers(report):
    """
    The discrimination zone, and the only tier that separates a real gate
    from a prompt that leans one way. These answers read well: invented
    statistics stated precisely, a false mechanism inside a sound argument,
    and one that discusses the topic knowledgeably without ever answering
    the question.
    """
    ok, n = report.score("flawed")
    assert ok == n, [r.case_id for r in report.tier("flawed") if not r.correct]


# ---------------------------------------------------------------------------
# The gate must also open — which it currently does not
# ---------------------------------------------------------------------------


def test_approves_a_good_answer(report):
    """
    A gate that never opens carries no information: every debate hits the
    round cap, every answer is reported as contested, and 'contested' stops
    meaning anything. Both cases are correct and useful and deliberately
    NOT exhaustive, because incompleteness is not a defect.

    THIS WAS AN XFAIL UNTIL 2026-08-28, and the history is the point. On
    gemma2:2b it failed 0/2 through four prompt revisions that swung between
    approving nothing and approving everything, never finding a middle - a
    model limit, not a wording problem. It was marked xfail(strict=False)
    rather than deleted, precisely so that fixing the cause would surface as
    an XPASS saying "this changed" instead of as silence.

    That is what happened: llama3.2:3b scores 2/2 here and 100% balanced on
    the identical prompt, so the marker was promoted to a real assertion. An
    xfail is a placeholder for a known defect, not a permanent excuse - when
    it starts passing, it stops being an xfail.
    """
    ok, n = report.score("good")
    assert ok == n, [r.case_id for r in report.tier("good") if not r.correct]


# ---------------------------------------------------------------------------
# The instrument itself
# ---------------------------------------------------------------------------


def test_every_case_produced_a_parsed_verdict(report):
    """
    Independent of calibration: a case that returned neither APPROVE nor
    REVISE means the run broke (an empty completion, a timeout), and would
    otherwise be silently counted as a scoring miss.
    """
    assert len(report.results) == len(CALIBRATION_CASES)
    for r in report.results:
        assert r.actual in ("APPROVE", "REVISE"), (r.case_id, r.actual)


def test_the_diagnosis_names_a_failure_mode(report):
    """The report must say WHICH way it is broken; the two need opposite fixes."""
    assert report.diagnosis
    assert any(word in report.diagnosis for word in
               ("PERFECTIONISM", "RUBBER STAMP", "CALIBRATED", "PARTIAL"))
