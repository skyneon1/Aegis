"""
Critic strictness modes (aegis/agents.py, aegis/config.py).

HERMETIC. These assert the WIRING — that the mode reaches the prompt, that
the default is unchanged, and that the anti-nitpick rules survive into the
strict mode. They deliberately do NOT assert that adversarial mode is more
accurate: that is a claim about a model's behaviour, it can only be settled
by measurement, and aegis/calibration.py is the thing that settles it.

    ./dev.sh calibrate --strictness adversarial
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aegis import load_settings  # noqa: E402
from aegis.agents import (  # noqa: E402
    CRITIC_ADVERSARIAL_CLAUSE,
    CRITIC_SYSTEM,
    critic_node,
)
from aegis.llm import FakeLLM  # noqa: E402
from aegis.state import DebateState  # noqa: E402


class _CapturingLLM(FakeLLM):
    """A FakeLLM that remembers the system prompt it was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.system = ""

    def complete(self, messages, **kw):
        self.system = next(
            (m["content"] for m in messages if m["role"] == "system"), "")
        return super().complete(messages, **kw)


def _system_for(strictness: str) -> str:
    settings = load_settings(provider="fake")
    settings.critic_strictness = strictness
    llm = _CapturingLLM()
    state = DebateState(topic="t", answer="an answer", max_rounds=3)
    critic_node(state, llm=llm, settings=settings)
    return llm.system


# ---------------------------------------------------------------------------
# The default must not move
# ---------------------------------------------------------------------------


def test_calibrated_is_the_default():
    """
    The default prompt is the one measured at 100% balanced accuracy. A new
    mode must not silently become the default, or that measurement stops
    describing what actually runs.
    """
    assert load_settings(provider="fake").critic_strictness == "calibrated"


def test_the_calibrated_prompt_is_unchanged_by_the_new_mode():
    system = _system_for("calibrated")
    assert CRITIC_ADVERSARIAL_CLAUSE not in system
    assert system.startswith(CRITIC_SYSTEM)


def test_an_unrecognised_mode_falls_back_to_calibrated(monkeypatch):
    """A typo in the environment must not quietly select a stricter gate."""
    monkeypatch.setenv("AEGIS_CRITIC_STRICTNESS", "extremely-strict")
    assert load_settings(provider="fake").critic_strictness == "calibrated"


def test_the_environment_can_select_adversarial(monkeypatch):
    monkeypatch.setenv("AEGIS_CRITIC_STRICTNESS", "adversarial")
    assert load_settings(provider="fake").critic_strictness == "adversarial"


# ---------------------------------------------------------------------------
# The strict mode reaches the model
# ---------------------------------------------------------------------------


def test_adversarial_mode_reaches_the_critic_prompt():
    system = _system_for("adversarial")
    assert CRITIC_ADVERSARIAL_CLAUSE in system


def test_adversarial_mode_keeps_the_anti_nitpick_rules():
    """
    The strict clause raises the standard of RIGOUR, not of completeness.
    If it let "lacks examples" back in it would reproduce the perfectionism
    failure the base prompt exists to prevent - every debate hitting the
    round cap, every answer reported contested, the verdict carrying no
    information.
    """
    system = _system_for("adversarial")
    assert "Absence is not a defect" in system
    assert "incomplete is still not wrong" in CRITIC_ADVERSARIAL_CLAUSE


def test_the_strict_clause_is_covered_by_the_echo_filter():
    """
    The filter that stops the Critic quoting its own instructions back as a
    "point" derives its phrases from the prompt constants. A clause left out
    of that list is one the critic can echo verbatim and have accepted.
    """
    from aegis.agents import _normalise, _prompt_phrases

    phrases = _prompt_phrases()

    # Derive the clause's own phrases the same way the filter does, rather
    # than pinning a literal sentence — a pinned line breaks on any rewording
    # and would fail for a reason unrelated to the rule being defended.
    clause_phrases = [
        _normalise(line.strip(" -*\u2022\t"))
        for line in CRITIC_ADVERSARIAL_CLAUSE.splitlines()
        if len(line.strip(" -*\u2022\t").split()) >= 6
    ]
    assert clause_phrases, "the clause should contribute filterable lines"
    assert all(p in phrases for p in clause_phrases), \
        "CRITIC_ADVERSARIAL_CLAUSE is not covered by the echo filter"
