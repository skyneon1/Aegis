#!/usr/bin/env python3
"""
Aegis calibration — is the Critic a working gate?

    ./dev.sh calibrate                       # the configured critic model
    ./dev.sh calibrate --model qwen3.5:4b    # a specific one
    ./dev.sh calibrate --compare gemma2:2b qwen3.5:4b
    ./dev.sh calibrate --tiers good          # just the fast half

WHY THIS IS ITS OWN FRONTEND
----------------------------
`evaluate.py` answers "how did the debates go". This answers "can the judge
judge" - and the second question is not visible in the first. A critic that
approves nothing produces a clean 100%-arbitrated distribution, which reads
as a working system right up until you notice the number never changes.

Same rule as every other frontend here: zero logic of its own. Scoring,
diagnosis, and the cases all live in aegis/calibration.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aegis import hostinfo, load_settings
from aegis.calibration import run_calibration

_TTY = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def _prepare(model: str, strictness: str = ""):
    """Resolve settings for one critic model, detecting thinking models."""
    settings = load_settings()
    if model:
        settings.critic_model = model
    if strictness:
        # The whole point of measuring a strictness mode is that raising the
        # bar is not free: it should catch more flawed answers AND risk
        # failing good ones. Both halves show up in balanced accuracy.
        settings.critic_strictness = strictness
    if settings.local:
        # Ask the server which models think, so a reasoning model gets the
        # bigger token budget it needs. Without this a thinking model spends
        # its whole budget reasoning and returns an empty verdict - which
        # would score as a failed calibration when it is really a
        # misconfiguration.
        settings.reasoning_models = hostinfo.reasoning_models(
            hostinfo.available_models(settings.base_url), settings.base_url)
    return settings


def _run(model: str, tiers: tuple[str, ...], strictness: str = ""):
    settings = _prepare(model, strictness)
    label = f"{settings.critic_model}  [{settings.critic_strictness}]"
    thinking = label in settings.reasoning_models
    print(_c(f"\n  CALIBRATING  {label}"
             f"{'  [thinking model - expect slow turns]' if thinking else ''}",
             "1;36"))
    print(_c("  " + "─" * 68, "2"))

    def _tick(case, result):
        mark = _c("ok  ", "32") if result.correct else _c("MISS", "31")
        print(f"  {mark} {case.tier:<7} {case.case_id:<26} "
              f"want {result.expected:<8} got {result.actual:<8} "
              + _c(f"{result.latency_s:.0f}s", "2"))
        if not result.correct and result.reasons:
            print(_c(f"        → {result.reasons[0][:110]}", "2"))

    return run_calibration(settings=settings, tiers=tiers, on_result=_tick)


def main() -> int:
    ap = argparse.ArgumentParser(prog="calibrate")
    ap.add_argument("--model", default="", help="critic model to test")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="score two critic models on the same cases")
    ap.add_argument("--strictness", default="",
                    choices=["", "calibrated", "adversarial"],
                    help="critic strictness mode to test (default: configured)")
    ap.add_argument("--tiers", nargs="+", default=["good", "flawed", "gross"],
                    choices=["good", "flawed", "gross"])
    ap.add_argument("--save", metavar="PATH", default="",
                    help="write the report(s) as JSON")
    args = ap.parse_args()

    tiers = tuple(args.tiers)
    reports = []

    if args.compare:
        reports.extend(_run(model, tiers, args.strictness)
                       for model in args.compare)
    else:
        reports.append(_run(args.model, tiers, args.strictness))

    print()
    for r in reports:
        print(_c("  " + "─" * 68, "2"))
        print("  " + r.summary().replace("\n", "\n  "))

    if len(reports) == 2:
        a, b = reports
        print(_c("\n  " + "─" * 68, "2"))
        print(_c(f"  {a.model} vs {b.model}", "1"))
        for cls in ("APPROVE", "REVISE"):
            oa, na = a.recall(cls)
            ob, nb = b.recall(cls)
            print(f"    recall {cls:<8} {oa}/{na}   ->   {ob}/{nb}")
        print(f"    balanced        {a.balanced_accuracy:.0%}   ->   "
              f"{b.balanced_accuracy:.0%}")
        # Same prompt, different model: if the scores differ materially, the
        # limitation is the model, not the wording. That is the one question
        # no amount of prompt editing can answer on its own.
        if a.prompt_fingerprint == b.prompt_fingerprint:
            print(_c(f"    identical prompt ({a.prompt_fingerprint}) - any "
                     f"difference above is the MODEL, not the wording.", "2"))

    if args.save:
        Path(args.save).write_text(
            json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8")
        print(_c(f"\n  saved: {args.save}", "2"))

    # Exit non-zero if any critic abandoned a whole verdict class - that is a
    # broken gate, not a low score.
    return 0 if all(r.balanced_accuracy > 0.5 for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
