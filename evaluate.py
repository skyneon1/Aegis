#!/usr/bin/env python3
"""
Aegis eval harness — Phase 0.3.

    python3 evaluate.py --provider fake                          # 8 built-in topics, offline
    python3 evaluate.py --topics topics.txt --provider openrouter --save
    python3 evaluate.py --topics topics.txt --rounds 2 --workers 4 --save
    python3 evaluate.py --compare runs/evals/<a>/report.json runs/evals/<b>/report.json

WHY A SEPARATE FRONTEND INSTEAD OF A cli.py SUBCOMMAND
--------------------------------------------------------
cli.py's contract is "run exactly one debate, print it, exit with a
meaningful code" — scripts depend on that exit-code meaning (see cli.py's
own docstring: 0/3/2 for approved/arbitrated/unreviewed). Bolting "run N
debates and print a table" onto the same entry point behind a flag would
make the exit code mean two different things depending on how you called
it — exactly the ambiguity this project avoids everywhere else. A second,
small frontend is cheaper than an overloaded one.

Same rule as always: zero orchestration logic lives in this file. Every
number below was already counted by aegis.evaluation; this file only
formats and prints.
"""

from __future__ import annotations

import argparse
import sys

from aegis import (
    DEFAULT_TOPICS,
    compare_reports,
    load_cases,
    load_report,
    load_settings,
    run_eval,
    save_report,
)

_TTY = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


BOLD, DIM, BLUE, YELLOW, GREEN, RED, MAGENTA = "1", "2", "34", "33", "32", "31", "35"

_OUTCOME_COLOUR = {"approved": GREEN, "arbitrated": MAGENTA, "max_rounds": YELLOW, "error": RED}


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="aegis-eval",
        description="Run N topics through the debate graph and score the outcomes.",
    )
    parser.add_argument(
        "-t", "--topics", default=None,
        help="Path to a .txt (one topic/line) or .jsonl case file. "
             f"Default: {len(DEFAULT_TOPICS)} built-in topics.",
    )
    parser.add_argument("-p", "--provider", default=None,
                        help="openrouter | deepseek | ollama | fake")
    parser.add_argument("-r", "--rounds", type=int, default=None,
                        help="Max debate rounds per case (default 3).")
    parser.add_argument("--budget", type=float, default=None,
                        help="Per-case budget guard in USD. 0 disables.")
    parser.add_argument("--no-arbiter", action="store_true",
                        help="On deadlock, count the last unreviewed revision "
                             "instead of escalating to the Arbiter.")
    # Keep the default at 1. On a LOCAL provider, concurrency is not a
    # speedup: every worker queues behind the same model on the same GPU,
    # and several long contexts competing for a 4GB KV cache can push the
    # model into partial CPU offload - making the whole batch slower than
    # running it serially. Raise it only for a remote endpoint.
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="Run cases concurrently. Safe: each case is fully "
                             "independent. Mind provider rate limits.")
    parser.add_argument("--save", action="store_true",
                        help="Write report.json + report.md to runs/evals/<id>/")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print the final summary table.")
    parser.add_argument(
        "--compare", nargs=2, metavar=("REPORT_A", "REPORT_B"),
        help="Skip running anything; diff two saved report.json files "
             "(e.g. two prompt versions run against the same topics).",
    )
    args = parser.parse_args()

    if args.compare:
        a = load_report(args.compare[0])
        b = load_report(args.compare[1])
        print(compare_reports(a, b, label_a="A", label_b="B"))
        return 0

    settings = load_settings(provider=args.provider)
    if args.rounds is not None:
        settings.max_rounds = args.rounds
    if args.budget is not None:
        settings.max_cost_usd = args.budget
    if args.no_arbiter:
        settings.use_arbiter = False

    cases = load_cases(args.topics) if args.topics else None
    n_cases = len(cases) if cases else len(DEFAULT_TOPICS)

    if not args.quiet:
        print(_c("\n  AEGIS EVAL  ", f"{BOLD};{BLUE}") + _c(
            f"provider={settings.provider}  cases={n_cases}  "
            f"cap={settings.max_rounds} rounds  workers={args.workers}", DIM))
        if settings.is_fake:
            print(_c("  Running with FakeLLM - no network, no cost. "
                     "Set OPENROUTER_API_KEY for real models.", YELLOW))
        print(_c("─" * 72, DIM))

    def _on_result(case, state) -> None:
        if args.quiet:
            return
        outcome = state.stop_reason or "error"
        colour = _OUTCOME_COLOUR.get(state.stop_reason, RED)
        print(
            f"  {_c(f'{outcome:<11}', colour)} "
            f"round={state.round}  cost=${state.cost_usd:.4f}   {case.case_id}"
        )

    report = run_eval(cases, settings=settings, workers=args.workers, on_result=_on_result)

    print(_c("\n" + "─" * 72, DIM))
    print(report.summary())

    if args.save:
        path = save_report(report, directory="runs/evals")
        print(_c(f"\n  saved: {path}", DIM))

    # Distinct, single-purpose exit code: did every case at least run
    # without raising. Outcome quality (approved vs arbitrated vs capped)
    # is a distribution, not a pass/fail — read the summary table for
    # that, the same way cli.py's own exit code only covers ONE run.
    return 1 if report.error_count else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130) from None
