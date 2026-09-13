#!/usr/bin/env python3
"""
Aegis CLI — run a debate from the terminal.

    python cli.py "Should a two-person startup use Kubernetes?"
    python cli.py "..." --rounds 4 --provider openrouter --save
    python cli.py "..." --provider fake          # offline, free, instant

WHY A CLI WHEN YOU ASKED FOR A WEB UI
-------------------------------------
Because the CLI is how you will actually debug this. When a run misbehaves,
you want a plain stdout trace you can pipe, grep, and diff - not a browser
tab with rerun semantics in the way. The CLI also becomes the backbone of
the batch evaluator in a later sprint.

Both frontends import the same two functions from `aegis`. Neither contains
a single line of orchestration logic. If you ever find yourself writing an
`if verdict == ...` inside a UI file, something has gone wrong.
"""

from __future__ import annotations

import argparse
import sys

from aegis import load_settings, save_run, stream_debate

# ANSI colours, disabled automatically when piping to a file.
_TTY = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


BOLD, DIM, BLUE, YELLOW, GREEN, RED, MAGENTA = "1", "2", "34", "33", "32", "31", "35"

# Verdict -> colour, as a table rather than a conditional. Same reason as in
# app.py: `if verdict == "APPROVE"` is precisely the shape that a frontend is
# not allowed to contain, and "it is only choosing a colour" is how that rule
# erodes. A dict also degrades gracefully if a new verdict appears.
VERDICT_COLOUR = {"APPROVE": GREEN, "REVISE": YELLOW, "RULED": MAGENTA}

# Agent -> (header, colour). A table, not a chain of elifs: the frontend
# decides presentation only, and a dict cannot mishandle an agent added later.
AGENT_HEADER = {
    "researcher": ("[RESEARCH] SOURCES", "36"),
    "proposer": ("PROPOSER", BLUE),
    "critic": ("CRITIC", YELLOW),
    "arbiter": ("ARBITER", MAGENTA),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="aegis",
        description="Two agents debate a topic until the critic approves.",
    )
    parser.add_argument("topic", help="The question or topic to debate.")
    parser.add_argument("-r", "--rounds", type=int, default=None,
                        help="Max debate rounds (default 3). The safety cap.")
    parser.add_argument("-p", "--provider", default=None,
                        help="openrouter | deepseek | ollama | fake")
    parser.add_argument("--budget", type=float, default=None,
                        help="Max estimated USD for this run. 0 disables.")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Wall-clock limit in seconds. This is the guard "
                             "that matters for LOCAL models, where tokens are "
                             "free but your GPU is not. 0 disables.")
    parser.add_argument("--no-stream", action="store_true",
                        help="Wait for each complete turn instead of printing "
                             "tokens as they arrive.")
    parser.add_argument("--no-arbiter", action="store_true",
                        help="On deadlock, return the last unreviewed revision "
                             "instead of escalating to the Arbiter.")
    parser.add_argument("--save", action="store_true",
                        help="Write JSON + Markdown transcript to runs/")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print the final answer.")
    args = parser.parse_args()

    settings = load_settings(provider=args.provider)
    if args.rounds is not None:
        settings.max_rounds = args.rounds
    if args.budget is not None:
        settings.max_cost_usd = args.budget
    if args.seconds is not None:
        settings.max_seconds = args.seconds
    if args.no_stream:
        settings.stream = False
    if args.no_arbiter:
        settings.use_arbiter = False

    if not args.quiet:
        print(_c("\n  AEGIS  ", f"{BOLD};{BLUE}") + _c(
            f"provider={settings.provider}  "
            f"proposer={settings.proposer_model}  "
            f"critic={settings.critic_model}  "
            f"cap={settings.max_rounds} rounds"
            + (f"/{settings.max_seconds:.0f}s" if settings.max_seconds else "")
            + (f"  research={settings.research_results} sources"
               if settings.use_researcher else "  research=off"), DIM))
        if settings.is_fake:
            print(_c("  Running with FakeLLM - no network, no cost. "
                     "Set OPENROUTER_API_KEY for real models.", YELLOW))
        print(_c("─" * 72, DIM))

    class _Live:
        """
        Prints tokens as they arrive, with a header when each turn opens.

        Live output belongs in the CLI as much as in the browser. A debate
        against a local model is a minute of wall time, and watching it
        arrive is how you notice a model looping or a prompt misfiring at
        the second token rather than at the end.
        """

        def __init__(self) -> None:
            self.open = False
            self.round_hint = 1

        def __call__(self, agent: str, token: str) -> None:
            if not self.open:
                self.open = True
                if agent == "researcher":
                    label = _c("\n[RESEARCH] SOURCES", f"{BOLD};36")
                elif agent == "arbiter":
                    label = _c("\n[DEADLOCK] ARBITER → FINAL RULING",
                               f"{BOLD};{MAGENTA}")
                elif agent == "critic":
                    label = _c(f"\n[ROUND {self.round_hint}] CRITIC",
                               f"{BOLD};{YELLOW}")
                else:
                    label = _c(f"\n[ROUND {self.round_hint}] PROPOSER",
                               f"{BOLD};{BLUE}")
                print(label)
            sys.stdout.write(token)
            sys.stdout.flush()

        def close(self) -> None:
            self.open = False

    live = _Live()
    streaming = settings.stream and not args.quiet

    final_state = None
    for node, state in stream_debate(
            args.topic, settings=settings,
            on_token=live if streaming else None):
        final_state = state
        if args.quiet or node.startswith("__"):
            continue

        turn = state.transcript[-1] if state.transcript else None
        if turn is None:
            continue

        # ttft and tok/s are reported separately on purpose: a slow ttft
        # means the model was loading or the prompt was long, a slow tok/s
        # means generation itself is slow (usually a model that did not fit
        # in VRAM). One combined latency number hides which it was, and the
        # two call for opposite fixes.
        telemetry = _c(
            f"  ({turn.model} · {turn.latency_s}s · ttft {turn.ttft_s}s · "
            f"{turn.tokens_per_s:.0f} tok/s · "
            f"{turn.tokens_in}in/{turn.tokens_out}out)", DIM)

        if streaming:
            live.close()
            verdict = ""
            if turn.verdict and turn.agent == "critic":
                colour = VERDICT_COLOUR.get(turn.verdict, YELLOW)
                verdict = _c(f"  → {turn.verdict}", f"{BOLD};{colour}")
            print(verdict + telemetry)
        else:
            if turn.agent == "proposer":
                header = _c(f"\n[ROUND {turn.round}] PROPOSER", f"{BOLD};{BLUE}")
            elif turn.agent == "arbiter":
                header = _c("\n[DEADLOCK] ARBITER → FINAL RULING",
                            f"{BOLD};{MAGENTA}")
            else:
                colour = VERDICT_COLOUR.get(turn.verdict, YELLOW)
                header = _c(f"\n[ROUND {turn.round}] CRITIC → {turn.verdict}",
                            f"{BOLD};{colour}")
            print(header + telemetry)
            print(turn.content)

        live.round_hint = state.round + 1

    if final_state is None:
        print(_c("No output produced.", RED), file=sys.stderr)
        return 1

    if args.quiet:
        print(final_state.answer)
    else:
        print(_c("\n" + "─" * 72, DIM))
        colours = {"approved": GREEN, "arbitrated": MAGENTA, "error": RED}
        verdict_colour = colours.get(final_state.stop_reason, YELLOW)
        print(_c(f"  stop: {final_state.stop_reason}", f"{BOLD};{verdict_colour}")
              + _c(f"   rounds: {final_state.round}/{final_state.max_rounds}"
                   f"   tokens: {final_state.tokens_in}in/{final_state.tokens_out}out"
                   f"   inference: {final_state.elapsed_s:.1f}s"
                   f"   est. cost: ${final_state.cost_usd:.4f}", DIM))
        # Why it ended, straight from the router's own log rather than
        # re-derived here. The CLI has no business re-implementing the
        # routing conditions to explain them.
        for decision in final_state.decisions[-1:]:
            print(_c(f"  why: {decision.reason}", DIM))
        if final_state.stop_reason == "arbitrated":
            print(_c("  Note: Proposer and Critic did not agree. The answer above "
                     "is the Arbiter's ruling on a contested question, not a "
                     "consensus.", MAGENTA))
        elif final_state.stop_reason == "max_rounds":
            print(_c("  Note: the critic never approved. The answer above is the "
                     "last revision, not a reviewed result.", YELLOW))
        if final_state.error:
            print(_c(f"  error: {final_state.error}", RED))

    if args.save:
        path = save_run(final_state, settings.redacted(), settings.transcript_dir)
        if not args.quiet:
            print(_c(f"  saved: {path}", DIM))

    # Distinct exit codes so the CLI composes in scripts and the future
    # eval harness. Three outcomes, three codes - collapsing "agreed" and
    # "adjudicated" into one would throw away the signal that a topic is
    # contested, which is exactly what you would want to measure.
    #   0 = critic approved
    #   3 = deadlocked, arbiter ruled
    #   2 = no approval and no ruling (weakest outcome), or error
    return {"approved": 0, "arbitrated": 3}.get(final_state.stop_reason, 2)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except RuntimeError as exc:
        # A CONFIGURATION problem, not a crash. The one that reaches here in
        # practice is a missing API key, and its message already says exactly
        # what to do; wrapping it in a traceback buries that advice under
        # eleven lines of stack that point at code the user cannot fix.
        # Genuine bugs still raise their own types and keep their traceback.
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(2) from None
