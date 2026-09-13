"""
aegis.transcript — persist every run to disk.

WHY THIS IS IN v1 AND NOT "LATER"
---------------------------------
Because the sprint list in your strategy report ends at "build an
evaluator, measure Pass@1, cost, tokens" - and you cannot evaluate runs
you did not record.

The moment you change a prompt, you need to answer: did that make it
better or worse? Without stored transcripts the honest answer is "I have
no idea, but it felt better", which is how prompt engineering turns into
superstition. Recording is cheap now and impossible to backfill later.

Format: one JSON file per run, plus an optional human-readable Markdown
version. JSON for machines and diffing, Markdown for reading.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agents import prompt_fingerprint
from .state import DebateState


def new_run_id() -> str:
    """Sortable-by-time id, so `ls runs/` is chronological."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def save_run(
    state: DebateState,
    settings_snapshot: dict[str, Any] | None = None,
    directory: str | os.PathLike[str] = "runs",
    also_markdown: bool = True,
) -> Path:
    """
    Write the run to <directory>/<run_id>.json (and .md).

    The settings snapshot is stored ALONGSIDE the transcript on purpose.
    A transcript without the model names, temperatures, and round cap
    that produced it is not reproducible, and a non-reproducible run is
    an anecdote rather than data.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = state.run_id or new_run_id()
    state.run_id = run_id

    payload = {
        "run_id": run_id,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": settings_snapshot or {},
        # Which prompt set produced this. See agents.prompt_fingerprint.
        "prompt_fingerprint": prompt_fingerprint(),
        "state": state.to_dict(),
    }

    json_path = out_dir / f"{run_id}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if also_markdown:
        (out_dir / f"{run_id}.md").write_text(to_markdown(state), encoding="utf-8")

    return json_path


def load_run(path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_markdown(state: DebateState) -> str:
    """Human-readable rendering. Also what the CLI's --save writes."""
    lines: list[str] = [
        f"# Aegis debate — {state.topic}",
        "",
        f"- Run id: `{state.run_id or 'unsaved'}`",
        f"- Started: {state.started_at}",
        f"- Rounds used: {state.round} / {state.max_rounds}",
        f"- Stop reason: **{state.stop_reason or 'unknown'}**",
        f"- Tokens: {state.tokens_in} in / {state.tokens_out} out",
        f"- Inference time: {state.elapsed_s:.1f}s",
        f"- Estimated cost: ${state.cost_usd:.4f}",
        f"- Guards: {state.max_rounds} rounds · "
        f"${state.max_cost_usd:.2f} · "
        f"{state.max_seconds:.0f}s" if state.max_seconds else
        f"- Guards: {state.max_rounds} rounds · ${state.max_cost_usd:.2f}",
        "",
    ]

    if state.error:
        lines += [f"> Error: `{state.error}`", ""]

    lines += ["---", "", "## Final answer", "", state.answer or "_(no answer produced)_", ""]

    # The routing log. A transcript that cannot say WHY it ended is an
    # anecdote: you can see that the debate stopped after three rounds, but
    # not whether that was agreement, a deadlock, or a guard firing - and
    # those three mean completely different things about the answer above.
    if state.decisions:
        lines += ["---", "", "## Why it routed the way it did", ""]
        for d in state.decisions:
            dest = "END" if d.next_node == "__end__" else d.next_node
            lines += [f"- **{d.rule.upper()}** (after {d.at}, round {d.round}) "
                      f"→ `{dest}` — {d.reason}"]
        lines += [""]

    lines += ["---", "", "## Full transcript", ""]

    for turn in state.transcript:
        label = turn.agent.upper()
        suffix = f" — verdict: {turn.verdict}" if turn.verdict else ""
        lines += [
            f"### Round {turn.round} · {label}{suffix}",
            f"_{turn.model} · {turn.latency_s}s "
            f"(ttft {turn.ttft_s}s, {turn.tokens_per_s} tok/s) · "
            f"{turn.tokens_in} in / {turn.tokens_out} out_",
            "",
            turn.content,
            "",
        ]

    return "\n".join(lines)
