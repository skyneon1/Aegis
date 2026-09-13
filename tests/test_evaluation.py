"""
Offline tests for the Aegis eval harness (aegis/evaluation.py).

Every test here runs with provider='fake': no network, no API key, no
cost. Same discipline as test_graph.py — these tests prove the HARNESS
counts correctly (one result per case, aggregates match, a raising case
doesn't kill the batch, order survives concurrency, save/load round-
trips), not that any particular topic gets a good answer. Answer quality
is a different question and needs a different tool.

Run:  pytest tests/ -v
      python tests/test_evaluation.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aegis import (
    EvalCase,
    EvalReport,
    EvalResult,
    compare_reports,
    load_cases,
    load_report,
    run_eval,
    save_report,
)
from aegis.evaluation import _slug

# ---------------------------------------------------------------------------
# EvalCase
# ---------------------------------------------------------------------------


def test_eval_case_defaults_its_id_to_a_slug_of_the_topic():
    case = EvalCase(topic="Is REST better than GraphQL?")
    assert case.case_id == "is-rest-better-than-graphql"


def test_eval_case_respects_an_explicit_id():
    case = EvalCase(topic="Anything", case_id="my-id")
    assert case.case_id == "my-id"


def test_slug_never_returns_empty():
    assert _slug("???") == "case"
    assert _slug("") == "case"


# ---------------------------------------------------------------------------
# Loading cases from disk
# ---------------------------------------------------------------------------


def test_load_cases_from_txt_skips_blanks_and_comments():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "topics.txt"
        path.write_text(
            "# a comment\n"
            "\n"
            "Should teams use trunk-based development?\n"
            "   \n"
            "# another comment\n"
            "Is code review theater or does it catch real bugs?\n",
            encoding="utf-8",
        )
        cases = load_cases(path)

    assert len(cases) == 2
    assert cases[0].topic == "Should teams use trunk-based development?"
    assert cases[1].topic == "Is code review theater or does it catch real bugs?"


def test_load_cases_from_jsonl_preserves_explicit_ids():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "topics.jsonl"
        path.write_text(
            '{"topic": "Topic A", "id": "case-a"}\n'
            '{"topic": "Topic B"}\n',
            encoding="utf-8",
        )
        cases = load_cases(path)

    assert cases[0].case_id == "case-a"
    assert cases[1].case_id == _slug("Topic B")  # no id given -> derived


# ---------------------------------------------------------------------------
# run_eval — the core loop
# ---------------------------------------------------------------------------

_CASES = [
    EvalCase(topic="Should a two-person startup use Kubernetes?"),
    EvalCase(topic="Is premature optimization always bad?"),
    EvalCase(topic="Should code review require two approvals?"),
]


def test_run_eval_produces_one_result_per_case():
    report = run_eval(_CASES, provider="fake", max_rounds=3, use_arbiter=False)
    assert len(report.results) == len(_CASES)
    assert {r.case_id for r in report.results} == {c.case_id for c in _CASES}


def test_run_eval_aggregates_match_a_manual_sum():
    report = run_eval(_CASES, provider="fake", max_rounds=3, use_arbiter=False)

    assert sum(report.outcome_counts.values()) == len(report.results)
    assert abs(report.total_cost_usd - sum(r.cost_usd for r in report.results)) < 1e-9
    assert report.avg_rounds == round(
        sum(r.round for r in report.results) / len(report.results), 2
    )
    assert report.error_count == sum(1 for r in report.results if r.error)


def test_run_eval_with_no_cases_given_uses_default_topics():
    from aegis.evaluation import DEFAULT_TOPICS

    report = run_eval(provider="fake", max_rounds=2)
    assert len(report.results) == len(DEFAULT_TOPICS)


def test_run_eval_survives_a_case_that_raises():
    """A single bad case (here: an empty topic, which run_debate rejects)
    must not take the whole batch down — that would make an eval run an
    all-or-nothing gamble on N-1 perfectly well-behaved topics.
    """
    cases = [
        EvalCase(topic="", case_id="broken"),
        EvalCase(topic="Is premature optimization always bad?", case_id="fine"),
    ]
    report = run_eval(cases, provider="fake", max_rounds=2)

    assert len(report.results) == 2
    by_id = {r.case_id: r for r in report.results}
    assert by_id["broken"].stop_reason == "error"
    assert by_id["broken"].error
    assert by_id["fine"].stop_reason in ("approved", "max_rounds", "arbitrated")
    assert not by_id["fine"].error


def test_on_result_callback_fires_once_per_case():
    seen = []
    run_eval(_CASES, provider="fake", max_rounds=2,
              on_result=lambda case, state: seen.append(case.case_id))
    assert sorted(seen) == sorted(c.case_id for c in _CASES)


def test_run_eval_with_workers_preserves_input_order():
    report = run_eval(_CASES, provider="fake", max_rounds=2, workers=4)
    assert [r.case_id for r in report.results] == [c.case_id for c in _CASES]


def test_run_eval_rejects_an_empty_case_list():
    try:
        run_eval([], provider="fake")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for an empty case list")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_load_report_roundtrip():
    report = run_eval(_CASES, provider="fake", max_rounds=2)

    with tempfile.TemporaryDirectory() as d:
        path = save_report(report, directory=d)
        assert path.exists()
        assert (path.parent / "report.md").exists()

        reloaded = load_report(path)

    assert reloaded.eval_id == report.eval_id
    assert len(reloaded.results) == len(report.results)
    assert reloaded.outcome_counts == report.outcome_counts
    assert abs(reloaded.total_cost_usd - report.total_cost_usd) < 1e-9


# ---------------------------------------------------------------------------
# compare_reports
# ---------------------------------------------------------------------------


def _fake_report(eval_id: str, outcomes: list[str]) -> EvalReport:
    """Build a report by hand — no need to run the graph to test comparison
    math and formatting in isolation."""
    results = [
        EvalResult(
            case_id=f"case-{i}", topic=f"topic {i}", stop_reason=outcome,
            round=1, tokens_in=10, tokens_out=10, cost_usd=0.01,
            latency_s=0.0, error="", run_id="",
        )
        for i, outcome in enumerate(outcomes)
    ]
    return EvalReport(eval_id=eval_id, settings_snapshot={}, results=results)


def test_compare_reports_shows_outcome_deltas():
    a = _fake_report("eval-a", ["approved", "approved", "max_rounds"])
    b = _fake_report("eval-b", ["approved", "approved", "approved"])

    text = compare_reports(a, b, label_a="before", label_b="after")

    assert "eval-a" in text and "eval-b" in text
    assert "approved" in text
    assert "max_rounds" in text


def test_compare_reports_flags_per_case_outcome_flips():
    a = _fake_report("eval-a", ["approved", "max_rounds"])
    b = _fake_report("eval-b", ["approved", "approved"])  # case-1 flipped

    text = compare_reports(a, b)

    assert "per-case outcome changes" in text
    assert "case-1: max_rounds -> approved" in text


def test_compare_reports_has_no_flip_section_when_nothing_changed():
    a = _fake_report("eval-a", ["approved", "approved"])
    b = _fake_report("eval-b", ["approved", "approved"])

    text = compare_reports(a, b)
    assert "per-case outcome changes" not in text


# ---------------------------------------------------------------------------
# EvalReport.summary()
# ---------------------------------------------------------------------------


def test_summary_reports_outcome_distribution_and_costs():
    report = _fake_report("eval-x", ["approved", "approved", "max_rounds"])
    text = report.summary()

    assert "eval-x" in text
    assert "approved" in text and "max_rounds" in text
    assert "avg cost" in text
    assert "3" in text  # case count appears somewhere


def test_summary_flags_errors_when_present():
    report = _fake_report("eval-y", ["approved"])
    report.results[0].error = "boom"
    assert "ERRORS" in report.summary()


# ---------------------------------------------------------------------------
# Dependency-free runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
