"""
Tests for model discovery (aegis/catalog.py).

HERMETIC. Nothing here reaches the internet: the parsing tests feed the
function literal payloads, and the failure tests point at a closed port on
127.0.0.1, which refuses instantly. A discovery module whose tests need the
network could only be run by someone who already has working connectivity to
fourteen providers — which is precisely the person who does not need it.

WHAT IS BEING DEFENDED
----------------------
catalog.py exists because hard-coded model ids go stale and produce a 404 in
round two. Its own failure mode is subtler: a provider that answers in a
shape nobody anticipated, silently yielding an empty picker. So the parsing
tests cover the shapes providers actually return, and every error path is
asserted to degrade rather than raise.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aegis import catalog  # noqa: E402


# ---------------------------------------------------------------------------
# Parsing — fourteen endpoints, one shape out
# ---------------------------------------------------------------------------


def test_the_standard_openai_shape():
    payload = {"data": [{"id": "gpt-4.1-mini"}, {"id": "o4-mini"}]}
    assert catalog._parse_models(payload) == ["gpt-4.1-mini", "o4-mini"]


def test_a_bare_list_is_accepted():
    """Not every implementation wraps its listing in {"data": ...}."""
    assert catalog._parse_models([{"id": "a"}, {"id": "b"}]) == ["a", "b"]
    assert catalog._parse_models(["a", "b"]) == ["a", "b"]


def test_a_models_key_is_accepted_as_well_as_data():
    assert catalog._parse_models({"models": [{"id": "a"}]}) == ["a"]


def test_a_name_field_is_accepted_when_there_is_no_id():
    assert catalog._parse_models({"data": [{"name": "a"}]}) == ["a"]


def test_googles_models_prefix_is_stripped():
    """
    Gemini's OpenAI-compatible listing returns "models/gemini-2.5-flash"
    while its chat endpoint wants "gemini-2.5-flash". Handing the picker an
    id the completion call will reject is worse than not listing it at all —
    the user selects a model that cannot work and the error names a string
    they never typed.
    """
    payload = {"data": [{"id": "models/gemini-2.5-flash"}]}
    assert catalog._parse_models(payload) == ["gemini-2.5-flash"]


def test_results_are_deduplicated_and_sorted():
    payload = {"data": [{"id": "b"}, {"id": "a"}, {"id": "b"}]}
    assert catalog._parse_models(payload) == ["a", "b"]


def test_junk_entries_are_skipped_rather_than_crashing_the_listing():
    """
    One malformed row must not cost the other three hundred.
    """
    payload = {"data": [{"id": "good"}, None, 42, {}, {"id": ""}]}
    assert catalog._parse_models(payload) == ["good"]


def test_an_unrecognised_payload_yields_an_empty_list():
    assert catalog._parse_models({"unexpected": "shape"}) == []
    assert catalog._parse_models("a string") == []
    assert catalog._parse_models(None) == []


# ---------------------------------------------------------------------------
# Probing — must degrade, never raise
# ---------------------------------------------------------------------------


def test_an_unreachable_endpoint_reports_failure_without_raising():
    # Port 1 on loopback: refused immediately, no packets leave the machine.
    probe = catalog.probe("http://127.0.0.1:1/v1", "any-key")
    assert probe.ok is False
    assert probe.models == []
    assert probe.detail, "a failed probe must say why"


def test_a_missing_base_url_is_reported_rather_than_requested():
    probe = catalog.probe("", "any-key")
    assert probe.ok is False
    assert "base URL" in probe.detail


def test_remote_models_returns_an_empty_list_on_any_problem():
    assert catalog.remote_models("http://127.0.0.1:1/v1") == []


def test_unauthorised_is_distinguished_from_unreachable():
    """
    401 and "connection refused" call for opposite fixes — retype the key
    versus check the URL — and a single "could not connect" hides which.
    """
    assert catalog.Probe(status=401).unauthorised is True
    assert catalog.Probe(status=403).unauthorised is True
    assert catalog.Probe(status=404).unauthorised is False
    assert catalog.Probe(status=0).unauthorised is False


def test_ok_and_models_are_independent():
    """
    A key can be valid against a provider that does not implement /models at
    all. That is a WORKING setup with an empty list, and collapsing the two
    fields into one would report it as broken.
    """
    probe = catalog.Probe(ok=True, models=[])
    assert probe.ok is True and probe.models == []


# ---------------------------------------------------------------------------
# Ranking — a 300-entry picker is not a choice
# ---------------------------------------------------------------------------


def test_free_models_sort_ahead_of_paid_ones():
    ranked = catalog.rank_models(["z-paid", "a-paid", "m:free"])
    assert ranked[0] == "m:free"


def test_the_current_selection_is_pinned_first_so_it_cannot_vanish():
    """
    OpenRouter alone serves over 300 models. If the selected one sorted into
    the middle of that list, every rerun would look like the picker had lost
    it.
    """
    ranked = catalog.rank_models(["a:free", "b-paid", "zzz-paid"], prefer="zzz-paid")
    assert ranked[0] == "zzz-paid"


def test_ranking_deduplicates_and_is_stable():
    ranked = catalog.rank_models(["b", "a", "b"])
    assert ranked == ["a", "b"]
