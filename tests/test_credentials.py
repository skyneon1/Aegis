"""
Tests for the runtime key store (aegis/credentials.py).

EVERY TEST HERE WRITES TO tmp_path. None of them may touch the real store at
~/.aegis/keys.json — a test suite that can overwrite the user's credentials is
a worse bug than anything it could be testing for. The store path is a
parameter on every function precisely so this file can redirect it.

WHAT IS BEING DEFENDED
----------------------
This module holds secrets on disk, so its failure modes are not "wrong answer"
but "key disclosed" and "key silently lost". The permission test and the
atomic-write test are the two that matter; the rest is round-tripping.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aegis import credentials  # noqa: E402


def test_round_trip(tmp_path):
    store = tmp_path / "keys.json"
    assert credentials.get_key("groq", store) == ""

    assert credentials.set_key("groq", "gsk_secret_value", store) is True
    assert credentials.get_key("groq", store) == "gsk_secret_value"
    assert credentials.saved_providers(store) == ["groq"]


def test_several_providers_coexist(tmp_path):
    store = tmp_path / "keys.json"
    credentials.set_key("groq", "a-key", store)
    credentials.set_key("moonshot", "b-key", store)

    assert credentials.all_keys(store) == {"groq": "a-key", "moonshot": "b-key"}
    assert credentials.saved_providers(store) == ["groq", "moonshot"]


def test_an_explicit_path_never_falls_back_to_the_real_store(tmp_path):
    """
    The isolation this whole file depends on. If an explicit path were ever
    ignored, these tests would start reading — and overwriting — the user's
    actual credentials, and would still pass.
    """
    store = tmp_path / "keys.json"
    credentials.set_key("groq", "a-key", store)

    assert credentials.store_path(str(store)) == store
    assert credentials.store_path(str(store)) != credentials.DEFAULT_PATH
    assert store.read_text().count("a-key") == 1


def test_the_file_is_not_readable_by_anyone_else(tmp_path):
    """
    0600. A key file the rest of the machine can read is not storage, it is
    disclosure with extra steps — and on a shared box it is the whole point
    of the module defeated silently.
    """
    store = tmp_path / "keys.json"
    credentials.set_key("groq", "gsk_secret_value", store)

    mode = stat.S_IMODE(store.stat().st_mode)
    assert mode == 0o600, f"key file is mode {mode:o}, expected 600"


def test_an_empty_key_deletes_rather_than_storing_a_blank(tmp_path):
    """
    "Saved but blank" is a state that should not be reachable: it reads as a
    configured provider that then fails authentication, which is a more
    confusing error than "no key set".
    """
    store = tmp_path / "keys.json"
    credentials.set_key("groq", "gsk_secret_value", store)
    credentials.set_key("groq", "   ", store)

    assert credentials.get_key("groq", store) == ""
    assert credentials.saved_providers(store) == []


def test_forget_removes_only_the_named_provider(tmp_path):
    store = tmp_path / "keys.json"
    credentials.set_key("groq", "a-key", store)
    credentials.set_key("moonshot", "b-key", store)

    credentials.forget_key("groq", store)

    assert credentials.get_key("groq", store) == ""
    assert credentials.get_key("moonshot", store) == "b-key"


def test_keys_are_stripped_of_stray_whitespace(tmp_path):
    """
    A key pasted from a web console routinely arrives with a trailing
    newline, and the resulting 401 says nothing about whitespace.
    """
    store = tmp_path / "keys.json"
    credentials.set_key("groq", "  gsk_secret_value\n", store)
    assert credentials.get_key("groq", store) == "gsk_secret_value"


# ---------------------------------------------------------------------------
# Never raises — a UI is the caller
# ---------------------------------------------------------------------------


def test_a_missing_store_reads_as_no_keys(tmp_path):
    assert credentials.all_keys(tmp_path / "nope.json") == {}
    assert credentials.saved_providers(tmp_path / "nope.json") == []


def test_a_corrupt_store_reads_as_no_keys_instead_of_crashing(tmp_path):
    """
    Truncated JSON is indistinguishable from "no keys" at read time, and the
    right response to both is to carry on. An app that will not start
    because a cache file is malformed has turned a convenience into a
    dependency.
    """
    store = tmp_path / "keys.json"
    store.write_text('{"keys": {"groq": "abc"')  # truncated mid-write
    assert credentials.all_keys(store) == {}


def test_a_store_of_the_wrong_shape_reads_as_no_keys(tmp_path):
    store = tmp_path / "keys.json"
    store.write_text('["not", "a", "mapping"]')
    assert credentials.all_keys(store) == {}

    store.write_text('{"keys": "not a mapping either"}')
    assert credentials.all_keys(store) == {}


def test_writing_creates_the_parent_directory(tmp_path):
    store = tmp_path / "nested" / "deeper" / "keys.json"
    assert credentials.set_key("groq", "a-key", store) is True
    assert store.exists()


def test_the_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    """
    Write-then-rename. A half-written key file would take every saved
    credential with it, and the wreckage reads as "no keys" — the loss is
    silent.
    """
    store = tmp_path / "keys.json"
    credentials.set_key("groq", "a-key", store)
    credentials.set_key("moonshot", "b-key", store)

    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "keys.json"]
    assert leftovers == [], f"temp files left behind: {leftovers}"
    assert json.loads(store.read_text())["keys"]["groq"] == "a-key"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def test_fingerprint_shows_enough_to_recognise_and_not_enough_to_use():
    fp = credentials.fingerprint("gsk_abcdefghijklmnopqrstuvwxyz")
    assert fp == "gsk_a…wxyz"
    assert "defghijklmnop" not in fp


def test_a_short_key_is_masked_completely():
    """
    The redaction has to hold for the WORST case, not the typical one: a
    12-character key rendered as first-5 + last-4 would disclose three
    quarters of itself.
    """
    assert credentials.fingerprint("shortkey") == "•" * len("shortkey")
    assert credentials.fingerprint("") == ""
