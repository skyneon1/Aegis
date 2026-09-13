"""
Tests for the provider table (aegis/providers.py).

WHAT THIS FILE IS DEFENDING
---------------------------
providers.py is pure data, and pure data fails differently from code: it does
not raise, it just produces a provider that cannot be reached, a picker entry
that cannot be selected, or a price that silently disarms the budget guard.
None of those show up as an exception — they show up as a 404 in round two,
or as a run that never stops.

So every test here asserts a STRUCTURAL invariant that the rest of the system
assumes but never checks at runtime. Adding a fifteenth provider should either
satisfy them or be told exactly which promise it broke.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aegis import providers as prov  # noqa: E402
from aegis.providers import PROVIDERS  # noqa: E402


# ---------------------------------------------------------------------------
# Structure — what every row must declare
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = (
    "label", "base_url", "key_env", "proposer", "critic",
    "requires_key", "local", "free_tier", "free_note", "console",
    "cost_in", "cost_out", "models",
)


def test_every_provider_declares_every_field():
    """
    A missing field is not a missing feature, it is a KeyError in the UI at
    render time — or worse, a silently inherited default that happens to be
    wrong. Both are cheaper to catch here.
    """
    for name, preset in PROVIDERS.items():
        for field in REQUIRED_FIELDS:
            assert field in preset, f"{name} does not declare {field!r}"


def test_labels_are_unique_and_non_empty():
    """
    The provider picker displays labels and resolves the user's choice by
    them. Two providers sharing a label makes one of them unselectable, and
    the failure looks like "the dropdown is broken", not like a data problem.
    """
    labels = [p["label"] for p in PROVIDERS.values()]
    assert all(labels), "every provider needs a display label"
    assert len(labels) == len(set(labels)), \
        f"duplicate provider labels: {sorted(labels)}"


def test_every_provider_needing_a_key_names_the_variable_that_holds_it():
    """
    The 'no API key' error tells the user which variable to set, reading it
    from here. A blank key_env would print advice with a hole in it.
    """
    for name, preset in PROVIDERS.items():
        if preset["requires_key"]:
            assert preset["key_env"], f"{name} requires a key but names no env var"


def test_key_env_variables_are_unique():
    """
    Two providers reading one variable means setting a Groq key silently
    authenticates something else — and auto-detection would pick whichever
    came first in the table.
    """
    envs = [p["key_env"] for p in PROVIDERS.values() if p["key_env"]]
    assert len(envs) == len(set(envs)), f"duplicate key_env: {sorted(envs)}"


def test_remote_providers_have_a_base_url_and_local_ones_are_reachable():
    """
    An empty base_url is legitimate for exactly two rows: 'fake' makes no
    call at all, and 'custom' expects the user to supply one. For anyone
    else it is an endpoint pointing at nothing.
    """
    for name, preset in PROVIDERS.items():
        if name in ("fake", "custom"):
            continue
        assert preset["base_url"].startswith("http"), \
            f"{name} has no usable base_url"


def test_default_models_are_set_for_every_usable_provider():
    """
    Both roles need a model before a debate can start. 'custom' is the one
    exception — the user names the models along with the endpoint.
    """
    for name, preset in PROVIDERS.items():
        if name == "custom":
            continue
        assert preset["proposer"], f"{name} has no default proposer model"
        assert preset["critic"], f"{name} has no default critic model"


def test_a_free_tier_is_explained_and_a_key_can_be_obtained():
    """
    A provider list that names fourteen services and says nothing about how
    to get into any of them has relocated the dead end, not removed it.
    """
    for name, preset in PROVIDERS.items():
        if preset["free_tier"]:
            assert preset["free_note"], f"{name} claims a free tier but does not say what it is"
        if preset["requires_key"]:
            assert preset["console"].startswith("http"), \
                f"{name} needs a key but does not say where to get one"


# ---------------------------------------------------------------------------
# Pricing — the budget guard reads these
# ---------------------------------------------------------------------------


def test_prices_are_non_negative_and_local_inference_is_free():
    for name, preset in PROVIDERS.items():
        assert preset["cost_in"] >= 0, f"{name} has a negative input price"
        assert preset["cost_out"] >= 0, f"{name} has a negative output price"
        if preset["local"]:
            assert preset["cost_in"] == 0 and preset["cost_out"] == 0, \
                f"{name} generates tokens locally but reports a price"


def test_paid_providers_price_above_zero_so_the_budget_guard_stays_armed():
    """
    Zero is not a safe default for a remote provider. `max_cost_usd` bounds
    a stuck loop by multiplying tokens by a rate, and a rate of zero makes
    that product zero forever — the guard is present, checked, and incapable
    of firing. Free TIERS still carry a price here for exactly this reason;
    see the note in providers.py.
    """
    for name, preset in PROVIDERS.items():
        if preset["local"]:
            continue
        assert preset["cost_in"] > 0 and preset["cost_out"] > 0, \
            f"{name} prices tokens at zero, which disarms the budget guard"


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


def test_reasoning_models_are_recognised_from_their_names():
    """
    A remote endpoint cannot be asked whether a model thinks — /models
    returns ids and nothing else. Guessing wrong in the cautious direction
    costs a larger token budget; guessing wrong the other way returns an
    EMPTY answer, because the model spent the whole budget thinking.
    """
    for name in ("deepseek-reasoner", "deepseek-ai/deepseek-r1",
                 "deepseek-r1-distill-llama-70b", "qwen/qwq-32b",
                 "o4-mini", "magistral-small-latest"):
        assert prov.looks_like_reasoning_model(name), f"{name} should read as a thinker"

    for name in ("llama-3.3-70b-versatile", "gemma2:2b", "gpt-4.1-mini",
                 "glm-4.5-flash", "mistral-small-latest"):
        assert not prov.looks_like_reasoning_model(name), \
            f"{name} should not read as a thinker"


def test_free_model_ids_are_recognised_from_their_suffix():
    """
    Two providers encode price in the id itself, which is the only
    machine-readable statement of cost available without a second API call.
    """
    assert prov.is_free_model("deepseek/deepseek-chat-v3.1:free")
    assert prov.is_free_model("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    assert not prov.is_free_model("deepseek/deepseek-chat")
    assert not prov.is_free_model("")


def test_the_offline_and_local_providers_stay_selectable_without_a_key():
    """
    The invariant a fresh clone depends on: at least one provider runs with
    no credential at all.
    """
    keyless = [n for n, p in PROVIDERS.items() if not p["requires_key"]]
    assert "fake" in keyless
    assert "ollama" in keyless
