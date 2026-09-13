"""
Tests for how a provider becomes a Settings (aegis/config.py).

The three sources of truth — provider preset, key store, environment — have a
stated precedence, and every test here pins one edge of it. They exist because
the failures are quiet: a key read from the wrong place authenticates against
the wrong service, and a model name carried across a provider switch 404s in
round one naming a model the user never chose.

Every test redirects AEGIS_KEYSTORE at tmp_path. None may touch the real store.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from aegis import credentials, load_settings  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An isolated key store, and an environment with no provider keys in it."""
    path = tmp_path / "keys.json"
    monkeypatch.setenv("AEGIS_KEYSTORE", str(path))
    for var in ("GROQ_API_KEY", "MOONSHOT_API_KEY", "OPENROUTER_API_KEY",
                "AEGIS_PROPOSER_MODEL", "AEGIS_CRITIC_MODEL",
                "AEGIS_BASE_URL", "AEGIS_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    return path


# ---------------------------------------------------------------------------
# Where the key comes from
# ---------------------------------------------------------------------------


def test_a_saved_key_is_picked_up(store):
    credentials.set_key("groq", "gsk_from_store", store)
    settings = load_settings(provider="groq")
    assert settings.api_key == "gsk_from_store"
    assert settings.key_source == "store"


def test_the_environment_beats_the_key_store(store, monkeypatch):
    """
    A key in the environment is a deliberate statement about THIS session; a
    key in the store is a convenience remembered from an earlier one. The
    surprising outcome — "I exported a key and it used a different one" —
    must not be reachable.
    """
    credentials.set_key("groq", "gsk_from_store", store)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_from_env")

    settings = load_settings(provider="groq")
    assert settings.api_key == "gsk_from_env"
    assert settings.key_source == "env"


def test_no_key_anywhere_is_reported_as_such(store):
    settings = load_settings(provider="groq")
    assert settings.api_key == ""
    assert settings.key_source == ""
    assert settings.requires_key is True


def test_a_saved_key_for_one_provider_does_not_leak_to_another(store):
    credentials.set_key("groq", "gsk_from_store", store)
    assert load_settings(provider="moonshot").api_key == ""


# ---------------------------------------------------------------------------
# Model names must not cross provider boundaries
# ---------------------------------------------------------------------------


def test_env_model_overrides_apply_only_to_the_provider_they_were_written_for(
        store, monkeypatch):
    """
    The bug this prevents: a .env written for Ollama says
    AEGIS_PROPOSER_MODEL=gemma2:2b, and switching the picker to Groq sent
    Groq a model it has never heard of. The run died at the first turn with a
    404 naming a model the user had not chosen and could not see.
    """
    monkeypatch.setenv("AEGIS_PROVIDER", "ollama")
    monkeypatch.setenv("AEGIS_PROPOSER_MODEL", "gemma2:2b")

    # The provider the environment was describing: the override applies.
    assert load_settings(provider="ollama").proposer_model == "gemma2:2b"

    # Any other provider gets its OWN default back.
    groq = load_settings(provider="groq")
    assert groq.proposer_model != "gemma2:2b"
    assert groq.proposer_model == "llama-3.3-70b-versatile"


def test_base_url_is_scoped_the_same_way(store, monkeypatch):
    """AEGIS_BASE_URL pointing at localhost must not redirect Groq."""
    monkeypatch.setenv("AEGIS_PROVIDER", "ollama")
    monkeypatch.setenv("AEGIS_BASE_URL", "http://localhost:11434/v1")

    assert load_settings(provider="groq").base_url == "https://api.groq.com/openai/v1"


def test_an_override_with_no_provider_named_still_applies(store, monkeypatch):
    """
    With no AEGIS_PROVIDER to scope them to, the model variables are a
    statement about whatever provider is selected — the pre-existing
    behaviour, kept for anyone relying on it.
    """
    monkeypatch.setenv("AEGIS_PROPOSER_MODEL", "some-model")
    assert load_settings(provider="groq").proposer_model == "some-model"


# ---------------------------------------------------------------------------
# Per-provider pricing and guards
# ---------------------------------------------------------------------------


def test_each_provider_carries_its_own_rate(store):
    """
    One blended rate for every provider made the budget guard wrong by up to
    4x depending on who was serving.
    """
    assert load_settings(provider="deepseek").cost_in_per_1m == 0.27
    assert load_settings(provider="groq").cost_in_per_1m == 0.59
    assert load_settings(provider="ollama").cost_in_per_1m == 0.0


def test_a_free_model_id_makes_the_run_free_even_on_a_paid_provider(store):
    """
    `is_free` is a property of the SETTINGS, not the provider: OpenRouter
    bills, but its ':free' model ids do not.
    """
    free = load_settings(provider="openrouter")  # defaults are both ':free'
    assert free.is_free is True

    paid = load_settings(provider="openrouter",
                         proposer_model="deepseek/deepseek-chat",
                         critic_model="deepseek/deepseek-chat")
    assert paid.is_free is False


def test_a_cloud_run_still_gets_a_wall_clock_guard(store):
    """
    On a free tier the budget guard is denominated in a currency the run does
    not spend, so it cannot fire. Time is the resource actually consumed.
    """
    assert load_settings(provider="groq").max_seconds > 0


def test_reasoning_models_are_inferred_for_remote_providers(store):
    """
    A remote endpoint cannot be asked which models think, so the id is all
    there is. DeepSeek's default critic is a reasoner and must get the larger
    token budget, or it spends the whole one thinking and returns "".
    """
    settings = load_settings(provider="deepseek")
    assert "deepseek-reasoner" in settings.reasoning_models
    assert settings.budget_for("deepseek-reasoner") > settings.budget_for("deepseek-chat")


# ---------------------------------------------------------------------------
# The offline path stays sealed
# ---------------------------------------------------------------------------


def test_the_fake_provider_ignores_a_saved_key(store):
    """
    NOTHING outside the code may change what 'fake' does. Its entire purpose
    is a path that behaves identically on a bare clone.
    """
    credentials.set_key("fake", "should-be-ignored", store)
    settings = load_settings(provider="fake")
    assert settings.api_key == ""
    assert settings.key_source == ""


def test_auto_detection_never_picks_a_provider_that_needs_a_key_it_lacks(store):
    """
    Auto-detection consults the ENVIRONMENT only, never the key store: a key
    saved from the UI must not silently become the default on the next start.
    Selecting a provider is a choice; remembering its key is not the same
    choice.
    """
    credentials.set_key("groq", "gsk_from_store", store)
    assert load_settings().provider == "fake"
