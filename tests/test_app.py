"""
Smoke tests for the Streamlit UI (app.py).

WHY THIS FILE EXISTS
--------------------
`curl http://localhost:8501` returning HTTP 200 proves almost nothing.
Streamlit serves a static HTML shell immediately and only executes your
script when a browser opens a websocket. An app.py that raises on line 1
still serves a perfectly healthy 200.

`AppTest` runs the script the way a real session does, in-process, and
surfaces any exception. That is the difference between "the server is up"
and "the app works".

These are SMOKE tests, deliberately. They check that the script executes,
that the widgets exist, and that a full debate can be driven end to end
through the UI layer. They do not check pixels or styling - that is not
what automated tests are good at.

Requires: streamlit (in .venv). Skipped cleanly if absent, so the core
suite still runs on a bare Python.

Run:  .venv/bin/pytest tests/test_app.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit not installed"
).AppTest

APP = str(ROOT / "app.py")
TIMEOUT = 60  # generous: a fake debate is instant, but CI can be slow

# EVERY test in this file must go through _fresh(), which pins the provider.
# Two of them once built AppTest directly, and the moment a local Ollama was
# configured as the default those two started driving REAL inference: 20
# seconds of runtime and a genuinely flaky assertion, because a real model
# does not reliably deadlock on cue. A UI test that reaches a GPU is not a
# UI test. If you add a case here, start it with _fresh().


def _fresh(provider: str = "fake"):
    """
    A freshly loaded app, pinned to a provider.

    PINNING IS NOT OPTIONAL. The app's default provider comes from config
    (AEGIS_PROVIDER in .env), so a machine with a local Ollama configured
    would make these tests drive REAL inference - which is slow, needs a
    running daemon, and produces different text every run. A UI test that
    depends on a background service is not testing the UI.

    'fake' keeps the assertions about routing and rendering, which is what
    this file is for. Answer quality is measured by the eval harness.
    """
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    # The provider picker DISPLAYS each provider's human label (format_func)
    # but its .value/set_value still work in terms of the internal id.
    if at.selectbox and at.selectbox[0].value != provider:
        at.selectbox[0].set_value(provider).run()
    return at


def test_app_script_executes_without_exception():
    """The single most valuable assertion in this file."""
    at = _fresh()
    assert not at.exception, [str(e) for e in at.exception]


def test_core_widgets_are_present():
    at = _fresh()
    assert at.text_area, "topic input missing"
    assert at.button, "run button missing"
    assert at.selectbox, "provider selector missing"
    assert at.slider, "max rounds slider missing"


def test_default_provider_never_requires_an_api_key(monkeypatch):
    """
    No API key on a fresh machine must not mean a broken app.

    Note what is asserted: the PRINCIPLE (the default costs nothing and
    needs no credential), not the literal string "fake". This test used to
    pin the string, and that made it fail the moment the default became
    configurable - even though the invariant it was defending still held
    perfectly. A test that pins an implementation detail as a proxy for a
    rule will eventually fail for reasons that have nothing to do with the
    rule.

    'fake' and a local 'ollama' both satisfy it. A cloud provider does not.

    THE ENVIRONMENT IS CLEARED FIRST, and that is the whole point. This test
    used to read whatever .env the developer happened to have, so it was
    really asserting "this machine is currently configured harmlessly" - and
    it duly failed the moment a real OpenRouter key was configured, which is
    not a regression in anything. "A FRESH CLONE runs with no credential" is a
    statement about an empty environment, so the test has to supply one.
    """
    from aegis import PROVIDERS

    # monkeypatch, not os.environ.pop: these have to be put back, or every
    # test that runs after this one inherits a stripped environment.
    for preset in PROVIDERS.values():
        if preset["key_env"]:
            monkeypatch.delenv(preset["key_env"], raising=False)
    monkeypatch.delenv("AEGIS_PROVIDER", raising=False)

    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    # The widget DISPLAYS each provider's human label (format_func) but
    # .value is still the internal id.
    default = at.selectbox[0].value
    assert PROVIDERS[default]["requires_key"] is False, (
        f"default provider {default!r} demands a credential, so a fresh "
        f"clone would not run"
    )
    assert PROVIDERS["fake"]["label"] in list(at.selectbox[0].options), \
        "the offline provider must always remain selectable"


def test_fake_provider_announces_that_it_is_offline():
    """It must say so, rather than silently pretending to be live."""
    at = _fresh("fake")
    assert any("Offline mode" in str(i.value) for i in at.info)


def test_run_button_explains_when_a_topic_is_missing():
    at = _fresh()
    assert at.button[0].disabled is False
    at.button[0].click().run()
    assert any("Enter a question" in str(w.value) for w in at.warning)

    at.text_area[0].set_value("Is premature optimization always bad?").run()
    assert at.button[0].disabled is False


def test_full_debate_runs_through_the_ui_and_renders_a_result():
    """End-to-end through the UI layer, using the offline fake provider."""
    at = _fresh()
    at.text_area[0].set_value("Should a two-person startup use Kubernetes?").run()
    at.button[0].click().run()

    assert not at.exception, [str(e) for e in at.exception]

    state = at.session_state["final_state"]
    assert state.stop_reason == "approved"
    assert state.answer
    assert len(state.transcript) == 4          # 2 rounds x 2 agents

    # The UI must actually communicate the outcome, not just hold it.
    assert at.success, "expected a success banner for an approved debate"
    # The rounds/tokens/cost strip is a single markdown block (own HTML, not
    # st.metric — see stat_strip in app.py: Streamlit columns compress rather
    # than wrap on a narrow screen, which broke the metric values there).
    assert any("ag-stats" in str(m.value) for m in at.markdown), \
        "expected the rounds/tokens/cost stat strip"


def _checkbox_by_label(at, needle: str):
    for cb in at.checkbox:
        if needle.lower() in cb.label.lower():
            return cb
    raise AssertionError(f"no checkbox matching {needle!r}")


def test_deadlock_is_shown_as_arbitrated_not_as_success():
    """
    The most important UX assertion in the project.

    An arbitrated answer means the two agents DISAGREED and a third broke
    the tie. Rendering that with a green success banner would launder a
    contested judgement as a consensus - destroying the most useful thing
    the system just learned about the question. It must be visually
    distinct from approval.
    """
    at = _fresh()                              # pinned: see _fresh's docstring
    at.slider[0].set_value(1).run()            # cap at 1: cannot reach approval
    at.text_area[0].set_value("Any topic at all").run()
    at.button[0].click().run()

    assert not at.exception, [str(e) for e in at.exception]

    state = at.session_state["final_state"]
    assert state.stop_reason == "arbitrated"
    assert not at.success, "a contested ruling must not look like agreement"
    assert any("Arbiter" in str(i.value) for i in at.info), \
        "the UI must say the answer was adjudicated"


def test_capped_run_with_arbiter_off_warns_it_was_never_approved():
    """
    With escalation disabled, the old weakest outcome returns - and the
    warning that goes with it becomes mandatory again.
    """
    at = _fresh()                              # pinned: see _fresh's docstring
    _checkbox_by_label(at, "Arbiter").set_value(False).run()
    at.slider[0].set_value(1).run()
    at.text_area[0].set_value("Any topic at all").run()
    at.button[0].click().run()

    assert not at.exception, [str(e) for e in at.exception]

    state = at.session_state["final_state"]
    assert state.stop_reason == "max_rounds"
    assert at.warning, "an unreviewed answer MUST warn the user"
    assert not at.success


def test_clear_button_resets_the_result():
    at = _fresh()
    at.text_area[0].set_value("Topic").run()
    at.button[0].click().run()
    assert "final_state" in at.session_state

    # Button index 1 is "Clear". It calls st.rerun(), which AppTest
    # surfaces as the script simply running again.
    at.button[1].click().run()

    # NOTE: use `in`, not `.get()`. AppTest exposes a SafeSessionState
    # proxy whose __getattr__ forwards into the state dict - so calling
    # `.get(...)` makes it hunt for a session key literally named "get"
    # and raise AttributeError. A proxy object that looks like a dict is
    # not a dict; check membership explicitly.
    assert "final_state" not in at.session_state


# ---------------------------------------------------------------------------
# Turn folding
# ---------------------------------------------------------------------------


def _preview_fn():
    """
    Pull `_preview` out of app.py without executing the module.

    app.py calls st.set_page_config at import, so it cannot simply be
    imported here. Compiling the one function is enough to test it, and it
    keeps this assertion on the real source rather than a copy that would
    drift.
    """
    import ast

    tree = ast.parse(Path(APP).read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_preview")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), APP, "exec"), ns)
    return ns["_preview"]


def test_a_short_turn_is_not_folded():
    assert _preview_fn()("short answer", 1400) == "short answer"


def test_a_long_turn_is_truncated_and_marked_as_truncated():
    """
    The fold must be visible. Silently showing 1400 of 4000 characters with
    no indication would read as the model stopping mid-sentence, which is a
    real failure mode this project reports elsewhere - so it must not be
    imitated by the renderer.
    """
    preview = _preview_fn()
    long = "word " * 2000
    out = preview(long, 1400)
    assert out.endswith("…")
    assert len(out) < len(long)


def test_the_cut_prefers_a_paragraph_break():
    """
    An arbitrary offset lands mid-table often enough to matter, and half a
    markdown table renders as a wall of pipes - which looks like a bug
    rather than a fold.
    """
    preview = _preview_fn()
    text = "a" * 200 + "\n\n" + "b" * 400
    out = preview(text, 300)
    assert "b" not in out, "should have cut at the paragraph break"


def test_a_hard_cut_is_used_when_there_is_no_late_break():
    """Falling back must not produce an almost-empty preview."""
    preview = _preview_fn()
    out = preview("z" * 1000, 300)
    assert 250 < len(out) <= 305


def test_the_strictness_control_defaults_to_the_calibrated_mode():
    """
    The UI must not be the thing that quietly selects a stricter gate.

    'calibrated' is the mode the 100%-balanced-accuracy measurement in
    aegis/calibration.py actually describes; if the picker defaulted to
    'adversarial', that measurement would stop describing what runs.
    """
    at = _fresh()
    strictness = [r for r in at.radio if "strictness" in r.label.lower()]
    assert strictness, "critic strictness control missing"
    assert strictness[0].value == "calibrated"
    assert list(strictness[0].options) == ["calibrated", "adversarial"]
