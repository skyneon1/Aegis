"""
Aegis — live observability UI for a multi-agent debate.

    ./dev.sh          (or: streamlit run app.py)

WHAT THIS SCREEN IS FOR
-----------------------
Not "chat with a model". The interesting object is the SYSTEM, not the
answer: who is speaking, what they were actually sent, how fast it
generated, which claims are still contested, and - above all - why the graph
routed the way it did. An agent system that shows only its final output is
one you cannot debug, because every failure looks identical from outside: a
disappointing paragraph.

WHY THE DEBATE IS LAID OUT SIDE BY SIDE, BY ROUND
-------------------------------------------------
The two agents are adversaries, and a single scrolling column hides that.
Each round gets one row with the Proposer on the left and the Critic on the
right, so a round reads as a clash rather than two unrelated posts, and a
rebuttal sits next to the objection it answers. The Arbiter breaks the grid
deliberately - it spans both columns, because it is not a participant in the
argument, it is the thing that ends it.

The layout follows the questions you actually ask, in the order you ask them:

    what is happening NOW     -> pipeline rail + live token stream
    what is still contested   -> the ledger (aegis.state.Point)
    why did it stop           -> Decisions tab (written by the router)
    what did the agent SEE    -> Prompts tab (captured per turn)
    why was it slow           -> Telemetry tab + the endpoint panel
    what did it produce       -> Answer tab

VISUAL LANGUAGE
---------------
An instrument panel, not a landing page. Hairlines instead of cards, flat
surfaces instead of gradients and blur, mono micro-type for every number and
identifier, and colour reserved for MEANING - agent identity and run status -
so that a coloured thing on this screen always tells you something.

That is a working constraint, not taste. The previous version rendered
drop-shadowed glass panels with a gradient wordmark, and the cost was
legibility: when the chrome is the most visually assertive thing on screen,
a contested point and a decorative border compete for the same attention.
Every pixel of emphasis here is spent on data.

DESIGN CONSTRAINT FOR THIS FILE
-------------------------------
ZERO orchestration logic. No routing decisions, no verdict parsing, no round
counting, no prompt text. It calls `stream_debate()` and renders what comes
back.

That constraint is why this file has been rewritten three times without
touching the engine. Every panel reads a field the engine already records.
Each time something was genuinely missing, the fix was to make the ENGINE
record it (`Decision` for routing, `Point` for the ledger), never to teach
the UI to re-derive it. Reaching for `if verdict ==` in here is always the
wrong fix - there are lookup tables below for the cases that tempt you.
"""

from __future__ import annotations

import html
import os
import time
from pathlib import Path

import streamlit as st

# OPTIONAL UI COMPONENTS.
#
# Imported defensively on purpose. The core engine depends on almost nothing
# and the UI is the only part that wants these, so a missing component library
# must degrade to the built-in widget rather than take the whole app down —
# `pip install -r requirements.txt` on a machine that only ever runs the CLI
# should not be a prerequisite for `streamlit run app.py` working at all.
#
# Every use site below is guarded by the matching HAVE_* flag.
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    HAVE_AGGRID = True
except ImportError:  # pragma: no cover
    HAVE_AGGRID = False

try:
    from streamlit_option_menu import option_menu
    HAVE_OPTION_MENU = True
except ImportError:  # pragma: no cover
    HAVE_OPTION_MENU = False

from aegis import (
    PROVIDERS,
    catalog,
    credentials,
    forget_api_key,
    hostinfo,
    keystore_path,
    load_settings,
    providers as prov,
    save_api_key,
    save_run,
    stream_debate,
    to_markdown,
)

st.set_page_config(page_title="Aegis — Multi-Agent Debate", page_icon="⬡",
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------
# Presentation tables.
#
# LOOKUPS, not logic. The engine decides what happened and names it
# (`stop_reason`, `agent`, `rule`, `status`); this file decides only what
# colour it gets. A dict also cannot mishandle a value added later - it
# renders it plainly instead of picking a wrong branch.
# --------------------------------------------------------------------------

AGENT = {
    "proposer":   {"mark": "P", "label": "PROPOSER", "c": "#4a86e8",
                   "role": "argues for an answer"},
    "critic":     {"mark": "C", "label": "CRITIC", "c": "#c9840f",
                   "role": "attacks it"},
    "arbiter":    {"mark": "A", "label": "ARBITER", "c": "#8257e6",
                   "role": "rules on the deadlock"},
    "researcher": {"mark": "R", "label": "RESEARCH", "c": "#0e9384",
                   "role": "gathers evidence first"},
}
FALLBACK = {"mark": "·", "label": "AGENT", "c": "#8b949e", "role": ""}

VERDICT_MARK = {"APPROVE": "✔", "REVISE": "↻", "RULED": "§"}

RULE_STYLE = {
    "approved":   ("✔", "#1fa35c"),
    "revise":     ("↻", "#c9840f"),
    "deadlock":   ("⚠", "#8257e6"),
    "arbitrated": ("§", "#8257e6"),
    "guard":      ("■", "#e0544f"),
    "error":      ("✕", "#e0544f"),
}

POINT_STYLE = {
    "open":      ("○", "#c9840f", "open"),
    "closed":    ("●", "#1fa35c", "closed"),
    "withdrawn": ("◍", "#8b949e", "withdrawn"),
    "conceded":  ("●", "#1fa35c", "conceded"),
}

# What the three outcomes mean, and how far to trust each. Kept as data
# because it is rendered in two places (the idle screen and the result
# banner) and a second copy would eventually disagree with the first.
OUTCOME = {
    "approved": ("✔", "#1fa35c", "approved",
                 "The Critic was satisfied. Strongest result."),
    "arbitrated": ("§", "#8257e6", "arbitrated",
                   "They disagreed; a third agent ruled. A considered "
                   "judgement on a contested question — not a consensus."),
    "max_rounds": ("●", "#c9840f", "max_rounds",
                   "No agreement and no Arbiter. An unreviewed draft. Weakest."),
}


def agent_style(name: str) -> dict:
    return AGENT.get(name, FALLBACK)


def esc(text: object) -> str:
    """
    Escape before interpolating into any HTML this file builds.

    Not paranoia about a hostile user - the danger is ordinary model output.
    A Critic that writes `<2 replicas` or `a < b` silently eats the rest of
    the line when it lands in innerHTML, and the bug presents as "the model
    stopped mid-sentence", which sends you looking at the model.
    """
    return html.escape("" if text is None else str(text), quote=True)


# --------------------------------------------------------------------------
# Styling
#
# Written against TRANSLUCENT GREYS (rgba(128,128,128,…)) rather than fixed
# hex values, because .streamlit/config.toml deliberately sets no `base`:
# the viewer's own light/dark preference stands, and the same stylesheet has
# to read correctly on both. A hardcoded #f8f9fa surface is invisible on one
# of them, and which one depends on a setting this file cannot see.
#
# The accent colours are the exception. They are chosen to clear 3:1 contrast
# against BOTH backgrounds, because agent identity is information and has to
# survive the theme.
# --------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;450;500;600;700&display=swap');

:root {
  /* Structure. Three weights of line and two of fill is the entire
     vocabulary — every panel below is built from these, which is what keeps
     the screen looking like one instrument rather than six widgets. */
  --ag-line:      rgba(128,128,128,0.30);
  --ag-line-soft: rgba(128,128,128,0.16);
  --ag-fill:      rgba(128,128,128,0.05);
  --ag-fill-2:    rgba(128,128,128,0.10);

  /* Identity. Used for the agent rules and nothing decorative. */
  --ag-proposer:   #4a86e8;
  --ag-critic:     #c9840f;
  --ag-arbiter:    #8257e6;
  --ag-researcher: #0e9384;

  /* State. */
  --ag-ok:   #1fa35c;
  --ag-warn: #c9840f;
  --ag-bad:  #e0544f;

  --ag-mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --ag-sans: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* [data-testid="stIconMaterial"] is EXCLUDED here and re-pinned below.
   Streamlit's chevrons (sidebar collapse, expander) are a ligature font —
   the glyph is literally the text "keyboard_double_arrow_left" rendered in
   Material Symbols. Overriding font-family broadly once turned every one of
   those glyphs into its own name, spelled out, overlapping the label next
   to it. The icon font has to survive this rule everywhere it is used. */
html, body, .stApp, [class*="st-"] { font-family: var(--ag-sans); }
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important; }
code, pre, kbd, .stCode, [data-testid="stCode"] { font-family: var(--ag-mono) !important; }

/* Numbers are compared vertically all over this screen — telemetry rows,
   the stat strip, the ledger. Proportional digits make that comparison
   require reading; tabular digits make it require looking. */
.stApp { font-variant-numeric: tabular-nums; }

/* Streamlit chrome: remove what we do not use, keep what navigates.
   The sidebar toggle is deliberately preserved and pinned — an earlier
   sweep hid the whole header and took the only way to reopen a collapsed
   sidebar with it. */
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"], [data-testid="stMainMenu"] { display: none !important; }

[data-testid="stHeader"] { height: 0 !important; min-height: 0 !important;
                           background: transparent !important; }

[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"] {
  display: inline-flex !important; visibility: visible !important;
  position: fixed !important; top: 0.5rem !important; left: 0.5rem !important;
  z-index: 1000001 !important;
  border: 1px solid var(--ag-line) !important; border-radius: 3px !important;
  background: var(--ag-fill-2) !important; backdrop-filter: none !important;
  box-shadow: none !important;
}

.block-container {
  max-width: 1400px !important;
  padding: 1.1rem 1.5rem 4rem !important;
}

/* ---------------------------------------------------------------- masthead */
.ag-head {
  display: flex; align-items: baseline; gap: 0.9rem; flex-wrap: wrap;
  padding-bottom: 0.55rem;
  border-bottom: 1px solid var(--ag-line);
  margin-bottom: 1.15rem;
}
.ag-brand {
  display: inline-flex; align-items: center; gap: 0.45rem;
  font-family: var(--ag-mono); font-weight: 600; font-size: 0.95rem;
  letter-spacing: 0.22em; text-transform: uppercase;
}
.ag-mark { width: 0.95rem; height: 0.95rem; flex: 0 0 auto; opacity: 0.85; }
.ag-sub { font-size: 0.83rem; opacity: 0.62; letter-spacing: 0.005em; }
.ag-headmeta {
  margin-left: auto; display: flex; gap: 0.55rem; flex-wrap: wrap;
  font-family: var(--ag-mono); font-size: 0.7rem; opacity: 0.72;
}
.ag-headmeta b { font-weight: 600; opacity: 1; }

/* Micro-label. The uppercase mono caption that titles every region. One
   class, used everywhere, so the screen has exactly one heading idiom. */
.ag-label {
  font-family: var(--ag-mono); font-size: 0.66rem; font-weight: 600;
  letter-spacing: 0.15em; text-transform: uppercase; opacity: 0.55;
  margin: 0 0 0.5rem;
}
.ag-note { font-size: 0.8rem; line-height: 1.55; opacity: 0.66; }
.ag-note code { font-size: 0.76rem; }

/* ------------------------------------------------------------ pipeline rail */
/* A rail, not a row of pills: the nodes sit ON a hairline so the eye reads a
   path with a position on it. The previous pill row read as three unrelated
   badges, which is exactly the wrong mental model for a state machine. */
.ag-rail {
  display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
  padding: 0.5rem 0.1rem; margin-bottom: 1.1rem;
  border-top: 1px solid var(--ag-line-soft);
  border-bottom: 1px solid var(--ag-line-soft);
  font-family: var(--ag-mono); font-size: 0.68rem;
}
.ag-node {
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.16rem 0.1rem; letter-spacing: 0.12em; font-weight: 600;
  opacity: 0.34; color: inherit; white-space: nowrap;
}
.ag-node::before {
  content: ""; width: 6px; height: 6px; flex: 0 0 6px;
  border: 1.5px solid currentColor; border-radius: 1px;
}
.ag-node.done { opacity: 0.68; }
.ag-node.done::before { background: currentColor; }
.ag-node.on { opacity: 1; }
.ag-node.on::before { background: currentColor; box-shadow: 0 0 0 3px color-mix(in srgb, currentColor 22%, transparent); }
.ag-link { flex: 1 1 1.4rem; min-width: 1rem; height: 1px; background: var(--ag-line); opacity: 0.7; }
.ag-back { opacity: 0.42; font-size: 0.8rem; }
.ag-railmeta { margin-left: auto; opacity: 0.6; white-space: nowrap; }

/* ------------------------------------------------------------------- turns */
/* No box. A 2px identity rule on the left, a header line, and the text.
   The card border, fill, blur and shadow it replaces were four separate
   ways of saying the same thing the colour already says. */
.ag-turn {
  border-left: 2px solid var(--c);
  padding: 0.1rem 0 0.1rem 0.75rem;
  margin: 0 0 0.35rem;
}
.ag-turn.live { border-left-style: dashed; }
.ag-thead {
  display: flex; align-items: baseline; gap: 0.5rem; flex-wrap: wrap;
  font-family: var(--ag-mono); font-size: 0.68rem; margin-bottom: 0.35rem;
}
.ag-who {
  color: var(--c); font-weight: 600; letter-spacing: 0.13em;
  text-transform: uppercase; white-space: nowrap;
}
.ag-vd {
  font-weight: 600; letter-spacing: 0.08em; padding: 0 0.3rem;
  border: 1px solid currentColor; border-radius: 2px; font-size: 0.62rem;
}
.ag-tele { margin-left: auto; opacity: 0.55; white-space: nowrap; }

/* A reasoning model's internal monologue. Set apart and de-emphasised: it
   is evidence, not output, and the one thing worse than hiding it is
   letting it compete with the answer for attention. */
.ag-think {
  font-family: var(--ag-mono); font-size: 0.72rem; line-height: 1.6;
  opacity: 0.7; white-space: pre-wrap;
  border-left: 1px solid var(--ag-line); padding: 0.35rem 0 0.35rem 0.6rem;
  margin: 0.3rem 0 0.6rem; max-height: 11rem; overflow-y: auto;
}

/* ---------------------------------------------------------- turn spacing */
/* The empty-gutter fix lives in Python, not here. A round is one row, so the
   taller cell decides how much blank space the shorter one shows; an
   1100-word Proposer answer beside an 80-word critique left most of a screen
   empty and stopped the two reading as a pair. The first attempt clamped the
   height in CSS, which cannot work: Streamlit wraps every st.markdown call in
   its own container, so a <div> opened in one call and closed in another
   produces two empty divs and wraps nothing. The text is truncated instead —
   see LiveDebate.FOLD_CHARS and _preview. */

/* Prose in a turn gets a reading measure. Full-bleed text in a 700px column
   is ~110 characters a line, which is roughly twice the comfortable span. */
.ag-turn .stMarkdown p, .ag-turn .stMarkdown li { max-width: 68ch; }

/* Round separator — a labelled rule, so the scroll has structure to it. */
.ag-round {
  display: flex; align-items: center; gap: 0.7rem;
  margin: 1.6rem 0 0.7rem;
  font-family: var(--ag-mono); font-size: 0.64rem; font-weight: 600;
  letter-spacing: 0.16em; text-transform: uppercase; opacity: 0.5;
}
.ag-round::after {
  content: ""; flex: 1 1 auto; height: 1px; background: var(--ag-line);
}

/* A long debate is a long page; keep the run's status line in view.

   THE BACKGROUND IS DELIBERATELY NOT A COLOUR, and the reason is a bug this
   replaced. The first version painted white and then flipped to near-black
   under `@media (prefers-color-scheme: dark)`. That media query reports the
   OPERATING SYSTEM's preference, which is not the theme Streamlit rendered:
   on a machine set to dark OS with Streamlit serving its light theme, the
   rail painted itself near-black behind dark-grey text and the run status
   became unreadable.

   Streamlit exposes no CSS variable for its own background (checked), so
   there is nothing to read. A blurred backdrop under a neutral grey needs no
   answer to the question: it darkens a light page slightly, lightens a dark
   one slightly, and the text keeps whatever colour the theme already gave it.
   The blur here is doing legibility work over scrolling content, which is
   the one thing it is actually for. */
.ag-rail.sticky {
  position: sticky; top: 0; z-index: 50;
  background: rgba(128,128,128,0.10);
  backdrop-filter: blur(12px) saturate(1.3);
  -webkit-backdrop-filter: blur(12px) saturate(1.3);
}

/* ------------------------------------------------------------ stat strip */
/* Own HTML rather than st.columns(6): six Streamlit columns do not stack,
   they compress, so on a phone the metric values wrapped mid-number. A grid
   with a minimum column width reflows into two rows instead. */
.ag-stats {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
  border: 1px solid var(--ag-line-soft); border-radius: 3px;
  margin: 0.2rem 0 1.3rem; overflow: hidden;
}
.ag-stat { padding: 0.6rem 0.8rem; border-left: 1px solid var(--ag-line-soft); }
.ag-stat:first-child { border-left: none; }
.ag-stat .k {
  display: block; font-family: var(--ag-mono); font-size: 0.62rem;
  letter-spacing: 0.13em; text-transform: uppercase; opacity: 0.52;
  margin-bottom: 0.22rem;
}
.ag-stat .v { display: block; font-size: 1.05rem; font-weight: 600; letter-spacing: -0.01em; }

/* --------------------------------------------------------------- ledger */
.ag-ledger { border-top: 1px solid var(--ag-line-soft); margin: 0.2rem 0 1rem; }
.ag-pt {
  display: flex; gap: 0.7rem; align-items: baseline;
  font-size: 0.83rem; line-height: 1.55; padding: 0.5rem 0.1rem;
  border-bottom: 1px solid var(--ag-line-soft);
}
.ag-pt .g {
  color: var(--pc); font-family: var(--ag-mono); font-weight: 600;
  font-size: 0.72rem; white-space: nowrap; flex: 0 0 auto;
}
.ag-pt .t { flex: 1 1 auto; min-width: 0; }
.ag-pt .s {
  font-family: var(--ag-mono); font-size: 0.65rem; opacity: 0.55;
  letter-spacing: 0.06em; text-transform: uppercase;
  white-space: nowrap; flex: 0 0 auto;
}

/* ------------------------------------------------------------- decisions */
.ag-dec {
  border-left: 2px solid var(--c); padding: 0.1rem 0 0.1rem 0.7rem;
  margin: 0.9rem 0; font-size: 0.85rem; line-height: 1.55;
}
.ag-dec .h {
  font-family: var(--ag-mono); font-size: 0.68rem; letter-spacing: 0.1em;
  margin-bottom: 0.2rem;
}
.ag-obs {
  font-family: var(--ag-mono); font-size: 0.68rem; opacity: 0.55;
  margin-top: 0.3rem;
}

/* --------------------------------------------------------- provider table */
/* The point of this table is the Key column: it answers "what can I
   actually run right now" in one glance, which is the question a fourteen-
   entry provider picker creates and cannot itself answer. */
.ag-tablewrap { overflow-x: auto; border: 1px solid var(--ag-line-soft); border-radius: 3px; }
table.ag-prov { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
table.ag-prov th {
  text-align: left; font-family: var(--ag-mono); font-size: 0.62rem;
  letter-spacing: 0.13em; text-transform: uppercase; opacity: 0.55;
  font-weight: 600; padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--ag-line);
  white-space: nowrap;
}
table.ag-prov td {
  padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--ag-line-soft);
  vertical-align: top; line-height: 1.5;
}
table.ag-prov tr:last-child td { border-bottom: none; }
table.ag-prov td.n { font-weight: 600; white-space: nowrap; }
table.ag-prov td.f { font-size: 0.78rem; opacity: 0.72; min-width: 17rem; }
table.ag-prov td.k { font-family: var(--ag-mono); font-size: 0.7rem; white-space: nowrap; }
.ag-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%;
          margin-right: 0.4rem; vertical-align: middle; }

/* ------------------------------------------------------------- outcomes */
.ag-out { display: flex; gap: 0.6rem; align-items: baseline; padding: 0.45rem 0;
          border-top: 1px solid var(--ag-line-soft); font-size: 0.83rem; line-height: 1.55; }
.ag-out .g { color: var(--pc); font-family: var(--ag-mono); font-weight: 600;
             font-size: 0.72rem; white-space: nowrap; flex: 0 0 7.5rem; }

/* ------------------------------------------------- sidebar / control column */
section[data-testid="stSidebar"] { border-right: 1px solid var(--ag-line); }
section[data-testid="stSidebar"] > div { background: transparent; }
section[data-testid="stSidebar"] .block-container,
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  padding-top: 0.9rem !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
  height: 0 !important; min-height: 0 !important; overflow: visible !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"] {
  visibility: visible !important; position: absolute !important;
  top: 0.5rem !important; right: 0.5rem !important; z-index: 2 !important;
}
/* Sidebar labels adopt the same micro-type as the rest of the screen, so a
   control and the panel it drives look like they belong to one system. */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font-family: var(--ag-mono); font-size: 0.66rem; font-weight: 600;
  letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.62;
}
.ag-kv {
  display: flex; justify-content: space-between; gap: 0.6rem;
  font-family: var(--ag-mono); font-size: 0.7rem; padding: 0.22rem 0;
  border-bottom: 1px solid var(--ag-line-soft);
}
.ag-kv .k { opacity: 0.55; white-space: nowrap; }
.ag-kv .v { text-align: right; word-break: break-all; }
.ag-hint { font-size: 0.72rem; line-height: 1.5; opacity: 0.62; margin: 0.35rem 0 0; }
.ag-hint a { color: inherit; }

/* ----------------------------------------------------- Streamlit widgets */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: 3px !important; font-weight: 500 !important;
  font-size: 0.85rem !important; letter-spacing: 0.01em !important;
  min-height: 2.35rem !important; box-shadow: none !important;
  transition: background-color 0.12s ease, border-color 0.12s ease !important;
}
.stButton > button:hover { transform: none !important; }
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {
  border-radius: 3px !important; font-size: 0.92rem !important;
}
[data-testid="stTextArea"] textarea { line-height: 1.5 !important; }
[data-testid="stExpander"] {
  border: 1px solid var(--ag-line-soft) !important; border-radius: 3px !important;
  box-shadow: none !important;
}
[data-testid="stExpander"] summary { font-size: 0.82rem !important; }
[data-testid="stExpander"] summary p { font-weight: 500 !important; }

.stTabs [data-baseweb="tab-list"] {
  gap: 0.15rem; border-bottom: 1px solid var(--ag-line);
  overflow-x: auto; scrollbar-width: thin;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--ag-mono); font-size: 0.7rem; font-weight: 500;
  letter-spacing: 0.08em; text-transform: uppercase;
  padding: 0.4rem 0.7rem; white-space: nowrap;
}
.stTabs [data-baseweb="tab-highlight"] { height: 2px; }

[data-testid="stAlert"] { border-radius: 3px !important; font-size: 0.86rem; }
[data-testid="stProgressBar"] div { border-radius: 2px !important; }
hr, [data-testid="stDivider"] hr { border-color: var(--ag-line-soft) !important; }

/* ------------------------------------------------------------- responsive */
/* Streamlit columns compress rather than wrap, so a two-agent debate row
   becomes two 45-character slivers on a phone. Below the breakpoint the
   horizontal blocks are forced to stack — which also puts the Proposer
   above the Critic, reading order the argument already has. */
@media (max-width: 880px) {
  .block-container { padding: 0.9rem 0.9rem 3rem !important; }
  [data-testid="stHorizontalBlock"] { flex-direction: column !important; gap: 0.3rem !important; }
  [data-testid="stColumn"], [data-testid="column"] {
    width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important;
  }
  .ag-head { gap: 0.4rem; }
  .ag-headmeta { margin-left: 0; width: 100%; }
  .ag-railmeta { margin-left: 0; width: 100%; padding-top: 0.2rem; }
  .ag-link { display: none; }
  .ag-stat { border-left: none; border-top: 1px solid var(--ag-line-soft); }
  .ag-stat:first-child { border-top: none; }
  .ag-out .g { flex: 0 0 auto; }
  table.ag-prov td.f { min-width: 13rem; }
}
@media (max-width: 520px) {
  .ag-pt { flex-wrap: wrap; }
  .ag-tele { margin-left: 0; width: 100%; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Small renderers
# --------------------------------------------------------------------------


def graph_dot(*, arbiter_on: bool, research_on: bool = False,
              visited: set[str] | None = None,
              fired: set[str] | None = None) -> str:
    """
    The actual graph as Graphviz DOT, with the routing conditions ON THE EDGES.

    WHY A DIAGRAM EARNS ITS PLACE HERE
    ----------------------------------
    The compact rail above shows live progress and counters, which a diagram
    is bad at. What a diagram is good at is the thing this project is
    actually about: that `critic` has THREE outgoing edges and which condition
    selects each. That is the whole control flow, and in a row of labels it is
    invisible - you can see the nodes and not the reason anything moves
    between them.

    `st.graphviz_chart` is built into Streamlit and renders a DOT string
    client-side, so this adds no dependency (the `graphviz` Python package is
    not installed and is not needed).

    `fired` comes from the engine's own decision log, so the highlighted path
    is the path the run actually took - not a re-derivation of the router's
    conditions, which this file is not allowed to make.
    """
    visited, fired = visited or set(), set(fired or set())
    # The static edges have no routing condition, so their firing is implied
    # by the nodes that ran. Deriving it here keeps callers from having to
    # know which edges are conditional and which are not.
    if "critic" in visited:
        fired.add("always")
    if "proposer" in visited and research_on:
        fired.add("research")
    if "arbiter" in visited:
        fired.add("arbitrated")

    idle = "#8b949e"

    def node(name: str, label: str, colour: str) -> str:
        on = name in visited
        return (f'  {name} [label="{label}", color="{colour if on else idle}", '
                f'fontcolor="{colour if on else idle}", '
                f'penwidth={"1.8" if on else "1"}, style="rounded"];')

    def edge(a: str, b: str, label: str, key: str, colour: str = idle) -> str:
        hit = key in fired
        return (f'  {a} -> {b} [label="{label}", color="{colour if hit else idle}", '
                f'fontcolor="{colour if hit else idle}", '
                f'penwidth={"1.8" if hit else "0.9"}, '
                f'style="{"solid" if hit else "dashed"}", fontsize=9];')

    lines = ['digraph aegis {', '  rankdir=LR; bgcolor="transparent";',
             '  node [shape=box, fontname="IBM Plex Mono", fontsize=9, height=.32];',
             '  edge [fontname="IBM Plex Mono"];']
    if research_on:
        lines.append(node("researcher", "RESEARCH", AGENT["researcher"]["c"]))
        lines.append(edge("researcher", "proposer", "sources", "research",
                          AGENT["researcher"]["c"]))
    lines += [
        node("proposer", "PROPOSER", AGENT["proposer"]["c"]),
        node("critic", "CRITIC", AGENT["critic"]["c"]),
        f'  END [label="END", shape=doublecircle, fontsize=8, color="{idle}", '
        f'fontcolor="{idle}", height=.26];',
        edge("proposer", "critic", "answer", "always", AGENT["proposer"]["c"]),
        # The back-edge. It is why this is a graph and not a pipeline, so it is
        # drawn as the loop it is rather than flattened into the row.
        edge("critic", "proposer", "REVISE", "revise", AGENT["critic"]["c"]),
        edge("critic", "END", "APPROVE", "approved", "#1fa35c"),
    ]
    if arbiter_on:
        lines += [node("arbiter", "ARBITER", AGENT["arbiter"]["c"]),
                  edge("critic", "arbiter", "deadlock", "deadlock",
                       AGENT["arbiter"]["c"]),
                  edge("arbiter", "END", "ruling", "arbitrated",
                       AGENT["arbiter"]["c"])]
    else:
        lines.append(edge("critic", "END", "cap hit", "deadlock",
                          AGENT["critic"]["c"]))
    lines += [edge("critic", "END", "guard tripped", "guard", "#e0544f"), '}']
    return "\n".join(lines)


def rail(active: str, visited: set[str], arbiter_on: bool,
         note: str = "") -> str:
    """The graph drawn small, with the current node lit."""
    names = ["proposer", "critic"] + (["arbiter"] if arbiter_on else [])
    out = []
    for i, name in enumerate(names):
        style = agent_style(name)
        cls = "on" if name == active else ("done" if name in visited else "")
        out.append(f"<span class='ag-node {cls}' style='color:{style['c']}'>"
                   f"{style['label']}</span>")
        if i < len(names) - 1:
            out.append("<span class='ag-link'></span>")
    # The back-edge, drawn between critic and proposer where it actually is.
    out.insert(2, "<span class='ag-back' title='the back-edge: REVISE "
                  "returns to the Proposer'>&#8634;</span>")
    return (f"<div class='ag-rail sticky'>{''.join(out)}"
            f"<span class='ag-railmeta'>{esc(note)}</span></div>")


def _preview(text: str, limit: int) -> str:
    """
    The first `limit`-ish characters of a turn, cut at a paragraph break.

    Cutting at an arbitrary offset lands mid-table or mid-list often enough
    to matter — a truncated markdown table renders as a wall of pipes, which
    looks like a bug rather than a fold. Backing up to the last blank line
    keeps the preview a complete set of blocks. If there is no break in the
    last third, fall back to a hard cut rather than showing almost nothing.
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    brk = head.rfind("\n\n")
    if brk > limit * 0.6:
        head = head[:brk]
    return head.rstrip() + "\n\n…"


def stat_strip(pairs) -> str:
    """The run's headline numbers, as one reflowing grid. See .ag-stats."""
    cells = "".join(
        f"<div class='ag-stat'><span class='k'>{esc(k)}</span>"
        f"<span class='v'>{esc(v)}</span></div>" for k, v in pairs)
    return f"<div class='ag-stats'>{cells}</div>"


def ledger_rail(points, title: str = "Contested points") -> str:
    """The disagreement ledger: every claim and where it stands."""
    if not points:
        return ""
    rows = []
    for p in points:
        glyph, colour, label = POINT_STYLE.get(p.status,
                                               ("•", "#8b949e", p.status))
        extra = ""
        if p.status == "withdrawn":
            extra = " · conceded"
        elif p.proposer_stance == "DISPUTED" and p.is_open:
            extra = " · rebutted"
        rows.append(
            f"<div class='ag-pt' style='--pc:{colour}'>"
            f"<span class='g'>{glyph} {esc(p.id)}</span>"
            f"<span class='t'>{esc(p.text)}</span>"
            f"<span class='s'>{esc(label)}{esc(extra)}</span></div>"
        )
    return (f"<div class='ag-label'>{esc(title)}</div>"
            f"<div class='ag-ledger'>{''.join(rows)}</div>")


def outcome_key() -> str:
    """The three stop reasons and how far each may be trusted."""
    rows = "".join(
        f"<div class='ag-out' style='--pc:{colour}'>"
        f"<span class='g'>{glyph} {esc(name)}</span>"
        f"<span>{esc(meaning)}</span></div>"
        for glyph, colour, name, meaning in OUTCOME.values())
    return rows


def provider_table(current: str, env_keys: dict, stored: set) -> str:
    """
    Every endpoint, what its free tier gives you, and whether it can run now.

    This exists because a fourteen-entry picker poses a question it cannot
    answer: which of these do I actually have access to? Without the Key
    column the only way to find out is to select each one in turn and read
    an error.
    """
    rows = []
    for name, preset in PROVIDERS.items():
        if name == "custom":
            continue
        if name in env_keys:
            dot, key_state = "#1fa35c", "in .env"
        elif name in stored:
            dot, key_state = "#1fa35c", "saved"
        elif not preset.get("requires_key"):
            dot, key_state = "#1fa35c", "none needed"
        else:
            dot, key_state = "rgba(128,128,128,0.45)", "not set"
        here = " ← selected" if name == current else ""
        rows.append(
            f"<tr><td class='n'><span class='ag-dot' style='background:{dot}'>"
            f"</span>{esc(preset.get('label', name))}"
            f"<span style='opacity:.5'>{esc(here)}</span></td>"
            f"<td class='f'>{esc(preset.get('free_note', ''))}</td>"
            f"<td class='k' style='opacity:.7'>{esc(key_state)}</td></tr>")
    return ("<div class='ag-tablewrap'><table class='ag-prov'>"
            "<tr><th>Provider</th><th>What you get free</th><th>Key</th></tr>"
            + "".join(rows) + "</table></div>")


# --------------------------------------------------------------------------
# Model discovery
#
# CACHED, and the cache is the point. Every Streamlit interaction re-runs
# this whole script, so an uncached /models call would fire on every
# keystroke in the topic box - a request per character, to a rate-limited
# free tier, to render a dropdown that has not changed. The TTL is long
# enough to cover a working session and the Refresh button covers the rest.
# --------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def _remote_catalogue(base_url: str, api_key: str,
                      headers: tuple) -> tuple[list, bool, str]:
    probe = catalog.probe(base_url, api_key, dict(headers))
    return probe.models, probe.ok, probe.detail


@st.cache_data(ttl=60, show_spinner=False)
def _local_catalogue(base_url: str) -> tuple[list, set]:
    installed = hostinfo.available_models(base_url)
    return installed, hostinfo.reasoning_models(installed, base_url)


def discover_models(settings) -> tuple[list, set, str]:
    """
    (models, thinking models, a one-line note about where the list came from).

    Three sources, in descending order of authority: the server's own
    listing, the curated table in providers.py, and nothing. The note is
    returned rather than logged because "why is my model not in this list"
    is otherwise unanswerable from the screen.
    """
    if settings.is_fake:
        return [settings.proposer_model, settings.critic_model], set(), ""

    if settings.local:
        installed, thinking = _local_catalogue(settings.base_url)
        if installed:
            return installed, set(thinking), f"{len(installed)} pulled locally"
        return [], set(), "Ollama is not answering — is `ollama serve` running?"

    known = prov.known_models(settings.provider)
    if settings.requires_key and not settings.api_key:
        return known, prov.reasoning_models(known), "no key yet — showing presets"

    models, ok, detail = _remote_catalogue(
        settings.base_url, settings.api_key,
        tuple(sorted((settings.extra_headers or {}).items())))
    if ok and models:
        return models, prov.reasoning_models(models), f"{len(models)} live from API"
    # A provider that will not list its models is not a provider that cannot
    # serve them: several implement chat-completions and nothing else. Fall
    # back rather than presenting an empty picker, and say which happened.
    return known, prov.reasoning_models(known), detail or "showing presets"


# --------------------------------------------------------------------------
# Live rendering
# --------------------------------------------------------------------------


class LiveDebate:
    """
    Streams turns into a per-round two-column grid, then freezes them.
    """

    REPAINT_S = 0.08

    # Characters of a finished turn shown before it folds. ~26rem of prose;
    # see the "turn spacing" note in the stylesheet for why folding at all.
    FOLD_CHARS = 1400

    def __init__(self, host) -> None:
        self._host = host              # container the rows are created in
        self._cells: dict[str, object] = {}
        self._slot = None
        self._agent = ""
        self._body: list[str] = []
        self._think: list[str] = []
        self._painted = 0.0
        self._started = 0.0

    def _open_row(self) -> None:
        """One row per round: Proposer | Critic."""
        left, right = self._host.columns(2, gap="medium")
        self._cells = {"proposer": left, "critic": right}

    def _cell(self, agent: str):
        # Researcher and Arbiter both break the grid, for symmetric reasons:
        # the Researcher speaks BEFORE the argument and the Arbiter AFTER it.
        # Neither is a side in the debate, so neither belongs in a column.
        if agent in ("arbiter", "researcher"):
            return self._host.container()
        if agent == "proposer" or not self._cells:
            self._open_row()
        return self._cells.get(agent, self._host.container())

    def __call__(self, channel: str, token: str) -> None:
        agent, _, kind = channel.partition(":")
        if self._slot is None:
            self._agent = agent
            self._slot = self._cell(agent).empty()
            self._body, self._think = [], []
            self._started = time.perf_counter()

        (self._think if kind == "thinking" else self._body).append(token)

        now = time.perf_counter()
        if now - self._painted >= self.REPAINT_S:
            self._painted = now
            self._paint()

    def _paint(self) -> None:
        style = agent_style(self._agent)
        elapsed = time.perf_counter() - self._started
        body = "".join(self._body)
        thought = "".join(self._think)
        approx = max(1, len(body + thought) // 4)
        rate = approx / elapsed if elapsed > 0 else 0
        phase = "thinking" if (thought and not body) else "writing"

        with self._slot.container():
            st.markdown(
                f"<div class='ag-turn live' style='--c:{style['c']}'>"
                f"<div class='ag-thead'>"
                f"<span class='ag-who'>{style['label']}</span>"
                f"<span class='ag-tele'>{phase} · ~{approx} tok · "
                f"{rate:.0f} tok/s · {elapsed:.0f}s</span>"
                f"</div></div>", unsafe_allow_html=True)
            if thought:
                st.markdown(
                    f"<div class='ag-think'>{esc(thought[-360:])}"
                    f"{'▌' if not body else ''}</div>",
                    unsafe_allow_html=True)
            if body:
                st.markdown(body + " ▌")

    def finalise(self, turn) -> None:
        target = self._slot if self._slot is not None else \
            self._cell(turn.agent).empty()
        style = agent_style(turn.agent)

        verdict = ""
        if turn.verdict:
            mark = VERDICT_MARK.get(turn.verdict, "·")
            verdict = (f"<span class='ag-vd' style='color:{style['c']}'>"
                       f"{mark} {esc(turn.verdict)}</span>")

        with target.container():
            st.markdown(
                f"<div class='ag-turn' style='--c:{style['c']}'>"
                f"<div class='ag-thead'>"
                f"<span class='ag-who'>R{turn.round} {style['label']}</span>"
                f"{verdict}"
                f"<span class='ag-tele'>{esc(turn.model)} · "
                f"{turn.latency_s:.1f}s · ttft {turn.ttft_s:.1f}s · "
                f"{turn.tokens_per_s:.0f} tok/s · "
                f"{turn.tokens_in}→{turn.tokens_out}</span>"
                f"</div></div>", unsafe_allow_html=True)
            if turn.reasoning:
                with st.expander(f"thinking ({len(turn.reasoning):,} chars)"):
                    st.markdown(f"<div class='ag-think'>{esc(turn.reasoning)}"
                                f"</div>", unsafe_allow_html=True)

            # FOLD A LONG TURN instead of letting it set the row height.
            # A round is one row, so the taller cell decides how much blank
            # space the shorter one shows; an 1100-word Proposer answer next
            # to an 80-word critique left most of a screen empty and stopped
            # the two reading as a pair. The threshold is in characters
            # because that is what predicts rendered height here — the text
            # is prose in a fixed-width column, not arbitrary markup.
            if len(turn.content) > self.FOLD_CHARS:
                # TRUNCATE THE TEXT, do not merely hide it with CSS.
                #
                # The first attempt opened a <div> in one st.markdown call and
                # closed it in another, intending to clamp the height between
                # them. Streamlit wraps every markdown call in its own
                # container, so the two tags became two empty divs and the
                # content in between was never inside either — the clamp
                # applied to nothing and the turn rendered at full height.
                # Markup cannot span st.markdown calls; only the text can be
                # shortened.
                st.markdown(_preview(turn.content, self.FOLD_CHARS))
                with st.expander(f"read the full turn "
                                 f"({len(turn.content):,} chars)"):
                    st.markdown(turn.content)
            else:
                st.markdown(turn.content)

        self._slot = None
        self._body, self._think = [], []


# --------------------------------------------------------------------------
# Sidebar — connection, models, guards
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<div class='ag-brand'><svg class='ag-mark' viewBox='0 0 24 24' "
        "aria-hidden='true' fill='none'><path d='M12 2.8 20 7.4v9.2L12 "
        "21.2 4 16.6V7.4L12 2.8Z' stroke='currentColor' stroke-width='2' "
        "stroke-linejoin='round'/></svg><span>AEGIS</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("Multi-agent debate platform")
    st.divider()

    default_settings = load_settings()
    names = list(PROVIDERS.keys())
    default = default_settings.provider
    provider = st.selectbox(
        "Provider", names,
        index=names.index(default) if default in names else names.index("fake"),
        format_func=lambda n: PROVIDERS[n]["label"],
        help="Where tokens come from. 'ollama' runs on this machine. "
             "'fake' runs the full graph with no model at all — good for "
             "checking routing without spending anything.")

    settings = load_settings(provider=provider)
    preset = PROVIDERS[provider]

    # ---- Connection ------------------------------------------------------
    # Where a cloud key actually gets typed. Kept separate from Models
    # below because these two questions are asked at different moments:
    # "can I reach this provider at all" comes before "which model on it".
    with st.expander(
        "Connection",
        expanded=settings.requires_key and not settings.api_key,
    ):
        if not settings.requires_key:
            st.caption("This provider needs no credential." if provider != "fake"
                       else "Offline — no network call is made.")
        else:
            env_has_it = settings.key_source == "env"
            if env_has_it:
                st.caption(
                    f"Using `{preset['key_env']}` from your environment "
                    f"(`{credentials.fingerprint(settings.api_key)}`). An "
                    f"env var always wins over a saved key.")
            else:
                stored = credentials.get_key(provider, keystore_path())
                if stored:
                    st.caption(
                        f"Saved key in use: "
                        f"`{credentials.fingerprint(stored)}`")

                with st.form(f"key_form_{provider}", border=False):
                    new_key = st.text_input(
                        "API key", type="password",
                        placeholder="paste key, then Save",
                        label_visibility="collapsed")
                    fc1, fc2 = st.columns(2)
                    save_clicked = fc1.form_submit_button(
                        "Save", width="stretch")
                    forget_clicked = fc2.form_submit_button(
                        "Forget", width="stretch", disabled=not stored)

                if save_clicked:
                    if new_key.strip():
                        if save_api_key(provider, new_key):
                            st.success(f"Saved to `{keystore_path()}` (0600).")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error("Could not write the key file.")
                    else:
                        st.warning("Paste a key first.")
                if forget_clicked and stored:
                    forget_api_key(provider)
                    st.cache_data.clear()
                    st.rerun()

                if preset.get("console"):
                    st.markdown(
                        f"<p class='ag-hint'>Get a key: "
                        f"<a href='{esc(preset['console'])}' target='_blank'>"
                        f"{esc(preset['console'])}</a></p>",
                        unsafe_allow_html=True)
                if preset.get("free_note"):
                    st.markdown(f"<p class='ag-hint'>{esc(preset['free_note'])}"
                               f"</p>", unsafe_allow_html=True)

        if provider == "custom":
            settings.base_url = st.text_input(
                "Base URL", settings.base_url or "",
                placeholder="https://your-endpoint/v1")

    max_rounds = st.slider(
        "Max rounds", 1, 8, settings.max_rounds,
        help="Hard iteration cap. Without it the two can disagree forever.")

    use_arbiter = st.checkbox(
        "Arbiter on deadlock", value=settings.use_arbiter,
        help="If they never agree, a third agent reads the whole exchange and "
             "rules. Runs at most once, and only on deadlock.")

    _STRICTNESS = ["calibrated", "adversarial"]
    critic_strictness = st.radio(
        "Critic strictness", _STRICTNESS,
        index=_STRICTNESS.index(settings.critic_strictness)
        if settings.critic_strictness in _STRICTNESS else 0,
        horizontal=True,
        help="'calibrated' is the tuned default — it passes a good answer "
             "and catches invented statistics, false mechanisms and "
             "non-answers. 'adversarial' also treats unstated load-bearing "
             "assumptions and underivable numbers as defects, and makes the "
             "Critic state the strongest counter-argument before it may "
             "approve. Longer debates, and NOT free — measure the cost with "
             "`./dev.sh calibrate --strictness adversarial`.")
    if critic_strictness == "adversarial":
        st.caption(
            "⚠ Raised bar. Expect more rounds — and a higher chance a good "
            "answer is sent back, which is the price of the stricter gate.")

    # ---- Models ------------------------------------------------------
    ready_for_models = (not settings.requires_key) or bool(settings.api_key)
    models, thinking, disco_note = ([], set(), "") if not ready_for_models \
        else discover_models(settings)
    settings.reasoning_models = thinking

    with st.expander("Models", expanded=settings.local or not ready_for_models):
        if not ready_for_models:
            st.caption("Add an API key above to pick a model.")
            proposer_model = settings.proposer_model
            critic_model = settings.critic_model
            arbiter_model = settings.arbiter_model
        elif models:
            rc1, rc2 = st.columns([4, 1])
            rc1.caption(disco_note)
            # "↻" rather than "Refresh": the sidebar is ~300px and a 1/5-width
            # column wrapped that word onto its own three lines, one letter
            # cluster per line. A single glyph plus a tooltip says the same
            # thing without needing room.
            if rc2.button("↻", width="stretch", key="refresh_models",
                         help="Re-fetch the model list from this provider"):
                st.cache_data.clear()
                st.rerun()

            def _fmt(name: str) -> str:
                tags = []
                if name in thinking:
                    tags.append("thinks")
                if prov.is_free_model(name):
                    tags.append("free")
                return f"{name}  ·  {', '.join(tags)}" if tags else name

            # WHEN THE CONFIGURED DEFAULT NO LONGER EXISTS.
            #
            # Preset model ids go stale — OpenRouter had withdrawn both of
            # this provider's defaults by the time a live key was pointed at
            # it. The old behaviour appended the dead id to the list and
            # selected it, so the picker showed a model that could not run and
            # the failure arrived as a 404 in round one.
            #
            # Substitute a live model instead, preferring another entry from
            # the curated list before falling back to whatever ranked first,
            # and SAY that a substitution happened. Silently choosing a
            # different model than the one configured is how you end up
            # debugging output from a model you did not know you were using.
            substituted: list[str] = []

            def _resolve(current: str, ranked: list) -> str:
                if current in ranked:
                    return current
                for candidate in prov.known_models(provider):
                    if candidate in ranked:
                        substituted.append(f"{current} → {candidate}")
                        return candidate
                substituted.append(f"{current} → {ranked[0]}")
                return ranked[0]

            ranked_p = catalog.rank_models(models, prefer=settings.proposer_model)
            ranked_c = catalog.rank_models(models, prefer=settings.critic_model)
            default_p = _resolve(settings.proposer_model, ranked_p)
            default_c = _resolve(settings.critic_model, ranked_c)

            # The widget key is SCOPED TO THE PROVIDER. With a bare "m_p",
            # Streamlit keeps the stored selection across a provider switch,
            # and a value held in session_state overrides `index=` — so
            # choosing a model on one provider silently pinned an unrelated
            # one on the next, ignoring that provider's defaults entirely.
            def _pick(label, default, ranked, key):
                return st.selectbox(label, ranked, index=ranked.index(default),
                                    format_func=_fmt, key=f"{key}::{provider}")

            proposer_model = _pick("Proposer", default_p, ranked_p, "m_p")
            critic_model = _pick("Critic", default_c, ranked_c, "m_c")
            arb_opts = ["(reuse critic)"] + catalog.rank_models(
                models, prefer=settings.arbiter_model)
            arbiter_model = st.selectbox(
                "Arbiter", arb_opts,
                index=arb_opts.index(settings.arbiter_model)
                if settings.arbiter_model in arb_opts else 0,
                format_func=lambda n: n if n.startswith("(") else _fmt(n),
                key=f"m_a::{provider}")
            arbiter_model = "" if arbiter_model.startswith("(") else arbiter_model

            if substituted:
                st.caption(
                    "⚠ This provider no longer serves the configured default, "
                    "so a live model was substituted: "
                    + "; ".join(f"`{s}`" for s in substituted))
        else:
            if disco_note:
                st.caption(disco_note)
            proposer_model = st.text_input("Proposer", settings.proposer_model)
            critic_model = st.text_input("Critic", settings.critic_model)
            arbiter_model = st.text_input(
                "Arbiter", settings.arbiter_model,
                placeholder="(blank = reuse critic model)")

        chosen = thinking & {proposer_model, critic_model,
                             arbiter_model or critic_model}
        if chosen:
            st.caption(
                f"⚠ **{', '.join(sorted(chosen))}** reasons before answering. "
                f"That thinking is billed against the token budget and never "
                f"appears in the reply, so the budget is raised automatically "
                f"(see Guards). Expect slower turns.")

        if proposer_model == critic_model and not settings.is_fake:
            st.caption("⚠ Both roles share one model. Fast, but self-critique "
                       "is weaker than cross-model critique.")

    with st.expander("Guards"):
        if settings.is_free:
            st.caption("These tokens are free (local, or a free-tier model) — "
                      "the wall-clock limit is what actually protects you.")
        max_seconds = st.number_input(
            "Wall-clock limit (s)", 0.0, 3600.0, float(settings.max_seconds),
            step=30.0,
            help="The guard that binds when tokens cost nothing: local "
                 "inference or a free-tier model. 0 disables.")
        budget = st.number_input(
            "Budget guard (USD)", 0.0, 20.0, float(settings.max_cost_usd),
            step=0.05,
            help=f"For paid tokens, at this provider's rate "
                 f"(${settings.cost_in_per_1m:.2f} in / "
                 f"${settings.cost_out_per_1m:.2f} out per 1M). "
                 f"Free tokens always report $0.00, so this never fires on "
                 f"them — that is what the wall-clock limit is for.")
        max_tokens = st.slider(
            "Max tokens per turn", 128, 4096, int(settings.max_tokens), step=64,
            help="Reply length per turn. A debate is 6+ turns, so this is "
                 "mostly a latency and cost dial.")
        reasoning_max_tokens = st.slider(
            "Max tokens for thinking models", 512, 16384,
            int(settings.reasoning_max_tokens), step=256,
            help="Separate dial: for a reasoning model this bounds THINKING "
                 "PLUS the reply. Too low and it can spend the whole budget "
                 "thinking and return an empty string.")

    save_transcript = st.checkbox("Save transcript to runs/", value=False)

    settings.max_rounds = max_rounds
    settings.max_cost_usd = budget
    settings.max_seconds = max_seconds
    settings.max_tokens = max_tokens
    settings.reasoning_max_tokens = reasoning_max_tokens
    settings.proposer_model = proposer_model
    settings.critic_model = critic_model
    settings.arbiter_model = arbiter_model
    settings.use_arbiter = use_arbiter
    settings.critic_strictness = critic_strictness

    st.divider()
    st.markdown("<div class='ag-label'>Endpoint state</div>", unsafe_allow_html=True)

    if settings.local:
        snap = hostinfo.snapshot(settings.base_url)
        if snap.gpu.available:
            st.progress(min(1.0, snap.gpu.mem_pct / 100),
                       text=f"VRAM {snap.gpu.mem_used_mb}/{snap.gpu.mem_total_mb} MB")
            st.caption(f"{snap.gpu.name} · {snap.gpu.utilisation_pct}% util · "
                      f"{snap.gpu.temperature_c}°C")
        else:
            st.caption(f"No GPU detected ({snap.gpu.detail})")
        if snap.ram_total_gb:
            st.caption(f"RAM {snap.ram_used_gb:.1f}/{snap.ram_total_gb:.1f} GB")
        for m in snap.loaded:
            full = m.gpu_fraction >= 0.999
            st.caption(f"{'●' if full else '○'} `{m.name}` {m.size_mb} MB · "
                      f"**{m.placement}** · ctx {m.context_length}")
        if any(m.gpu_fraction < 0.999 for m in snap.loaded):
            st.caption("○ partly on CPU — expect roughly 3× slower generation.")
        elif not snap.loaded and settings.local and snap.ollama_reachable:
            st.caption("No model resident — the first turn will load one.")
        if not settings.is_fake and not snap.ollama_reachable:
            st.error("Ollama is not reachable. Start it with `ollama serve`.")
    elif not settings.is_fake:
        st.markdown(
            f"<div class='ag-kv'><span class='k'>endpoint</span>"
            f"<span class='v'>{esc(settings.base_url)}</span></div>"
            f"<div class='ag-kv'><span class='k'>rate</span>"
            f"<span class='v'>${settings.cost_in_per_1m:.2f} / "
            f"${settings.cost_out_per_1m:.2f} per 1M</span></div>",
            unsafe_allow_html=True)

    if settings.is_fake:
        st.info("Offline mode. The graph runs for real; the text is "
                "synthetic — good for checking routing and the UI.")
    elif settings.requires_key and not settings.api_key:
        st.error(f"No API key. Paste one above, set `{preset['key_env']}` "
                 f"in `.env`, or switch provider to 'fake'.")

    with st.expander("All providers"):
        env_keys = {n for n, p in PROVIDERS.items()
                   if p.get("key_env") and os.getenv(p["key_env"])}
        stored_names = set(credentials.saved_providers(keystore_path()))
        st.markdown(provider_table(provider, env_keys, stored_names),
                    unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Masthead
# --------------------------------------------------------------------------

_model_note = f"{settings.proposer_model} vs {settings.critic_model}" \
    if settings.proposer_model else "no model selected"
st.markdown(
    "<div class='ag-head'>"
    "<span class='ag-brand'><svg class='ag-mark' viewBox='0 0 24 24' "
    "aria-hidden='true' fill='none'><path d='M12 2.8 20 7.4v9.2L12 "
    "21.2 4 16.6V7.4L12 2.8Z' stroke='currentColor' stroke-width='2' "
    "stroke-linejoin='round'/></svg><span>AEGIS</span></span>"
    "<span class='ag-sub'>a Proposer argues · a Critic attacks · "
    "an Arbiter rules on the deadlock</span>"
    f"<span class='ag-headmeta'>"
    f"<span><b>{esc(PROVIDERS[provider]['label'])}</b></span>"
    f"<span>{esc(_model_note)}</span>"
    f"</span></div>",
    unsafe_allow_html=True)

topic = st.text_area(
    "Topic", height=86, label_visibility="collapsed",
    placeholder="Should a two-person startup adopt Kubernetes?")

c_run, c_clear, c_note = st.columns([1.3, 1, 4.2])
run_clicked = c_run.button("Run debate", type="primary", width="stretch")
if c_clear.button("Clear", width="stretch"):
    st.session_state.pop("final_state", None)
    st.rerun()
with c_note:
    st.markdown(
        "<div class='ag-note' style='display:flex;align-items:center;height:2.35rem'>"
        + esc(
            f"cap {settings.max_rounds} rounds"
            + (f" / {settings.max_seconds:.0f}s" if settings.max_seconds else "")
            + (" · arbiter on" if use_arbiter else " · arbiter off")
            + (" · free" if settings.is_free else
               f" · guard ${settings.max_cost_usd:.2f}")
        )
        + "</div>",
        unsafe_allow_html=True)

can_run = bool(settings.proposer_model and settings.critic_model) and \
    (not settings.requires_key or bool(settings.api_key))

if "final_state" not in st.session_state:
    left_i, right_i = st.columns([1.6, 1], gap="large")
    with left_i:
        st.markdown("<div class='ag-label'>Live debate view</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<p class='ag-note'>The screen is arranged to show the machine as "
            "it runs: routing, evidence, contested points, and the exact "
            "prompt each turn saw. That makes failures inspectable instead of "
            "hiding them inside a finished paragraph.</p>", unsafe_allow_html=True)
        st.markdown("<div class='ag-label' style='margin-top:1rem'>The machine"
                    "</div>", unsafe_allow_html=True)
        st.graphviz_chart(
            graph_dot(arbiter_on=use_arbiter, research_on=settings.use_researcher),
            width="stretch")
        st.markdown(
            "<p class='ag-note'>Note that <b>Critic has three outgoing "
            "edges</b> — that is the whole control flow. The back-edge to "
            "Proposer is what makes this a graph rather than a pipeline. The "
            "Arbiter has <i>no</i> edge back into the loop, so adding a third "
            "agent did not add a way to fail to terminate.</p>",
            unsafe_allow_html=True)
    with right_i:
        st.markdown("<div class='ag-label'>Current setup</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='ag-kv'><span class='k'>provider</span>"
            f"<span class='v'>{esc(PROVIDERS[provider]['label'])}</span></div>"
            f"<div class='ag-kv'><span class='k'>proposer</span>"
            f"<span class='v'>{esc(settings.proposer_model)}</span></div>"
            f"<div class='ag-kv'><span class='k'>critic</span>"
            f"<span class='v'>{esc(settings.critic_model)}</span></div>"
            f"<div class='ag-kv'><span class='k'>rounds</span>"
            f"<span class='v'>{max_rounds}</span></div>"
            f"<div class='ag-kv'><span class='k'>arbiter</span>"
            f"<span class='v'>{'on' if use_arbiter else 'off'}</span></div>"
            f"<div class='ag-kv'><span class='k'>grounding</span>"
            f"<span class='v'>{'on' if settings.use_researcher else 'off'}</span></div>",
            unsafe_allow_html=True)
        st.markdown("<div class='ag-label' style='margin-top:1.1rem'>Three "
                    "outcomes, not equally trustworthy</div>", unsafe_allow_html=True)
        st.markdown(outcome_key(), unsafe_allow_html=True)
        st.markdown(
            "<p class='ag-note' style='margin-top:0.6rem'>Collapsing these "
            "into 'done' would launder a contested ruling as agreement — "
            "precisely the signal you most want to keep.</p>",
            unsafe_allow_html=True)


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

if run_clicked and not topic.strip():
    st.warning("Enter a question in the topic box, then run the debate.")
elif run_clicked and not can_run:
    if not settings.proposer_model or not settings.critic_model:
        st.warning("Pick a Proposer and Critic model in the sidebar first.")
    else:
        st.warning("Add an API key in the sidebar's Connection panel first.")

if run_clicked and topic.strip() and can_run:
    pipe = st.empty()
    pipe.markdown(rail("proposer", set(), use_arbiter, "starting…"),
                  unsafe_allow_html=True)

    head_l, head_r = st.columns(2, gap="medium")
    for col, name in ((head_l, "proposer"), (head_r, "critic")):
        style = agent_style(name)
        col.markdown(
            f"<div class='ag-thead'><span class='ag-who' "
            f"style='color:{style['c']}'>{style['label']}</span>"
            f"<span class='ag-tele'>{style['role']}</span></div>",
            unsafe_allow_html=True)

    arena = st.container()
    ledger_slot = st.empty()
    live = LiveDebate(arena)

    final_state, visited, started = None, set(), time.perf_counter()
    try:
        for node, state in stream_debate(topic, settings=settings,
                                         on_token=live):
            final_state = state
            if node.startswith("__") or not state.transcript:
                continue

            turn = state.transcript[-1]
            live.finalise(turn)
            visited.add(turn.agent)

            total = state.tokens_in + state.tokens_out
            pipe.markdown(
                rail(turn.agent, visited, use_arbiter,
                     f"round {state.round}/{state.max_rounds} · "
                     f"{total:,} tok · "
                     f"{time.perf_counter() - started:.0f}s wall"),
                unsafe_allow_html=True)
            if state.points:
                ledger_slot.markdown(ledger_rail(state.points),
                                     unsafe_allow_html=True)

        pipe.markdown(
            rail("", visited, use_arbiter,
                 f"finished in {time.perf_counter() - started:.0f}s"),
            unsafe_allow_html=True)
        st.session_state["final_state"] = final_state

    except Exception as exc:  # noqa: BLE001 - surface it, never swallow it
        st.error("The run failed. The partial transcript above is still valid.")
        st.exception(exc)


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

# ruff: noqa: SIM401 - do NOT rewrite this as st.session_state.get(...).
# Streamlit's AppTest exposes a SafeSessionState proxy whose __getattr__
# forwards into the state dict, so `.get("final_state")` makes it hunt for a
# session key literally named "get" and raise AttributeError. A proxy that
# looks like a dict is not a dict. Membership-test explicitly; the linter is
# right about the general pattern and wrong about this object.
final_state = (st.session_state["final_state"]
               if "final_state" in st.session_state else None)

if final_state is not None:
    if final_state.stop_reason == "approved":
        st.success(f"Critic approved after {final_state.round} round(s). "
                   f"Ledger: {final_state.points_summary()}.")
    elif final_state.stop_reason == "arbitrated":
        st.info(
            f"**Deadlocked after {final_state.round} rounds — resolved by the "
            f"Arbiter.** The Proposer and Critic did not agree, so a third "
            f"agent read the full exchange and ruled. Treat this as a "
            f"considered judgement on a contested question, not a consensus. "
            f"Ledger: {final_state.points_summary()}.")
    elif final_state.stop_reason == "max_rounds":
        st.warning(
            f"**Stopped at the {final_state.max_rounds}-round cap without "
            f"approval.** What follows is the last revision, not a reviewed "
            f"result — treat it with more suspicion than an approved answer.")
    elif final_state.error:
        st.error("The run ended with an error.")
        st.code(final_state.error, language="text")

    st.markdown(stat_strip([
        ("Rounds", f"{final_state.round}/{final_state.max_rounds}"),
        ("Turns", len(final_state.transcript)),
        ("Points", f"{len(final_state.points) - len(final_state.open_points)}"
                   f"/{len(final_state.points)} closed"),
        ("Tokens", f"{final_state.tokens_in + final_state.tokens_out:,}"),
        ("Inference", f"{final_state.elapsed_s:.0f}s"),
        ("Est. cost", f"${final_state.cost_usd:.4f}"),
    ]), unsafe_allow_html=True)

    # NINE PANELS, ONE VISIBLE. st.tabs renders every panel's content into
    # the DOM and hides the inactive ones, which on this screen meant the
    # full transcript, every prompt and a chart were built on every rerun
    # even when nobody was looking at them. A nav that renders only the
    # chosen panel is both faster and, with nine of them, easier to scan.
    _VIEWS = ["Answer", "Ledger", "Sources", "Path", "Decisions",
              "Transcript", "Prompts", "Telemetry", "History"]
    if HAVE_OPTION_MENU:
        _view = option_menu(
            None, _VIEWS,
            icons=["file-text", "list-check", "search", "diagram-3",
                   "signpost-split", "chat-left-text", "terminal",
                   "speedometer2", "clock-history"],
            orientation="horizontal", default_index=0,
            styles={
                # Matched to the stylesheet rather than the library's default
                # blue pills, so the nav belongs to the same instrument.
                "container": {"padding": "0", "background-color": "transparent",
                              "border-bottom": "1px solid rgba(128,128,128,0.30)"},
                "icon": {"font-size": "0.78rem"},
                "nav-link": {
                    "font-family": "IBM Plex Mono, monospace",
                    "font-size": "0.70rem", "font-weight": "500",
                    "letter-spacing": "0.08em", "text-transform": "uppercase",
                    "padding": "0.45rem 0.7rem", "margin": "0",
                    "border-radius": "0", "--hover-color": "rgba(128,128,128,0.10)",
                },
                "nav-link-selected": {
                    "background-color": "transparent", "color": "#4a86e8",
                    "font-weight": "600",
                    "border-bottom": "2px solid #4a86e8",
                },
            })
    else:
        _view = st.radio("View", _VIEWS, horizontal=True,
                         label_visibility="collapsed")

    if _view == "Answer":
        st.markdown(final_state.answer or "_No answer produced._")
        if final_state.ruling:
            with st.expander("Arbiter's full ruling — how the tie was broken"):
                st.markdown(final_state.ruling)
        st.download_button(
            "Download transcript (.md)", data=to_markdown(final_state),
            file_name=f"aegis-{final_state.run_id or 'run'}.md",
            mime="text/markdown")

    if _view == "Ledger":
        st.markdown(
            "<p class='ag-note'>Every claim the Critic raised and where it "
            "ended up. This is what makes the exchange a debate rather than a "
            "sequence of reviews: the Critic must close its own points before "
            "opening new ones, so APPROVE comes to mean <i>nothing I raised is "
            "still open</i> instead of <i>I failed to think of anything this "
            "time</i>. <b>withdrawn</b> means the Critic conceded — a debate "
            "where nothing is ever withdrawn is one where the rebuttals are "
            "decorative.</p>", unsafe_allow_html=True)
        if final_state.points:
            st.markdown(ledger_rail(final_state.points, "All points"),
                        unsafe_allow_html=True)
            for p in final_state.points:
                if p.proposer_note or p.critic_note:
                    with st.expander(f"{p.id} — {p.text[:80]}"):
                        if p.proposer_note:
                            st.markdown(f"**Proposer ({p.proposer_stance}):** "
                                        f"{p.proposer_note}")
                        if p.critic_note:
                            st.markdown(f"**Critic ({p.status}):** "
                                        f"{p.critic_note}")
        else:
            st.info("No points recorded. On the first round the Critic may "
                    "reply without a structured point list.")

    if _view == "Sources":
        st.markdown(
            "<p class='ag-note'>What the debate was grounded in. Retrieved "
            "ONCE before the argument starts, so both agents reason over the "
            "same fixed set of facts — if the sources changed between rounds, "
            "an improvement could be a better argument or just a better "
            "search, with no way to tell which.</p>", unsafe_allow_html=True)
        if final_state.sources:
            st.caption(f"query: `{final_state.research_query}`")
            for n, src in enumerate(final_state.sources, start=1):
                st.markdown(
                    f"<div class='ag-pt' style='--pc:#0e9384'>"
                    f"<span class='g'>S{n}</span><span class='t'>"
                    f"<b>{esc(src.title)}</b><br>{esc(src.snippet)}<br>"
                    f"<a href='{esc(src.url)}' target='_blank'>"
                    f"{esc(src.label)}</a></span></div>",
                    unsafe_allow_html=True)
        else:
            st.info("This debate ran **ungrounded** — no sources were "
                    "retrieved, so every factual claim in it is unverified. "
                    "Set `TINYFISH_API_KEY` in `.env` to enable retrieval.")

    if _view == "Path":
        st.markdown(
            "<p class='ag-note'>The path this run actually took. Solid, "
            "coloured edges fired; dashed ones did not. Both are read from "
            "the engine's own decision log — the UI is not permitted to "
            "re-derive the router's conditions, so what you see here is what "
            "the router recorded.</p>", unsafe_allow_html=True)
        st.graphviz_chart(
            graph_dot(arbiter_on=final_state.max_rounds is not None
                      and any(t.agent == "arbiter"
                              for t in final_state.transcript) or use_arbiter,
                      research_on=bool(final_state.sources),
                      visited={t.agent for t in final_state.transcript},
                      fired={d.rule for d in final_state.decisions}),
            width="stretch")
        st.caption(
            f"nodes run: {', '.join(sorted({t.agent for t in final_state.transcript}))}"
            f"  ·  rules fired: "
            f"{', '.join(d.rule for d in final_state.decisions) or 'none'}")

    if _view == "Decisions":
        st.markdown(
            "<p class='ag-note'>Every branch the router took, recorded by "
            "the engine at the moment it was taken. This is the answer to "
            "<i>why did it stop?</i> — read off the run itself, not "
            "reconstructed here.</p>", unsafe_allow_html=True)
        for d in final_state.decisions:
            icon, colour = RULE_STYLE.get(d.rule, ("•", "#8b949e"))
            dest = "END" if d.next_node == "__end__" else d.next_node
            st.markdown(
                f"<div class='ag-dec' style='--c:{colour}'>"
                f"<div class='h' style='color:{colour}'>{icon} "
                f"{esc(d.rule.upper())}</div>"
                f"<span class='ag-tele' style='opacity:.6'>after {d.at}, "
                f"round {d.round} → <b>{esc(dest)}</b></span><br>"
                f"{esc(d.reason)}</div>", unsafe_allow_html=True)
            if d.observed:
                st.markdown("<div class='ag-obs'>observed: " + " · ".join(
                    f"{k}={v}" for k, v in d.observed.items()) + "</div>",
                    unsafe_allow_html=True)
        if not final_state.decisions:
            st.info("No routing decisions recorded.")

    if _view == "Transcript":
        for turn in final_state.transcript:
            style = agent_style(turn.agent)
            v = f" · {turn.verdict}" if turn.verdict else ""
            st.markdown(
                f"<div class='ag-thead'><span class='ag-who' "
                f"style='color:{style['c']}'>ROUND {turn.round} · "
                f"{style['label']}{esc(v)}</span>"
                f"<span class='ag-tele'>{esc(turn.model)}</span></div>",
                unsafe_allow_html=True)
            if turn.reasoning:
                with st.expander(f"thinking ({len(turn.reasoning):,} chars)"):
                    st.markdown(f"<div class='ag-think'>{esc(turn.reasoning)}"
                                f"</div>", unsafe_allow_html=True)
            st.markdown(turn.content)
            st.divider()

    if _view == "Prompts":
        st.markdown(
            "<p class='ag-note'>Exactly what each agent was sent — captured "
            "on the turn, not rebuilt afterwards. Most 'the model is being "
            "stupid' bugs turn out to be 'the model was sent something other "
            "than what I assumed', and you cannot see that from the output "
            "alone.</p>", unsafe_allow_html=True)
        for i, turn in enumerate(final_state.transcript):
            with st.expander(f"Round {turn.round} · {turn.agent.upper()} — "
                             f"{turn.tokens_in} prompt tokens",
                             expanded=(i == 0)):
                st.caption("SYSTEM")
                st.code(turn.prompt_system or "(not recorded)", language="text")
                st.caption("USER")
                st.code(turn.prompt_user or "(not recorded)", language="text")

    if _view == "Telemetry":
        rows = [{
            "round": t.round, "agent": t.agent, "model": t.model,
            "verdict": t.verdict or "", "ttft_s": round(t.ttft_s, 2),
            "latency_s": round(t.latency_s, 2),
            "tok/s": round(t.tokens_per_s, 1),
            "tokens_in": t.tokens_in, "tokens_out": t.tokens_out,
            "thinking_chars": len(t.reasoning),
        } for t in final_state.transcript]
        if rows:
            if HAVE_AGGRID:
                # Sortable and filterable, which st.dataframe is not. The
                # question this table exists to answer — "which turn was the
                # slow one, and was it ttft or generation?" — is a sort, so
                # making the user eyeball twelve numbers was the wrong shape.
                import pandas as _pd

                frame_t = _pd.DataFrame(rows)
                gb = GridOptionsBuilder.from_dataframe(frame_t)
                gb.configure_default_column(
                    resizable=True, sortable=True, filterable=True,
                    cellStyle={"fontFamily": "IBM Plex Mono, monospace",
                               "fontSize": "12px"})
                gb.configure_column("model", width=230)
                gb.configure_column("agent", width=110)
                for numeric in ("ttft_s", "latency_s", "tok/s"):
                    gb.configure_column(numeric, type=["numericColumn"], width=110)
                AgGrid(frame_t, gridOptions=gb.build(),
                       fit_columns_on_grid_load=False,
                       theme="streamlit", height=min(400, 60 + 32 * len(rows)),
                       allow_unsafe_jscode=True)
            else:
                st.dataframe(rows, width="stretch", hide_index=True)
            # Altair rather than st.bar_chart, and not for looks. bar_chart
            # took a {label: value} dict, so it could show one number per turn
            # and had to smuggle the agent and round into a string key. The
            # useful shape is ttft and generation time as separate segments of
            # each bar, because that is the comparison the numbers exist for: a
            # tall dark segment means the model was loading or the prompt was
            # long, a tall light one means generation itself was slow, and the
            # two call for opposite fixes. Altair already ships with Streamlit,
            # so this adds no dependency.
            import altair as alt
            import pandas as pd

            frame = pd.DataFrame([
                {"turn": f"r{r['round']} {r['agent'][:4]}", "n": i,
                 "agent": r["agent"], "phase": phase,
                 "seconds": round(max(0.0, value), 2)}
                for i, r in enumerate(rows)
                for phase, value in (
                    ("time to first token", r["ttft_s"]),
                    ("generating", r["latency_s"] - r["ttft_s"]))
            ])
            st.altair_chart(
                alt.Chart(frame)
                .mark_bar()
                .encode(
                    x=alt.X("turn:N", sort=alt.SortField("n"), title=None,
                            axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("seconds:Q", title="seconds", stack="zero"),
                    color=alt.Color(
                        "phase:N", title=None,
                        scale=alt.Scale(
                            domain=["time to first token", "generating"],
                            range=["#8b949e", "#4a86e8"]),
                        legend=alt.Legend(orient="top")),
                    tooltip=["turn", "agent", "phase", "seconds"])
                .properties(height=210),
                width="stretch")
            st.caption(
                "**ttft** and **tok/s** are separated deliberately. A slow ttft "
                "means the model was loading or the prompt was long; a slow "
                "tok/s means generation itself is slow. They look identical in "
                "one latency number and call for opposite fixes.")
            if any(r["thinking_chars"] for r in rows):
                st.caption(
                    "**thinking_chars** never appears in the reply but is "
                    "charged against the same token budget — which is why "
                    "reasoning models get their own dial.")

    if _view == "History":
        runs_dir = Path(settings.transcript_dir)
        saved = sorted(runs_dir.glob("*.json"), reverse=True) \
            if runs_dir.is_dir() else []
        if saved:
            st.caption(f"{len(saved)} saved run(s) in `{runs_dir}/`.")
            for path in saved[:25]:
                st.markdown(f"`{path.name}`")
        else:
            st.info("No saved runs yet. Tick **Save transcript to runs/** in "
                    "the sidebar to keep them.")

    if save_transcript and final_state.run_id:
        saved_for = (st.session_state["_saved_run_id"]
                     if "_saved_run_id" in st.session_state else None)
        if saved_for != final_state.run_id:
            path = save_run(final_state, settings.redacted(), settings.transcript_dir)
            st.session_state["_saved_run_id"] = final_state.run_id
            st.caption(f"Saved to `{path}`")
