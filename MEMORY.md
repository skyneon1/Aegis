# Aegis — project memory

Context handoff for future sessions. Read this first before changing anything.

Last updated: 2026-08-25 · Version 0.4.0 · Status: working, 82/82 tests passing

**A real model has now run through this system.** That sentence was the headline gap in
every previous version of this file. Inference is local, on this machine, for free.

---

## Who this is for

ANIKET. Hardware: ASUS TUF F15, NVIDIA RTX 1650 with **4GB VRAM**. Python 3.12.3.
Working style: wants every design decision explained, not just working code. Wants
explicit permission asked before any file write or terminal command.

---

## What was decided, and why

Settled in session 1. Do not silently re-litigate; if reversing one, say which.

**Inference backend: OpenRouter** (~$5 credit), through an OpenAI-compatible
endpoint. Chosen over free-cloud-GPU-plus-tunnel because there is no session to
babysit. The 4GB VRAM makes local inference of any useful model impossible — 7B
weights alone need ~4.5GB before any context. Arithmetic, not a tuning problem.
Orchestration runs local, inference goes remote.

**Agent roles: Proposer vs Critic** (adversarial debate), not Planner/Coder. It
exercises the whole engine — loops, routing, termination, cost accounting —
without needing sandboxes, diffs, or git. The code pipeline reuses these rails
later; only roles and tools change.

**Interface: Streamlit** (`app.py`), with a CLI (`cli.py`) alongside, because the
CLI is what you actually debug and batch-evaluate with.

**Engine: hand-written `MiniGraph` (~70 lines)**, not LangGraph — for now.
LangGraph was not installed, and for three agents its extra features (durable
checkpoints, interrupts, parallel branches) are cost without benefit. MiniGraph
uses the *same contract*, so migration is mechanical. Now scheduled for 0.4.

**Session 2 — the Arbiter.** Added a third agent that runs *only* on deadlock.
Rationale: v0.1's cap produced the system's weakest output (an unreviewed draft)
at exactly the moment the question proved hardest. The Arbiter reads the full
exchange and rules. Terminal by construction — no edge back into the loop.

**Session 3 — the eval harness (Phase 0.3).** Added `aegis/evaluation.py` +
`evaluate.py`. It is a thin CONSUMER of `run_debate` — every case gets its own
fresh `DebateState` and fresh LLM client, exactly like a standalone CLI run.
Deliberate, not unoptimised: sharing one LLM/graph across cases would make
results depend on case order (FakeLLM's heuristic mode counts calls on the
*instance* — see llm.py — so a shared instance would let case 2 inherit case
1's call count). A separate frontend rather than a `cli.py` subcommand, because
`cli.py`'s exit code already means something specific (0/3/2 for one run) and
overloading it with a batch mode would make the same exit code mean two
different things depending on a flag. Measures PROCESS metrics only (outcome
distribution, rounds, cost) — deliberately no answer-quality grading yet; that
is a different kind of measurement (content vs. process) and bolting an
LLM-judge onto the counting logic would repeat the exact mistake
`tests/test_graph.py` warns against elsewhere in this file.

**Session 4 — local inference and observability (Phase 0.4).** Reversed the project's
founding assumption that local inference was impossible here. It was impossible for a *7B*
model; nobody had measured a 2B one. Benchmarked on this GTX 1650:

    gemma2:2b   1.9GB resident -> 100% GPU         ~50 tok/s   <- default
    qwen3.5:4b  3.7GB resident -> 41% CPU/59% GPU  ~15 tok/s   <- does not fit

Consequences, all of them forced by that table rather than chosen:

* **One model for both roles.** Cross-model critique is better in principle, but two models
  that cannot co-reside in 4GB evict each other — 7-12s of reload on every turn where the
  speaker changes, and a debate alternates every turn. Unaffordable, not merely pricier.
  Surfaced in the UI sidebar so it stays a known weakness. This ANSWERS the open question
  from session 1 for this hardware; it is not a rejection of the reasoning.
* **A wall-clock guard (`AEGIS_MAX_SECONDS`).** Local tokens are free, so the budget guard
  can never fire, so a stuck loop would hold the GPU forever at zero cost. Same
  load-bearing position as the budget guard (above the Arbiter branch). The general rule:
  a guard denominated in a currency the run does not spend is not a guard.
* **max_tokens 1200 -> 500 for local.** 1200 tokens is 24s of generation per turn at
  50 tok/s, times seven calls.
* **Token streaming.** Not cosmetic: a blocking client gives a minute of silence and then a
  wall of text, so a looping model is discovered at the end instead of at the second token.
* **A decision log.** `route_after_critic` now appends a `Decision` (rule, destination,
  reason, observed values). The UI needed to answer "why did it stop?", and the only
  alternative was re-implementing the router in the frontend — which invariant #1 forbids
  and which would drift.
* **Nodes declare their role** (`agent="critic"`) instead of the LLM inferring it from
  prompt text. Three historical bugs in this file came from that inference. The fix for a
  bad guess is to stop guessing.
* **`aegis/hostinfo.py`** — read-only GPU/VRAM/model-residency probing, never raises. On
  local hardware "is the model fully on the GPU?" is the difference between a 4s and a 25s
  turn, and it is invisible from inside the OpenAI dialect.

**Session 5 — the Critic is not a working gate, and reasoning-model support.**

*Critic calibration.* The perfectionism found in session 4 turned out to be a MODEL
limit, not a prompt bug, and it took four prompt revisions to establish that. Measured on
gemma2:2b against `aegis/calibration.py`:

    v1  approves 0/5 good answers        perfectionism
    v2  approves 8/8 debates, round 1    rubber stamp; missed 2/3 subtle flaws
    v3  0/5 good, 3/3 flawed             back to perfectionism
    v4  0/2 good, 6/6 bad                balanced accuracy exactly 50%

Never a middle. That pattern IS the finding: a model that swings wholesale with the lean
of the wording is not representing the distinction, and no further prompt edit conjures
the capability. Same prompt (fingerprint 32a24e869fc7) on qwen3.5:4b APPROVES the strong
answer that gemma2:2b rejects 5/5 — so the limitation is the 2B model.

Do not "fix" this by editing CRITIC_SYSTEM again without running
`./dev.sh calibrate` first. Four attempts are on record; a fifth done by feel is worse
than none, because it will look like progress.

*But qwen is not a drop-in Critic either.* On good-strong it APPROVES (reproduced twice,
257-263s). On good-thin it generated 3208 tokens of pure reasoning and stopped WITHOUT
answering - 888 tokens short of its budget, so not a room problem; it thought itself to a
standstill. Better judgement when it arrives; unreliable arrival on 4GB. Hence next-step
item 0(b): a small NON-thinking 3-4B model is the most promising untried option.

That failure also improved the diagnostic. `finish_reason` now distinguishes "ran out of
room" (raise the budget) from "gave up N tokens short" (a bigger budget will not help).
The first message said "spent all N of its budget" in both cases, advising a change to a
limit the model had never reached. A diagnostic that names the wrong fix costs more than
no diagnostic - and only real data exposed it.

*Why the metric is per-verdict recall.* `CalibrationReport.balanced_accuracy` averages
APPROVE-recall with REVISE-recall. Overall accuracy would score a stuck-on-REVISE critic
at 75% (six of eight cases expect REVISE). Per-TIER averaging — which this module's first
version used — still scored always-REVISE at 67% against always-APPROVE's 33%, while the
docstring claimed they came out level. They did not. Test a metric against known-
degenerate inputs before trusting it; both failure modes now score exactly 50%.

*Reasoning models.* qwen3.5:4b "not working" was a real and instructive bug. It is a
thinking model: it reasons on a separate channel and that reasoning is billed against
`max_tokens` while never appearing in the reply. With gemma2's 500-token budget it spent
the lot thinking and returned an EMPTY STRING while reporting 300 generated tokens.
Nothing raised. The empty answer flowed into the Critic, which could not approve it, so
the debate ran the cap and reported a confident "contested" result about nothing.

Four separate fixes, and the last is the important one:
  * detect via Ollama `/api/show` capabilities (`thinking`) — never guess from the name;
  * `reasoning_max_tokens` (4096) — one number cannot mean both "answer length" and
    "thinking plus answer length";
  * `reasoning_num_ctx` (8192) — num_ctx bounds prompt PLUS generation, so a 4096 window
    with a 4096 budget cannot work. Sizing context from reply length is the trap;
  * `EmptyCompletionError` — returning "" WAS the bug. The budget was only the trigger.

Reasoning is kept out of `content` deliberately: the verdict is parsed from the first line
of the reply, so prepended musing would turn a rendering choice into a routing change.

**Session 6 — the point ledger, and the UI as a debate.**

*The debate was not a debate.* The Critic saw the topic and the current answer and
nothing else - not even its own previous objections - so each round it answered
"what is wrong with this?" (unbounded) instead of "were my objections answered?"
(bounded, and able to converge). Convergence was impossible by construction, and
the round cap was quietly doing the work agreement should have done. The Proposer
could also dispute a point and never be answered: two monologues, not an argument.

`state.Point` fixes that. A point is raised, answered FIXED or DISPUTED by the
Proposer, then RULED on by the Critic as RESOLVED / WITHDRAWN / OPEN. The Critic
must close its own points before opening new ones, so APPROVE now means "nothing
I raised is still open".

Three properties, each of which was a bug before it was a rule:
  * **Only the Critic closes a point.** DISPUTED sets `disputed`, never
    `resolved` - otherwise the party under review decides when review ends.
  * **Unmentioned points stay open.** Same fail-safe direction as parse_verdict.
  * **Re-raised points are dropped in the engine.** Observed live: the Critic
    closed P2 RESOLVED and raised near-identical P3 in the same reply. Asking the
    prompt not to is not enough; a stateless critic re-derives the same objection
    from the same answer.

*A metric bug worth remembering.* Dedupe first used `intersection / min(len)` at
0.7; two real paraphrases scored 0.68 and slipped through. The fix was NOT a lower
bar - `intersection / min` is inflated whenever one text is short and mostly
contained in a longer one, so lowering it starts merging genuinely distinct
objections about the same subject. Jaccard (intersection / union) is symmetric,
scores that pair 0.50, and containment survives only as a separate strict test.
Second time this session a similarity/scoring metric was wrong in the direction
that flattered it; test metrics against known-degenerate inputs.

*Routing still branches on the verdict alone.* The ledger informs, never decides.
An APPROVE with the Critic's own points still open is recorded as a contradiction
in the decision log rather than smoothed over - it is a measurable symptom of a
critic not tracking itself.

*A real parser bug the tests caught.* `VERDICT: APPROVE / REASONS: - Looks fine
now.` had its justification parsed as a fresh objection, so approving a debate
INVENTED a point. REASONS bullets mean opposite things depending on the verdict -
grounds for rejecting, or for accepting - so the same text cannot be parsed the
same way in both. `parse_new_points(strict=True)` on APPROVE.

*UI rebuilt as a per-round grid.* Proposer left, Critic right, one row per round,
Arbiter spanning both. A single column hid that the two are adversaries. Also: an
empty state that states the three outcomes BEFORE there is an answer to be
persuaded by; `esc()` on everything interpolated into HTML (an answer containing
a bare `<` silently swallowed the rest of a card); Streamlit chrome hidden and top
padding raised to 3.4rem, because a hidden header still occupies layout space and
was clipping the masthead.

**Session 7 — retrieval (TinyFish), and the same .env leak twice.**

*The keys provided are TOOLS, not inference providers.* TinyFish = Search / Fetch /
Browser / Agent (header `X-API-Key`, not Bearer). Apify = scraping/automation.
Neither speaks chat-completions, so neither replaces AEGIS_PROVIDER; Ollama still
does inference. Do not re-litigate this as "wire them in as a model".

*What they fixed.* The Critic's only possible objection to a fabricated statistic
was that it was UNSUPPORTED - a complaint about absence, because absence was all it
could detect. It could never say "wrong, here is the real figure". `aegis/tools.py`
+ a `researcher` node put evidence on the state and in both prompts.

Forced design decisions (each with a reason, not a preference):
  * retrieved ONCE before the argument - per-round retrieval makes a round-over-round
    improvement unattributable between argument and search;
  * SNIPPETS not full pages - 4096-token local context; one article would eat the
    whole budget. TinyFish Fetch deliberately unused;
  * NO model call in the researcher - a model writing a query for its own topic just
    paraphrases it. This is where agent systems accumulate ceremony;
  * numbered `[S1]` handles, not URLs - a small model mangles long URLs, and a
    mangled citation is worse than none because it still looks checkable;
  * Apify NOT wired - same capability, async run-and-poll instead of one GET, and
    TinyFish Search is free. Seam is `SearchProvider` if ever wanted.

*THE SAME MISTAKE, SECOND TIME.* Adding TINYFISH_API_KEY to .env auto-enabled
research for EVERY provider including `fake`, so the hermetic suite began making
real HTTP searches: 5s -> 34s, three failures. Identical in shape to session 4's
model-name leak. The rule is now general and belongs in invariants: **nothing in
.env may change what the fake provider does.** Its entire purpose is a path that
behaves the same on a bare clone.

*Ungrounded is reported, never implied.* A failed search records "No evidence
retrieved ... proceeds ungrounded" in the transcript, because a run with no
citations must not look identical to a run where retrieval broke.

*What did not improve.* The Critic used ZERO citations in every round of a live
grounded debate. It is the component that most needed them. Grounding raised the
ceiling; gemma2:2b still does not reach it. Consistent with next-step item 0.

**Session 8 — the Critic gate is FIXED, and the keys are gone.**

*Credentials removed at the user's request* (shared by accident). TINYFISH_API_KEY
blanked in .env; whole tree grepped clean of both keys and of the proxy password
Apify's whoami leaked. Retrieval is keyed on the credential, so it is now dormant
and the system behaves exactly as it did before grounding existed - the code and
its 17 hermetic tests remain, inert. The Apify key was never persisted anywhere
(it was never wired in). ADVISE ROTATION: both were pasted into a chat transcript.

*llama3.2:3b resolves next-step item 0(b).* Calibration on an IDENTICAL prompt
(b3d3171ca0d7), so the delta is the model:

    gemma2:2b    APPROVE 0/2  REVISE 6/6  balanced  50%  PERFECTIONISM
    llama3.2:3b  APPROVE 2/2  REVISE 6/6  balanced 100%  CALIBRATED

~46 tok/s, 2436 MB, 100% GPU, and NON-thinking (['completion','tools']) so no
reasoning-budget machinery applies. Now the default for BOTH roles.

Eval outcome distribution, finally informative: approved 1/8 -> 3/8, avg rounds
2.75 -> 2.38, approvals on rounds 1-2, different topics getting different outcomes.
NOTE the attribution: that eval also changed the prompt, and `--compare` correctly
flags both as CHANGED. Only the calibration table isolates the model.

*A REAL BUG the calibration data exposed.* Both models emitted "P1: RESOLVED /
P2: OPEN" on FIRST-round answers with no prior points - rulings on objections
nobody made - and the invented "P2: OPEN" drove the verdict to REVISE. The engine
was robust (unknown ids ignored, P-lines never become new points) so the ledger
stayed clean; the verdict did not. Cause: the RULINGS format slot was shown
unconditionally. **A format slot is an instruction.** Now appended only when prior
points exist (CRITIC_RULINGS_CLAUSE), and that fix took llama from 75% to 100%.
Third instance this project of "showing a model a shape makes it produce that
shape" - see also the example-leakage bug in session 6.

*One model both roles, measured.* gemma2:2b + llama3.2:3b = 4336 MB vs 4096 MB, so
they evict each other: 5.6-6.6s per call alternating vs 0.4s staying put, i.e.
~5.5s reload per speaker change (~38s per 7-turn debate). Mixed config documented
in .env for anyone who wants cross-model critique at that price.

*The xfail was promoted.* `test_approves_a_good_answer` was
xfail(strict=False) precisely so a fix would surface as XPASS rather than silence.
It XPASSed, so it is now a plain assertion. An xfail is a placeholder for a known
defect, not a permanent excuse.

**Session 9 — code review, and two bugs found by LOOKING at the UI.**

*`dev.sh ui-bg` reported success while serving stale code.* It fire-and-forgot
`nohup streamlit &` and printed "UI running (pid N)" with the shell's pid,
never checking the port bound. A server from an earlier session still held
8899, the new process died instantly, and the browser showed OLD code with an
OLD .env - the sidebar advertised gemma2:2b hours after .env said llama3.2:3b.
Fixed: detect a busy port, kill and wait, then POLL until it answers or fail
with the log tail. Also `./dev.sh restart`. Note WHY a restart is needed at
all: config.py calls load_dotenv() at import, so a live server holds whatever
.env said when it started.

*The Critic filed the prompt's own checklist as ledger points.* Seen in the UI:
3 of 4 points were verbatim quotes of the pre-approval checks ("Is every claim
presented as fact actually true? ..."). Inflates the ledger, leaves fake points
OPEN, and makes an approval look self-contradictory. THIRD instance of "show a
model a shape and it produces that shape" (after the copied format example and
the invented RULINGS). Fixed in the prompt (checks are now prose, explicitly
"to RUN, not text to repeat") plus an engine backstop.

*That backstop was itself wrong first.* At intersection/min 0.6 it filtered a
GENUINE objection, because the prompt names a defect category and a real point
instantiates it - they share vocabulary by design. Silently dropping a real
finding is much worse than keeping a junk one, so the filter is now Jaccard at
0.8, near-verbatim only. FOURTH time an intersection/min similarity measure
has been wrong here, always in the flattering direction. Use Jaccard.

*Review findings.* ruff over the tree: ~30 issues, 4 real (dead import in
cli.py, `zip()` that could silently drop a metric -> strict=True, two
raise-from). One suggestion REFUSED: SIM401 would rewrite the session_state
lookup as `.get()` and reintroduce the documented SafeSessionState bug - noqa
with the reason in app.py. Dead code removed (an unused OUTCOME table, three
orphaned CSS classes from the deleted stances_block).

*UI, using only what already ships.* No third-party component library: altair
and pandas are already Streamlit dependencies and st.graphviz_chart is built
in. Added a real DOT graph of the machine (the point being that CRITIC has
THREE outgoing edges and the condition on each - invisible in a row of pills),
a Path tab highlighting the route the run actually took from the decision log,
and an altair telemetry chart stacking ttft against generation time because
those two diagnose opposite faults.

**Environment: `.venv`, not `--break-system-packages`.** Ubuntu PEP 668 blocks
system pip. The venv is the correct fix; breaking system packages risks the
Python that OS tooling depends on.

---

## Architecture in one paragraph

`DebateState` (aegis/state.py) is the single source of truth. Agent nodes are
plain functions `(state) -> dict of updates`; they never mutate state directly.
`MiniGraph` merges those updates and follows edges. The only decision point is
`route_after_critic()` in aegis/graph.py — read that function to answer "why did
the debate stop?". Proposer → Critic cycles until APPROVE, then deadlock routes
to the Arbiter, and everything terminates at END.

---

## Invariants — do not break these

1. **`cli.py` and `app.py` contain zero orchestration logic.** An
   `if verdict == ...` in a frontend means the layering has broken.
2. **`aegis/config.py` is the only module that reads `os.environ`.**
3. **`aegis/llm.py` is the only module that makes an INFERENCE call.**
   Narrowed in session 4, deliberately and not quietly: `aegis/hostinfo.py`
   now makes read-only HTTP calls to a local Ollama (`/api/ps`, `/api/tags`).
   That does break the original wording. The invariant's *purpose* was to keep
   one swappable model boundary so providers are a config change and tests need
   no network — and host introspection generates no tokens, influences no
   agent's output, and is confined to localhost. What the rule must still
   forbid: any module other than `llm.py` asking a model for text. If a
   frontend ever needs a probe, it goes through `hostinfo`, not `urllib`.
4. **Routing branches on the parsed `verdict` enum, never on prose.**
5. **`parse_verdict` fails SAFE (→ REVISE); `extract_final_answer` fails OPEN
   (→ full text).** Different directions on purpose: the first drives control
   flow, the second drives presentation. Pick the failure you can live with.
6. **The budget guard is checked BEFORE the arbiter branch.** A guard the
   escalation path can bypass is not a guard. This ordering is load-bearing and
   has a test (`test_budget_guard_beats_the_arbiter_branch`).
7. **The Arbiter is terminal.** No edge back into the loop. A judge you can
   appeal to repeatedly is just another debater, and reintroduces the
   non-termination the whole design avoids.
8. **Nodes return update dicts.** Preserves LangGraph compatibility.
9. **The fake provider must always work with no API key.** A fresh clone runs
   before it asks for money.
10. **The three outcomes stay visually distinct** in CLI, exit code, and UI.
    Collapsing `arbitrated` into `approved` launders a contested ruling as a
    consensus.
11. **Both resource guards are checked BEFORE the arbiter branch.** Budget *and*
    wall-clock. Escalation costs a model call, which costs money and time.
    Tests: `test_budget_guard_beats_the_arbiter_branch`,
    `test_time_guard_beats_the_arbiter_branch`.
12. **Cost is a property of the PROVIDER, not a constant.** Local inference
    reports `$0.00`. Billing phantom cents for tokens generated on your own GPU
    corrupts the budget guard, which is a safety mechanism.
13. **The `fake` provider ignores model env vars.** A `.env` naming real models
    made offline runs record `gemma2:2b` in their transcripts — a model that
    generated none of that output. The offline path must stay reproducible from
    a bare clone, so it cannot depend on `.env`.
14. **Agent nodes declare their own role to the LLM.** Never infer identity from
    prompt text. Three bugs, one root cause.
15. **The router records every decision it makes.** The UI reads that log; it
    never re-derives the routing conditions.
16. **Every UI test goes through `_fresh()`, which pins the provider.** Two tests
    once built `AppTest` directly, and the moment a local Ollama became the
    configured default they started driving real inference — 20s runtime and a
    genuinely flaky assertion. A UI test that reaches a GPU is not a UI test.
17. **`aegis/hostinfo.py` never raises.** Observability that can crash the thing
    it observes is a liability. Every probe degrades to "unknown".
18. **An empty model response RAISES.** `EmptyCompletionError`, never `""`. An
    empty answer propagates into a plausible-looking contested verdict about
    nothing, and the system goes on looking healthy. Loud beats silent.
19. **Reasoning text never enters `content`.** Two channels. The verdict is
    parsed from line one; prepended thinking would change routing.
20. **`max_tokens` and `num_ctx` are per-model, via `budget_for()` /
    `ctx_for()`.** A thinking model needs both raised. `num_ctx` bounds prompt
    PLUS generation.
21. **No sentinel inside the valid range of what it guards.** `ttft` used `0.0`
    for "not measured", and an instant response made the generation window
    1 microsecond and reported 190,000,000 tok/s. Use `None`.
22. **Prompt changes need `./dev.sh calibrate` first.** Four CRITIC_SYSTEM
    revisions are on record. A fifth argued from a feeling is not an argument.
29. **NOTHING in `.env` may change what the `fake` provider does.** Not model
    names (session 4), not retrieval keys (session 7). Both leaked and both
    broke the hermetic suite. The offline path must behave identically on a
    bare clone; tests that want a non-default fake setup set it on the
    Settings object explicitly, where it is visible at the call site.
30. **Retrieval never raises and is never mandatory.** A dead search engine
    degrades a debate; it must not end one. A failed search is RECORDED as
    ungrounded, so no-citations and retrieval-broke are distinguishable.
31. **Grounding is keyed on the credential, not a flag.** Research without a
    key would cite FakeSearch's invented sources - worse than no grounding.
24. **Only the Critic may close a point.** The Proposer's DISPUTED sets
    `disputed`, never `resolved`. A point the Critic did not rule on stays
    open - a parse slip must never resolve an objection.
25. **The ledger informs routing; it never decides it.** Branch on the parsed
    verdict only. Record an APPROVE-with-open-points as a contradiction.
26. **The UI never parses model output.** It renders fields the engine already
    stored (`Point`, `Decision`, `Turn`). If a panel needs something parsed,
    the parser goes in `agents.py` and the result on the state.
27. **Escape model output before interpolating it into HTML** (`esc()` in
    app.py). An answer containing a bare `<` swallows the rest of the card.
28. **Widget ORDER in app.py is load-bearing.** tests/test_app.py addresses
    widgets positionally; inserting a slider above "Max rounds" silently
    retargets those tests rather than failing them. Add new controls after
    the existing ones, or inside an expander.
23. **Errors are stored as `f"{type}: {msg}"`, never `repr(exc)`.** `repr`
    escapes newlines, so a multi-line actionable diagnostic arrives as one
    unreadable line. An error message is part of the interface.

---

## Bugs already found and fixed — do not reintroduce

**Markdown-decorated verdicts.** `**VERDICT:** APPROVE` did not match the regex,
fell through to the safe `REVISE` default, and would have presented as "the
critic never approves anything" — a reasoning bug in appearance, a regex bug in
reality. Fix: treat `* _ ` # : -` as ignorable noise. Lesson: write the parser
defensively even when the prompt is strict. Prompts are requests, not guarantees.

**Identity inferred from an incidental substring.** `FakeLLM` checked whether
`"critic"` appeared in the system prompt — but the *Proposer's* prompt ends with
"A Critic will attack your answer." The fake proposer began emitting verdicts.
Fix: match the opening declaration (`YOU ARE THE CRITIC`) or the model name.

**Inferring state that could be counted.** The fake critic searched its own
prompt for `"REVISION"` — text only the Proposer receives — so it revised
forever. Fix: explicit call counter. Count what you control; don't infer what you
can measure.

**Role-check ordering (session 2).** The Arbiter reuses the *critic's model name*
by default, so `"critic" in model` matched and the fake Arbiter would have
emitted a verdict instead of a ruling — the same misidentification class arriving
by a new route. Fix: `_is_arbiter` is checked FIRST. Lesson: overlapping role
signals need an explicit priority order, most specific first.

**`SafeSessionState` is not a dict (test bug).** `at.session_state.get("x")` made
Streamlit hunt for a session key named `"get"` and raise `AttributeError`. Use
`"x" in at.session_state`. A proxy that looks like a dict is not a dict.

**Session 4 bugs — all from a substrate change, not from new features.**

*The constructor demanded a key that the provider does not use.* `OpenAICompatibleLLM`
raised whenever `api_key` was empty. Correct for OpenRouter, wrong for a local Ollama —
and because it raised in `__init__`, the entire local path was dead before a single token.
The assumption "all providers need a key" was invisible because the only two providers
written first both satisfied it. Fix: providers declare `requires_key`. Lesson: when two
examples share an incidental property, that property gets silently promoted to a rule.

*The SDK demanded one too.* Even after the above, `openai` refuses to construct without
some `api_key`, and its error names `OPENAI_API_KEY` — actively misleading when you are
pointed at localhost. Fix: pass a placeholder for keyless providers, so a transport detail
stops surfacing as a fake auth failure.

*Local inference was billed.* `_estimate_cost` charged $0.20/$0.60 per 1M unconditionally.
Not a rounding error: `max_cost_usd` is a guard the router checks before escalating, so
phantom cents would fire it on money nobody spent. Fix: cost is a property of the provider.
Lesson: a wrong number that feeds a safety mechanism is a safety bug, not a reporting bug.

*The default provider sniffed the environment.* The UI first defaulted to Ollama if it was
listening. That made the same click behave differently on different days, put a 2s network
probe on the startup path of a framework that reruns on every interaction, and made a UI
test execute real inference until it timed out. Fix: the default comes from config.
Lesson: environment *sensing* and environment *configuration* are different things, and
only the second belongs on a default.

*Two UI tests bypassed the provider pin.* They built `AppTest` directly, so once a local
Ollama became the configured default they drove real inference — 20s runtime and a flaky
assertion, because a real model does not deadlock on cue. Fix: every case goes through
`_fresh()`. Also rewrote `test_defaults_to_fake_provider` to assert the *invariant* (the
default requires no credential) instead of the literal string "fake" — it had been pinning
an implementation detail as a proxy for a rule, and failed when the detail changed while
the rule still held.

*A `.env` leaked into the offline path.* `AEGIS_PROPOSER_MODEL` overrides every provider,
so offline runs began recording `gemma2:2b` in transcripts — a model that generated none of
that output. It also silently disabled a FakeLLM role heuristic keyed on the model name.
Fix: the fake provider ignores model env vars. Lesson: a transcript field that is a guess
is worse than an absent one, because it will be believed.

*A test asserted something physically impossible.* The first draft of
`test_timeout_decision...` set `max_seconds` to a microsecond against a zero-latency fake
model. `elapsed_s` accumulates from real turn latencies, so no amount of tightening could
ever trip it. Fix: give the fake model a latency. Lesson: when a guard test passes
suspiciously easily or fails inexplicably, check that the quantity it bounds can actually
grow.

All have regression tests.

---

## Current state

```
aegis/{state,config,llm,agents,graph,transcript,evaluation,hostinfo,
       calibration,__init__}.py
cli.py  app.py  evaluate.py  calibrate.py  dev.sh  topics.txt
tests/{test_graph,test_app,test_evaluation,test_local,test_reasoning,
       test_critic_calibration}.py
pytest.ini  requirements.txt  .env  .env.example  .gitignore
README.md  MEMORY.md
Aegis_Agentic_Orchestration_Platform_Report.pdf   (original strategy doc, 19pp)
```

`./dev.sh` is the entry point for everything: `ui` (default), `doctor`, `cli`,
`test`, `test-live`, `eval`, `calibrate`, `stop`. It preflights the venv, Ollama,
and the pulled models, because those three fail differently and only one fails
loudly.

`.env` is gitignored and configured for local inference on this machine.

Verified: **144/144 hermetic tests pass in ~6s**, and **5/5 live calibration tests pass with no xfails** (32 engine + 9 Streamlit
`AppTest` + 18 eval harness + 23 local-inference + 16 reasoning), plus 4 `live`
calibration tests deselected by default.

`pytest.ini` sets `addopts = -m "not live"`. That is deliberate: live tests are
DESELECTED rather than skipped, so having Ollama running never silently turns a
5-second suite into a 40-second one. Two UI tests already drifted that way once,
picking up real inference the moment a local model became the configured default.
Run them with `./dev.sh test-live`. · Streamlit executes cleanly and serves on :8899 · CLI exit codes 0 /
3 / 2 for approved / arbitrated / unreviewed · `evaluate.py` exit code 0 / 1
for no-errors / some-case-raised · transcripts save as JSON + Markdown with
the API key redacted · eval reports save to `runs/evals/<eval_id>/` as
report.json + report.md · `openai` 3.3.1 client constructs against the
OpenRouter base URL.

**First real measurements (2026-08-25, gemma2:2b local).**

A 3-round debate with an Arbiter ruling: **~45s wall, 7 turns, ~5,000 tokens, $0.00.**
Generation held ~51 tok/s on every turn, which is how we know the model stayed fully
GPU-resident and never got swapped. TTFT climbed 1.2s -> 5.5s across the run as prompts
grew 161 -> 1,488 tokens — prompt processing, exactly what TTFT is meant to expose.

**The first honest quality signal, and it is not good.** `evaluate.py --topics topics.txt
--rounds 3` over 2 topics: **arbitrated 2/2 (100%), avg rounds 3.0.** The Critic approved
**nothing**, ever. Session 3 predicted two failure modes to watch for — rubber-stamping
(approves everything on round 1) and perfectionism (approves nothing). This is
unambiguously the second.

Note what that costs: if the Critic never approves, the round cap always fires, the Arbiter
always runs, and every answer is delivered as "contested" — so the outcome signal the whole
three-outcome design exists to carry is currently constant, and therefore carries no
information. Report saved at `runs/evals/20260825-161818-662ad3/`.

Do not fix this by weakening the Critic prompt on instinct. The `--compare` machinery
exists precisely so that a prompt change can be justified with numbers; this run is the
"before" baseline. Two candidate causes, worth separating before touching anything:
(a) `CRITIC_SYSTEM` genuinely sets an unreachable bar ("no material gap"), or (b) a 2B model
is simply weak at judging "good enough" and defaults to finding fault. These call for
different fixes, and the second would also predict that the same prompt behaves better on a
larger model — which is a testable claim, not a matter of taste.

**Still not done:** answer quality beyond the outcome distribution is unmeasured (no
LLM-judge). LangGraph is not installed. No OpenRouter key is set, so the local-vs-cloud
comparison that would separate cause (a) from (b) has not been run.

---

## Next steps, in order

0. ~~**The Critic gate is not calibrated**~~ **RESOLVED 2026-08-28** by
   llama3.2:3b (100% balanced, CALIBRATED) plus the conditional-RULINGS fix.
   Kept for the reasoning, which still applies to any future model swap: Options, in order of cost:
   (a) use qwen3.5:4b as the Critic only — it discriminates, at ~260s/turn;
   (b) pull a small non-thinking 3-4B model (Llama 3.2 3B, Phi-3 mini) and
   calibrate it — likely the best speed/quality point, and ≤1.8GB would even
   co-reside with gemma2:2b for genuine cross-model critique;
   (c) accept gemma2:2b's perfectionism and read `arbitrated` as "unreviewed".
   Do NOT reach for a fifth prompt revision first. Run `./dev.sh calibrate`.

1. **[SUPERSEDED by 0 — kept for the reasoning] Fix the Critic's perfectionism, with numbers.** The baseline exists
   (`runs/evals/20260825-161818-662ad3/`, arbitrated 2/2). Reword `CRITIC_SYSTEM`
   to make APPROVE actually reachable — the current bar ("no material gap") reads
   as unreachable — then run the same `topics.txt` through both and
   `evaluate.py --compare` the reports. Read the per-case **flips** first; that is
   the signal a change altered a real decision rather than nudging an average.
   This is the first prompt change in the project that can be justified with data.
2. **Widen `topics.txt` before trusting any of it.** Two cases is an anecdote.
   Ten short topics still runs in ~10 minutes locally and costs nothing, which is
   the actual dividend of moving inference on-machine: iteration is now free, so
   there is no longer an excuse for a 2-case eval.
3. **Check for length inflation.** Still unmeasured, and still the classic failure
   of this pattern. `tokens_out` per round is in every transcript; if revisions get
   longer without getting better, the loop is laundering verbosity as improvement.
4. Optional: pull a second model of **≤1.8GB** so it co-resides with gemma2:2b in
   4GB, and get real cross-model critique with no swap penalty. This is the one
   cheap way to test whether the perfectionism is the prompt or the model.
5. Phase 0.5 — LangGraph migration, for checkpoints and human-in-the-loop.
6. Phase 0.6 — the code pipeline (report's Sprints 1–5).

---

## Open questions not yet resolved

- Should Proposer and Critic use *different model families*? **Answered for this
  hardware, and not the way session 1 hoped.** Same-model self-critique is still
  the weaker pattern, but on 4GB two models cannot co-reside, and evicting one per
  turn costs 7-12s on a 6s turn. So: one model locally, cross-model on a remote
  endpoint, and a ≤1.8GB second model is the only local route to testing it. Note
  the observed behaviour is the *opposite* of the predicted weakness — a
  self-critiquing gemma2:2b approves nothing rather than rubber-stamping itself.
  Worth understanding before assuming the prediction was simply wrong.
- Should the Arbiter use a *stronger* model? It runs at most once, on the hardest
  questions, so it is the cheapest place to spend. `AEGIS_ARBITER_MODEL` exists;
  no data yet on whether it helps.
- Should the Proposer see the full transcript, or only its last answer plus the
  latest critique? Currently the latter, to protect context budget. May cause it
  to re-introduce problems fixed in earlier rounds — watch for this.
- The cost model in `llm.py` is a hardcoded blended estimate (`$0.20/$0.60` per
  1M) for REMOTE providers; local correctly reports $0.00. Still wrong for real
  accounting. Replace with per-model pricing when spend starts mattering.
- `elapsed_s` accumulates *inference* time, not wall-clock time. On local runs the
  two are within a per cent of each other, so the wall-clock guard is honest today.
  If orchestration ever gains real overhead (tool calls, retries, sandboxes), the
  guard would start under-counting and would need a monotonic clock instead.
- The UI's host panel probes `nvidia-smi` and `/api/ps` on every Streamlit rerun.
  Measured at ~25-45ms, so it is fine now; if it ever bites, cache it with a
  short TTL rather than removing it.
- No human-in-the-loop checkpoint yet. LangGraph's `interrupt` is its natural
  home after migration.
