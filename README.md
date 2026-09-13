# Aegis — Agentic Orchestration Platform

Give it a topic. A **Proposer** answers, a **Critic** attacks the answer, the Proposer revises. The loop repeats until the Critic approves. If they deadlock, an **Arbiter** reads the full exchange and rules on it.

Three agents, three possible outcomes: agreement, adjudication, or (with the Arbiter disabled) an unreviewed draft. The system tells you which one you got, because they are not equally trustworthy.

## Run it right now

Two agents arguing on your own GPU. No API key, no account, no network:

```bash
ollama serve &                          # if it is not already running
./.venv/bin/python cli.py "Should a two-person startup use Kubernetes?"
./.venv/bin/streamlit run app.py        # the observability UI
```

That is the default. `.env` sets `AEGIS_PROVIDER=ollama`, so nothing above needs a flag.

Or run it with no model at all:

```bash
python3 cli.py "Is premature optimization always bad?" --provider fake --rounds 1
python3 evaluate.py --provider fake
./.venv/bin/pytest tests/ -q
```

The `fake` provider runs the **real** orchestration graph against a scripted model. Routing, looping, the iteration cap, guards, and transcripts all execute for real; only the text is synthetic. This exists so you can develop and debug the machine without waiting on inference — and it stays reproducible from a bare clone, which is why the fake provider deliberately ignores the model names in `.env`.

## Cloud providers: fourteen endpoints, one dialect

Ollama is the default, but the same graph runs against any endpoint speaking
the OpenAI chat-completions dialect. That is not an integration effort — a
provider is a dict literal in `aegis/providers.py`:

| Provider | Free tier | Get a key |
|---|---|---|
| **Ollama** | free forever, your GPU | — |
| **Offline** | no model at all | — |
| **Groq** | free, no card, very fast | console.groq.com/keys |
| **Cerebras** | free, no card, fastest here | cloud.cerebras.ai |
| **Z.ai (GLM)** | `glm-4.5-flash` free outright | z.ai |
| **Google (Gemini)** | free, no card | aistudio.google.com/apikey |
| **Mistral** | free after phone verification | console.mistral.ai |
| **OpenRouter** | any model id ending `:free` | openrouter.ai/keys |
| **Together** | any model id ending `-Free` | api.together.ai |
| **NVIDIA NIM** | 1000 free credits | build.nvidia.com |
| **SambaNova** | free developer tier | cloud.sambanova.ai |
| **Moonshot (Kimi)** | trial credit | platform.moonshot.ai |
| **DeepSeek** | paid, very cheap | platform.deepseek.com |
| **OpenAI** | paid | platform.openai.com |
| **Custom** | anything else — vLLM, LM Studio, a gateway | set a base URL |

**You do not have to edit a file to use one.** The sidebar's Connection panel
takes a key at runtime and saves it to `~/.aegis/keys.json` (mode `0600`,
outside the repo so it cannot be committed by accident). `.env` still works and
always wins — an environment variable is a deliberate statement about this
session, a saved key is a convenience remembered from an earlier one.

Model lists come from the provider's own `/models` endpoint, not from a
hard-coded table. That matters more than it sounds: both of the OpenRouter
defaults shipped in this file had been **withdrawn** by the time a live key was
first pointed at them. The curated list in `providers.py` is a fallback for
when the endpoint cannot be reached; the picker's refresh button asks the
endpoint for the truth.

### What a free tier actually breaks, and what was done about it

Free endpoints are oversubscribed by construction, and a debate is six or more
calls — so a 5%-flaky call is a 26%-flaky *run*. Three failures showed up the
first time a real key was used, and all three are now handled:

- **`503 Service temporarily overloaded` mid-debate.** Retried up to twice with
  exponential backoff. Only retryable statuses (429, 408, 5xx) qualify — a 401
  will fail identically forever, and retrying it just delays a clear error.
- **A reasoning model that does not admit it.** `nex-agi/nex-n2.5-pro` spent its
  entire 1200-token budget thinking and returned an empty string. Its id
  contains no marker a name heuristic could catch, and a remote `/models`
  listing cannot be asked the way Ollama can. So the *response* is treated as
  the evidence: a reply that is all reasoning, no text, stopped by the budget,
  is a model telling you what it is. It is recorded on the `Settings` and
  retried with the larger budget, so later turns are sized correctly up front.
- **A critic that narrates instead of answering.** The verdict is parsed, so a
  model that opens with `Here's a thinking process:` is useless however capable
  it is. The default critic is chosen for format adherence and the alternatives
  were measured against the real prompt, not guessed at.

Guards adapt too. `max_cost_usd` bounds a paid run, but on a free tier it is
denominated in a currency the run does not spend and can never fire — so a
wall-clock limit is what actually protects you there. Each provider carries its
own price, because one blended rate made the budget guard wrong by up to 4x
depending on who was serving.

## Running locally — and why these particular numbers

Inference runs on this machine through Ollama's OpenAI-compatible endpoint. That is a
config change, not a code change: `aegis/llm.py` speaks one dialect and does not know
who is serving the tokens.

**Measured on an ASUS TUF F15 — GTX 1650, 4GB VRAM, i5-10300H, 15GB RAM:**

| Model | Placement | Generation | Verdict |
|---|---|---|---|
| `gemma2:2b` (1.9GB) | **100% GPU** | **~50 tok/s** | the default |
| `qwen3.5:4b` (3.7GB) | 41% CPU / 59% GPU | ~15 tok/s | does not fit |

Three consequences follow from that table, and they are the whole local design:

**One model, both roles.** Cross-model critique is genuinely better — a model tends to
approve its own reasoning style — but two models that cannot co-reside in 4GB evict each
other, costing **7–12s of reload on every turn** where the speaker changes. A debate
alternates every turn, so the pathological case is the normal case. On this hardware
cross-model critique is not "better but pricier", it is unaffordable. The UI says so in
the sidebar rather than hiding it, so the weakness stays a known limitation instead of an
invisible assumption.

**The budget guard cannot protect a local run.** Local tokens are free, so estimated cost
is `$0.0000` and `max_cost_usd` never fires — a stuck loop would hold your only GPU
indefinitely without spending a cent. A guard denominated in a currency the run does not
spend is not a guard, so there is now a second one denominated in the resource local
inference actually consumes: **`AEGIS_MAX_SECONDS`**. It sits in the same load-bearing
position as the budget guard, above the Arbiter branch, because escalating costs a model
call and a model call costs time.

**Fewer tokens per turn.** 1200 tokens is 24s of generation *per turn* at 50 tok/s, and a
three-round debate is seven calls. The local default is 500. Cloud latency hides this
dial; local latency hands you the bill.

To get more VRAM headroom instead: a second model of roughly **1.8GB or less** would
co-reside with `gemma2:2b` and give you real cross-model critique with no swapping.

### Setup from scratch

Ubuntu blocks system-wide `pip install` (PEP 668), so this project uses a virtualenv. That is the correct fix — never `--break-system-packages`, which risks the Python your OS tools depend on.

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
ollama pull gemma2:2b

cp .env.example .env          # already set up for local inference

./.venv/bin/python cli.py "Your topic here" --save
./.venv/bin/streamlit run app.py
./.venv/bin/pytest tests/ -q
```

Switching to a paid endpoint is two lines in `.env` (`AEGIS_PROVIDER=openrouter` plus
`OPENROUTER_API_KEY`). Cost accounting and the budget guard re-arm automatically, because
pricing is a property of the provider rather than a constant in the client.

## Grounding: giving the agents something to check

**These keys are tools, not models.** TinyFish exposes Search / Fetch / Browser /
Agent endpoints (`X-API-Key`, not `Authorization: Bearer`); Apify is a scraping and
automation platform. Neither speaks chat-completions, so neither can replace
`AEGIS_PROVIDER` — inference still runs on Ollama. What they add is the capability
this system was actually missing.

Consider what the Critic could previously say about a fabricated statistic. Its best
available objection was *"the answer gives no support for the 40% figure"* — a
complaint about **absence**, because absence was the only thing it could detect. It
had no way to say *"the figure is wrong, here is the real one."* A reviewer with no
access to evidence can audit form but never substance, and that is a ceiling no
prompt reaches past. My own calibration suite's hardest tier is full of exactly
these cases.

So `aegis/tools.py` adds retrieval and a `researcher` node:

    [researcher] --> [proposer] <--> [critic] --> [arbiter]
     once, up front      the argument, over a fixed set of facts

Live, this works — the Proposer cites real sources inline:

```
[RESEARCH] SOURCES
[S1] www.reddit.com — Is Kubernetes the best option for my startup
[S2] medium.com — Why Kubernetes Is Overrated for Small Startups
[S3] dev.to — Do you really need Kubernetes in your company/startup?

[ROUND 1] PROPOSER
* [S2] - This source explicitly criticizes Kubernetes' suitability for small
        startups due to its inherent complexity.
```

Five decisions worth stating, because each was forced rather than chosen:

**Retrieved once, before the argument.** Not per round. If the sources changed under
the debate, an improvement between round 1 and round 3 could be a better argument or
just a better search, and there would be no way to tell which. It also keeps the
context budget for the answer.

**Snippets, not full pages.** TinyFish also has a Fetch endpoint returning whole
articles, and on this hardware that is the wrong tool: the local context window is
4096 tokens and one article would consume the entire budget the Proposer needs for
the topic, its previous answer, the open points, and the critique. Snippets are
~200 characters and already carry the checkable part — the number, the date, the
claim. The constraint picks the granularity, and picks well.

**No model call in the Researcher.** A model asked to write a search query for its
own topic mostly paraphrases the topic, so the extra turn buys a rewording for 5–10s
of local inference. This is exactly where agent systems accumulate ceremony: a node
exists, therefore it must call a model. It must not.

**Numbered handles, not URLs.** `[S1]` rather than the raw link, because a small
model asked to reproduce a long URL inside prose will mangle it — and a mangled
citation is worse than none, since it still looks checkable.

**Research is keyed on the credential, and never touches the `fake` provider.**
Enabling retrieval without a key would ground the debate in `FakeSearch`'s canned
snippets — agents citing sources that do not exist, which is worse than no grounding
at all. And a credential in `.env` must never change what the offline path does:
adding this key silently made every fake-provider test issue real HTTP searches,
taking the suite from 5s to 34s and breaking three cases. That is the **second** time
an `.env` value leaked into the offline path (the first was model names), so the rule
is now general: *nothing in `.env` may change the `fake` provider's behaviour.*

**Why Apify is not wired in.** It covers the same capability through Google SERP
actors, but via an async run-and-poll API rather than one GET, and TinyFish Search is
free. Two integrations for one capability is waste. The seam is `SearchProvider` in
`aegis/tools.py` if it is ever wanted — the same provider-agnostic shape as `llm.py`,
for the same reason.

**What did not improve.** The Critic used **zero** citations across every round. It
is the component that most needs evidence, and it ignored it — consistent with every
other measurement of `gemma2:2b` in this project. Grounding raised the ceiling; it
did not raise this model to it.

## "The Critic approved on round one" — is the gate broken?

Usually not, and there is a harness that settles it rather than arguing about it:

```bash
./dev.sh calibrate                              # the configured critic
./dev.sh calibrate --strictness adversarial     # the stricter mode
./dev.sh calibrate --compare modelA modelB
```

It scores the critic on eight answers whose correct verdict is known in
advance, in three tiers: **good** (must APPROVE), **flawed** (a sound-looking
answer containing one real defect — the discrimination zone) and **gross**
(must REVISE). The metric is **balanced accuracy**, mean per-verdict recall,
chosen because it is the only one of three candidates that is symmetrical: a
critic stuck on REVISE and one stuck on APPROVE both score exactly 50%,
because both have thrown away one of the two things a gate is for.

Measured on `poolside/laguna-s-2.1:free`:

```
good    expect APPROVE  2/2
flawed  expect REVISE   3/3
gross   expect REVISE   3/3
balanced accuracy: 100%
diagnosis: CALIBRATED - both verdicts reachable and discriminating.
```

So a round-one APPROVE is usually the gate opening legitimately, not a
skipped review. The prompt is deliberately biased that way: it forbids
blocking on "could also mention X", "lacks examples", "too vague" or "would
be stronger if", and ends *"An answer you could not find a real fault in has
passed."* The opposite failure is worse — a critic that approves nothing makes
every debate hit the round cap and marks every answer contested, so the
verdict stops carrying information.

### Adversarial strictness

If you want longer arguments anyway, `Critic strictness` in the sidebar (or
`AEGIS_CRITIC_STRICTNESS=adversarial`) raises the bar. Three further things
become material: a central claim resting on an **unstated load-bearing
assumption**, a number given **without the derivation** that would tell you
how much weight it deserves, and a recommendation that never says **under what
conditions it would be wrong**. Before approving, the Critic must also state
the strongest argument against its own conclusion and say why it fails.

What does *not* change: the anti-nitpick rules. Absence is still not a defect
and incomplete is still not wrong — the clause raises the standard of RIGOUR,
not of completeness. Without that guard it would simply reproduce the
perfectionism failure.

**This is not free, and the cost is measurable rather than asserted.** A
stricter gate should catch more flawed answers *and* risk failing good ones,
and both halves land in balanced accuracy. Run
`./dev.sh calibrate --strictness adversarial` against your own critic and
compare before trusting it. `calibrated` remains the default precisely because
it is the mode the 100% figure above describes.

## Making it an actual debate: the point ledger

Until now the "debate" was not one. The Critic saw the topic and the current
answer and **nothing else — not even its own previous objections.** So every round
it answered a fresh question:

    "what is wrong with this?"     unbounded - there is always something
    "were my objections answered?" bounded - and therefore able to converge

Two consequences followed, and neither was a prompt problem:

- **Convergence was impossible by construction.** A critic with no memory of what
  it already asked for can go on asking forever. The round cap was silently doing
  the work that agreement should have been doing.
- **The Proposer could dispute a point and never be answered.** It was invited to
  push back, and nothing in the system ever ruled on the pushback. That is not an
  argument; it is two monologues.

So `aegis/state.py` now has a `Point` ledger, and a point has a life:

    raised -> FIXED or DISPUTED by the Proposer -> RESOLVED / WITHDRAWN / OPEN

The Critic must rule on every open point **before** raising new ones, and
`WITHDRAWN` is a first-class outcome: conceding that an objection was wrong is a
working gate, not a loss. A debate where nothing is ever withdrawn is one where
the rebuttals are decorative. `APPROVE` now means *nothing I raised is still
open*, rather than *I failed to think of anything this time*.

Three details worth knowing, each of which is a bug that was found rather than a
design anticipated:

**Only the Critic can close a point.** The Proposer's `DISPUTED` marks a point
`disputed`, not `resolved` — otherwise the party under review decides when review
is finished.

**Unmentioned points stay open.** If the Critic forgets to rule on P2, P2 is still
open. Same fail-safe direction as `parse_verdict`: the cheap error is one extra
round, the expensive one is shipping unreviewed work as reviewed.

**Re-raised points are dropped by the engine.** On a real run the Critic closed P2
as `RESOLVED` and in the same reply raised P3 with almost identical wording — so
the debate deadlocked on something it had just agreed was fixed. The prompt asks
it not to; asking is not enough, because a stateless critic re-derives the same
objection from the same answer. The first dedupe measure (`intersection / min`)
scored those two real paraphrases at 0.68 and let them through, and the fix was
not a lower bar: that measure is inflated whenever one text is short and mostly
contained in a longer one, so lowering it would start merging genuinely distinct
objections. Jaccard scores the same pair at 0.50 and is symmetric.

Routing still branches on the **verdict alone** — the ledger informs, it never
decides. But an `APPROVE` issued while the Critic's own points are still open is
incoherent, so that contradiction is recorded in the decision log rather than
smoothed over. It is a measurable symptom of a critic that is not tracking itself.

## The UI: what you can actually watch

    ./.venv/bin/streamlit run app.py

Not a chat window. The object of interest is the **system**, not the answer — so the
screen is organised around the questions you actually ask when an agent run misbehaves:

The debate renders **side by side, one row per round** — Proposer left, Critic
right. A single scrolling column hides that the two are adversaries; in a grid a
round reads as a clash, and a rebuttal sits level with the objection it answers.
The Arbiter deliberately breaks the grid and spans both columns, because it is
not a participant in the argument, it is the thing that ends it.

**Long turns fold.** Because a round is one row, the taller cell decides how much
blank space the shorter one shows — an 1100-word Proposer answer beside an
80-word critique left most of a screen empty and stopped the two reading as a
pair. Turns past ~1400 characters are truncated at a paragraph break with the
full text one click away. Note this is done by shortening the *text*: the first
attempt clamped the height in CSS and could not work, because Streamlit wraps
every `st.markdown` call in its own container, so a `<div>` opened in one call
and closed in another produces two empty divs and wraps nothing.

The chrome is deliberately quiet — hairlines instead of cards, flat surfaces
instead of gradients and blur, mono micro-type for every number and identifier,
and colour reserved for **meaning** (agent identity, run status). When the chrome
is the most visually assertive thing on screen, a contested point and a
decorative border compete for the same attention.

| Panel | Answers |
|---|---|
| Pipeline strip + live token stream | What is it doing *right now?* Tokens appear as they generate, with the active node lit and the back-edge drawn. |
| **Path** | Which route did this run take? The real graph, with edges that fired drawn solid and coloured — read from the decision log, not re-derived. |
| **Ledger** | What is still contested? Every point, who said what about it, and where it ended up. |
| **Decisions** | *Why did it stop?* Every branch the router took, its reason, and the values it was taken on. |
| **Prompts** | *What did the agent actually see?* The exact system + user prompt per turn. |
| **Telemetry** | *Why was it slow?* Per-turn ttft, tok/s, tokens in/out. |
| Host panel | Is the model on the GPU or spilling to CPU? VRAM, utilisation, residency per model. |
| Transcript / Answer / History | The output, and past runs. |

Two of these deserve a note.

**Decisions is written by the engine, not reconstructed by the UI.** "Why did it stop?" is
the question you most want answered, and a frontend can only answer it two ways:
re-implement the router's conditions — breaking the no-logic-in-the-UI rule and
guaranteeing the copy drifts — or read a record the engine wrote at the moment it decided.
So `route_after_critic` now appends a `Decision` to the state. Control flow is an agent
system's least visible and most consequential part; log it where it happens.

**ttft and tok/s are separated deliberately.** Locally they diagnose different faults: a
slow time-to-first-token means the model was being loaded into VRAM or the prompt was
long, while a slow tok/s means generation itself is slow — usually a model that did not
fit on the GPU. A single latency number makes those identical, and they call for opposite
fixes.

## How it is put together

```
              entry
                |
                v
           [proposer] <------------------+
                |                        |
                v                        | REVISE, rounds
            [critic]                     | and budget remain
                |                        |
                +--- route_after_critic --+
                      |      |
      APPROVE, or     |      | deadlock: cap reached
      budget hit      |      | without agreement
                      |      v
                      |  [arbiter]   (at most once, terminal)
                      |      |
                      v      v
                        END
```

Two things worth noticing. The back-edge from Critic to Proposer is the whole point — a pipeline runs each step once; a **graph with a cycle** is what makes this an agent system rather than three chained prompts. And the Arbiter has *no* edge back into the loop: it is terminal by construction, so adding a third agent did not add a new way to fail to terminate.

**Three outcomes, and they are not equally trustworthy:**

| Outcome | Exit code | What it means |
|---|---|---|
| `approved` | 0 | The Critic was satisfied. Strongest result. |
| `arbitrated` | 3 | The two disagreed; a third agent ruled. A considered judgement on a **contested** question — not a consensus. |
| `max_rounds` | 2 | Arbiter disabled and no agreement. An **unreviewed draft**. Weakest result. |

The system distinguishes these everywhere — CLI colour, exit code, UI banner — because collapsing them would launder a contested ruling as agreement, and that is precisely the signal you most want to keep.

| File | Responsibility |
|---|---|
| `aegis/state.py` | `DebateState` — the single source of truth. Nodes return update dicts; they never mutate. |
| `aegis/llm.py` | The only module that touches the network. Plus `FakeLLM`. |
| `aegis/agents.py` | Prompts and the three agent nodes. Contains `parse_verdict`. |
| `aegis/graph.py` | `MiniGraph` engine + the debate wiring + `route_after_critic`. |
| `aegis/config.py` | The only module that reads `os.environ`. Turns one provider row into a `Settings`. |
| `aegis/providers.py` | The provider table — pure data, no I/O. A new endpoint is a dict literal here. |
| `aegis/catalog.py` | Asks an endpoint what models it actually serves, and whether the key works. Never raises. |
| `aegis/credentials.py` | Runtime key store (`~/.aegis/keys.json`, `0600`). Lowest-priority credential source. |
| `aegis/hostinfo.py` | Read-only host introspection: GPU, VRAM, which models are resident and how much of each is on the GPU. Never raises. |
| `aegis/transcript.py` | Persists every run to `runs/` as JSON and Markdown. |
| `aegis/evaluation.py` | Eval harness — run N topics, aggregate outcomes, compare two reports. |
| `cli.py`, `app.py`, `evaluate.py` | Frontends. **Zero** orchestration logic in any of them. |

## Five design decisions worth understanding

**State is one explicit object.** Everything a run knows lives in `DebateState`. That is what makes replay, checkpointing, evaluation, and debugging possible. Systems that hide state in closures and session variables cannot do any of those things.

**The Critic emits a machine-readable verdict.** `VERDICT: APPROVE` or `VERDICT: REVISE`, parsed into an enum, and the router branches on *that field only* — never on the prose. If control flow depends on interpreting free text, your control flow is a guess.

**Parsing fails safe — but "safe" depends on what the parse controls.** Unparseable critic output becomes `REVISE`: a wrong REVISE costs one extra round, while a wrong APPROVE ships unreviewed work while pretending it passed review. Compare `extract_final_answer`, which fails *open* and returns the whole text: it drives presentation, not control flow, so showing slightly too much beats showing nothing. Same principle both times — pick the failure you can live with.

**Three independent stop conditions, because there are three scarce resources.** A round cap, a budget guard, and a wall-clock guard. None is derivable from the others: one long-context round can outspend five short ones, and a *free* local round still consumes your only GPU. The general lesson is that a guard denominated in a currency the run does not spend is not a guard — so when you change execution substrate, re-ask what the scarce resource now is. Plus a structural step limit inside `MiniGraph` as a fourth backstop against a buggy router.

**Stop conditions are checked in a load-bearing order.** Both resource guards run *before* the deadlock branch, because the Arbiter costs a model call — and a model call costs both money and time. A guard that the escalation path can bypass is not a guard. Whenever you have several stop conditions, ask which one must win when two fire at once.

**The router records why it branched.** `route_after_critic` appends a `Decision` — the rule that fired, the destination, a sentence of reasoning, and the values it observed. Control flow is the least visible and most consequential part of an agent system; the alternative is a UI that re-derives the conditions and drifts out of sync with them.

**Callers declare their own role.** Agent nodes pass `agent="critic"` to the LLM rather than letting it infer identity from prompt text. Three separate bugs in this project came from that inference — the Proposer's prompt contains the word "critic", the Arbiter reuses the Critic's model name. The fix for a bad guess is not a better guess; it is to stop guessing.

**A fake model from day one.** Orchestration bugs and model-quality problems are different categories. `FakeLLM` lets you prove the machine is correct first, then evaluate output quality separately. Tangling the two is why agent projects stall.

## Why `MiniGraph` instead of LangGraph

`MiniGraph` is about 70 lines and does exactly one thing: run nodes, follow edges, ask routers where to go next. It is here so that (a) the project runs with no dependencies, and (b) you can read it and see that agent frameworks are not magic.

LangGraph does considerably more — durable checkpoints, interrupts for human-in-the-loop, parallel branches, distributed execution — and the contract is the same (nodes return update dicts, edges are static or routed). Migration is planned for the multi-agent phase, where those features start to earn their complexity. Right now, for two agents, they would be cost without benefit.

## Eval harness (Phase 0.3)

    python3 evaluate.py --provider fake                              # 8 built-in topics, offline
    python3 evaluate.py --topics topics.txt --provider openrouter --save
    python3 evaluate.py --topics topics.txt --rounds 2 --workers 4 --save
    python3 evaluate.py --compare runs/evals/<a>/report.json runs/evals/<b>/report.json

Runs a batch of topics through the same debate graph and counts what came
back: the outcome distribution (approved / arbitrated / max_rounds /
error), average rounds, and cost — the process metrics, not answer
quality. `aegis/evaluation.py` is a thin consumer of `run_debate`, the
same public entry point the CLI uses, so a case in an eval is identical
to a case run by hand; it adds no orchestration of its own.

Two case-file formats: a `.txt` with one topic per line (`#` comments and
blank lines ignored), or a `.jsonl` with `{"topic": ..., "id": ...}` per
line when you want a stable case id independent of the topic's wording.
No file given falls back to 8 built-in topics spanning different kinds of
disagreement, so a bare `--provider fake` run exercises something before
you write your own list.

`--compare` is the other half of "you cannot tune prompts honestly
without data": run the same case file through two `Settings` (a reworded
prompt, a different model, a different `max_rounds`), save both reports,
and diff them. The output leads with per-case outcome **flips** — same
topic, different outcome — because that is the signal that a change
altered a real decision, not just nudged an aggregate percentage.

What it does **not** do: grade whether an answer is actually *good*. That
is a different kind of measurement (content quality vs. process outcome)
and the same discipline that keeps `tests/test_graph.py` from testing
answer quality applies here — conflating "did the machine behave
correctly" with "was the output good" is why agent projects stall. An
LLM-judge layer is a natural future addition on top of this, not inside it.

## Running it

    ./dev.sh              # the UI on :8899
    ./dev.sh doctor       # GPU, VRAM, what is loaded, what is installed
    ./dev.sh cli "topic"  # one debate in the terminal, streaming
    ./dev.sh test         # 144 hermetic tests, ~5s, no GPU needed
    ./dev.sh test-live    # calibration against the local model
    ./dev.sh eval         # batch eval over the built-in topics
    ./dev.sh calibrate    # is the Critic a working gate?

`dev.sh` checks three things before starting anything: the venv exists, Ollama is up,
and the configured models are pulled. Each of those fails differently and only one fails
loudly — a missing model surfaces as a 404 mid-run, after you have typed a topic and
waited. Three confusing runtime failures become one clear message up front.

## Reasoning models: why `qwen3.5:4b` silently produced nothing

Worth writing down, because the failure was invisible and the cause is not obvious.

`qwen3.5:4b` is a **thinking model**. It reasons before answering, that reasoning arrives
on a separate channel, and — critically — it is **billed against `max_tokens` while never
appearing in the reply**. With the budget tuned for `gemma2:2b` (500 tokens), qwen spent
the entire budget thinking and returned an **empty string** while reporting 300 generated
tokens.

Nothing errored. The empty answer flowed into the Critic, which correctly could not
approve it, so the debate ran the full round cap and produced a confident *"contested"*
verdict about nothing at all. **The system looked like it was working.** That is the
worst category of bug in an agent pipeline: every failure looks the same from outside — a
disappointing paragraph — so a silent one is indistinguishable from a bad model.

Four things fix it:

| Fix | Why |
|---|---|
| **Detect, don't guess** | Ollama's `/api/show` advertises a `thinking` capability. `qwen3.5:4b` does not say "reasoning" anywhere in its name. |
| **Separate token budget** | `max_tokens` means "how long may the answer be" for gemma2 and "how long may the thinking *plus* the answer be" for qwen. One number cannot serve both. Measured: qwen needs ~2450 tokens where gemma2 needs ~200. |
| **Separate context window** | `num_ctx` bounds prompt **plus** generation. A 4096 window with a 4096 budget cannot work. Sizing context from the reply length is the intuition that breaks here. |
| **Fail loudly** | `EmptyCompletionError` names the cause and the setting to change. Returning `""` was the actual bug; the empty budget was just the trigger. |

Reasoning is kept strictly out of the answer text, and that is load-bearing rather than
tidy: the Critic's verdict is parsed from the first line of its reply, so a paragraph of
prepended musing is the difference between a parsed `APPROVE` and a defaulted `REVISE`. A
rendering decision would have become a routing decision.

The honest cost on this hardware: **~260s per turn** (3.5GB, so 40% spills to CPU at
~9 tok/s). It is much the better critic — see below — and a 3-round debate takes 20+
minutes. The UI shows its thinking live and labels the model `· thinks` in the picker.

## First real results — and the first real problem

Local inference means iteration is free, so the eval harness finally has real output to
count. Two topics, three rounds, `gemma2:2b` on both roles:

```
$ ./.venv/bin/python evaluate.py --topics topics.txt --rounds 3 --save
  cases: 2   duration: 138.2s
  outcome distribution:
    arbitrated     2/2  (100.0%)
  avg rounds: 3.0   avg cost: $0.0000
```

**The Critic approved nothing.** Every debate ran the cap and every answer was delivered as
a contested Arbiter ruling.

That is worth stating plainly rather than filing as a quirk. The project's three-outcome
design exists to tell you *how much to trust an answer* — and an outcome that is constant
carries no information. Right now `arbitrated` does not mean "this question is contested",
it means "the Critic ran". The signal is dead until this is fixed.

The two failure modes to watch for were named before any model ran: rubber-stamping
(approves everything immediately) and perfectionism (approves nothing). This is
unambiguously the second — and notably the *opposite* of what same-model self-critique was
predicted to do, which is itself worth understanding rather than explaining away.

The fix is deliberately not "soften the Critic prompt until it approves things". Two causes
were separable: `CRITIC_SYSTEM` may set an unreachable bar, or a 2B model may just be bad at
judging "good enough". Those call for different remedies, and the second predicts the same
prompt behaves better on a bigger model — a testable claim rather than a matter of taste.

### Judging the judge

`evaluate.py` cannot answer that question, because a broken gate produces a *clean-looking*
distribution: 0% approved reads as rigour, 100% reads as a system that works. So there is a
second instrument, `aegis/calibration.py`, which scores the Critic against answers whose
correct verdict is known in advance, in three tiers:

    good    correct, useful, deliberately INCOMPLETE   must APPROVE
    flawed  fluent, confident, materially wrong        must REVISE  <- the point
    gross   obviously broken                           must REVISE

The middle tier is the whole exercise. Any prompt catches a fabricated "Kubernetes was
created by Microsoft in 2003"; the question is whether it catches *"Git stores full
snapshots rather than deltas, so a monorepo never slows down as it grows"* — fluent, sound
in structure, and false.

    ./dev.sh calibrate                                  # the configured critic
    ./dev.sh calibrate --compare gemma2:2b qwen3.5:4b    # same prompt, two models

The score is mean per-**verdict** recall, not accuracy, and that choice matters. Six of the
eight cases expect REVISE, so plain accuracy scores a stuck-on-REVISE critic at 75%. Even
per-*tier* averaging is lopsided (67% vs 33%). Averaging APPROVE-recall with REVISE-recall
puts both degenerate critics at exactly **50%** — neither can buy one failure mode cheaply
to escape the other. *(This module's first version used per-tier averaging and claimed the
two came out level. They did not. Test a metric against known-degenerate inputs before
trusting it.)*

### Resolved: llama3.2:3b is a working gate

`ollama pull llama3.2:3b`, then `./dev.sh calibrate --compare gemma2:2b llama3.2:3b`
on an **identical prompt** (`b3d3171ca0d7`), so any difference is the model:

| | gemma2:2b | llama3.2:3b |
|---|---|---|
| APPROVE recall | 0/2 | **2/2** |
| REVISE recall | 6/6 | **6/6** |
| Balanced accuracy | 50% | **100%** |
| Diagnosis | PERFECTIONISM | **CALIBRATED** |
| Speed | ~50 tok/s | ~46 tok/s |
| Placement | 100% GPU | 100% GPU (2436 MB) |

Both verdicts reachable, both discriminating, at the same speed and still fully
GPU-resident. It is also **non-thinking** (`['completion','tools']`), so none of the
reasoning-budget machinery applies.

And the outcome distribution finally carries information:

    gemma2:2b    approved 1/8 (12.5%)   arbitrated 7/8    avg 2.75 rounds
    llama3.2:3b  approved 3/8 (37.5%)   arbitrated 5/8    avg 2.38 rounds

Different topics now get different outcomes, and approvals land on rounds 1–2. So
`arbitrated` has gone back to meaning *this question was contested* rather than *the
Critic ran*. (`evaluate.py --compare` flags **both** prompt and model as CHANGED
between those two runs, so the eval delta is not attributable to the model alone —
the clean attribution is the calibration table above, which held the prompt fixed.)

**A bug the calibration data exposed.** Both models emitted `P1: RESOLVED / P2: OPEN`
on *first-round* answers that had no prior points — inventing rulings on objections
nobody had made, and the fabricated `P2: OPEN` then drove the verdict to REVISE. A
good answer was rejected on the strength of an imaginary objection. The engine was
already robust (a ruling for an unknown id is ignored, P-prefixed lines never become
new points) so the ledger stayed clean, but the *verdict* did not. The cause was
showing the `RULINGS:` format slot unconditionally: **a format slot is an
instruction** — show a model a section and it will fill it in whether or not it has
anything to put there. The clause is now appended only when prior points exist, and
that fix is what carried llama3.2:3b from 75% to 100%.

**One model for both roles.** Cross-model critique is the stronger pattern, but
gemma2:2b + llama3.2:3b need 4336 MB against 4096 MB of VRAM, so they evict each
other: measured **5.6–6.6s per call alternating versus 0.4s staying on one model**,
i.e. ~5.5s of pure reload on every turn where the speaker changes, ~38s on a 7-turn
debate. `.env` documents the mixed config for anyone who wants to pay that.

### Four prompt revisions, and what they proved

| Version | good | flawed | Diagnosis |
|---|---|---|---|
| v1 | 0/5 | — | Perfectionism. Listed "a missing counter-argument" as grounds for REVISE — a condition every answer satisfies. |
| v2 | 8/8 approved | 1/3 | Rubber stamp. Caught fabrications, missed a false mechanism and an answer that never answered. |
| v3 | 0/5 | 3/3 | Perfectionism again. Its own exclusion list was stated *before* the checks, and a 2B model followed the later, more emphatic instruction. |
| v4 | 0/2 | 3/3 | Perfectionism. Balanced accuracy exactly **50%**. |

Four attempts, never a middle — and that pattern is itself the result. **A model that swings
wholesale with the lean of the wording is not representing the distinction being asked for,**
and no further prompt edit conjures the capability.

The decisive test is one model swap with the prompt held fixed: on the identical v4 prompt
(`32a24e869fc7`), **`qwen3.5:4b` approves the strong answer that `gemma2:2b` rejects 5 times
out of 5** — reproduced twice, 257–263s per call. The perfectionism was the model, not the
wording.

But qwen is not a drop-in replacement, and the reason is worth recording. On the second
"good" case it generated **3208 tokens of pure reasoning and then stopped without answering
at all** — 888 tokens *short* of its budget, so it did not run out of room; it thought
itself to a standstill. Its judgement is better when it arrives, and on 4GB it does not
reliably arrive.

That distinction is now in the error message, because the two causes need opposite fixes:

    finish_reason=length  ->  ran out of room; raise the budget and the context
    finish_reason=stop    ->  gave up N tokens short; a larger budget will not help

The first version of that message said "spent all N of its budget" in both cases, telling
you to raise a limit the model had never reached. A diagnostic that names the wrong fix is
worse than none, and it took real data to notice.

**Where that leaves the Critic.** Neither installed model is a good gate here: `gemma2:2b`
is fast and reliable but cannot approve anything; `qwen3.5:4b` judges well but takes four
minutes a turn and sometimes fails to answer. The open question is now model *selection*,
not prompt tuning — which is progress of a different kind, and cheaper to act on. The most
promising untried option is a small **non-thinking** 3–4B model (Llama 3.2 3B, Phi-3 mini):
fast enough to iterate on, stronger than 2B, and at ≤1.8GB it would co-reside with
`gemma2:2b` for genuine cross-model critique with no swapping. Options are ranked in
`MEMORY.md`; the one thing not to do is reach for a fifth prompt revision by feel.

## Roadmap

| Phase | Work |
|---|---|
| 0.1 ✅ | Proposer/Critic debate, MiniGraph engine, CLI + Streamlit, transcripts |
| 0.2 ✅ | Arbiter (third agent) resolves deadlocks; Streamlit verified via `AppTest`; **40 tests** |
| 0.3 ✅ | Eval harness — run N topics, score outcomes, compare prompt versions with data; **58 tests** |
| 0.4 ✅ | **Local inference + observability.** Keyless local providers, token streaming, wall-clock guard, decision log, per-turn prompt capture, host/VRAM panel; **82 tests**. First real model ever run through the system. |
| 0.5 | Migrate engine to LangGraph (checkpoints, interrupts, human-in-the-loop) |
| 0.6 | Swap in the code pipeline: Planner → Coder → Reviewer, unified diffs, sandboxed test runs |

Phase 0.4 is where the original strategy report's Sprints 1–5 land. The orchestration rails built here carry over unchanged; only the agent roles and their tools differ.

## Note on hardware — a conclusion worth revisiting

The strategy report in this folder concluded that the GTX 1650's 4GB VRAM "cannot host a
useful model", and the project was built around remote inference on that basis. Measured
directly, that conclusion was **half right**, and the half that was wrong mattered.

Right: a 7B model is impossible. Weights alone need ~4.5GB before any context, and
`qwen3.5:4b` already spills 41% of itself to the CPU at 3.7GB. That is arithmetic, not
tuning.

Wrong: "no useful model fits." `gemma2:2b` sits entirely on the GPU at 1.9GB and sustains
~50 tok/s with a 4096-token context — enough to run a full three-round debate with an
Arbiter ruling in about 45 seconds, for free, offline. The report's reasoning was sound but
its premise was a 7B model; the conclusion was then applied to *all* local inference.

Two things generalise from that. **A capacity conclusion is only valid for the size class
it was computed on** — re-derive it when the size class changes. And the architecture
survived being wrong precisely because the provider boundary was drawn properly:
supporting local inference needed no change to `agents.py`, `graph.py`, or `state.py`. What
it *did* need was the removal of assumptions the two cloud providers had quietly baked in —
that a key always exists, and that tokens always cost money. Those were not in the
abstraction; they were in the details underneath it, which is where this kind of
assumption usually hides.
