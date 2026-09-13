"""
aegis.config — all environment-dependent settings in exactly one place.

DESIGN RULE: no other module in this package reads os.environ.

Why that rule matters: the moment settings get read from three different
files, you can no longer answer "what model did that run actually use?"
without grepping the codebase. Centralising it means config is a value
you can print, log into a transcript, and diff between runs.

PROVIDER AGNOSTICISM
--------------------
Everything targets an OpenAI-COMPATIBLE HTTP endpoint. A local Ollama,
Groq, Kimi, Gemini, Cerebras, a vLLM server behind a tunnel — all speak the
same dialect, so switching between them is a change to base_url + model
name. Nothing in agents.py or graph.py knows or cares who serves the tokens.

The table of endpoints lives in providers.py, which is pure data. This file
turns one row of it into a Settings, layering in the environment and the
runtime key store. Three sources, one precedence order, stated once:

    provider preset  <  key store  <  environment  <  explicit override

The key store sits BELOW the environment on purpose. A key in the
environment is a deliberate statement about this session; a key in the store
is a convenience remembered from some earlier one. When they disagree, the
deliberate one should win, and the surprising outcome — "I exported a key
and the app used a different one" — is not available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

from . import credentials
from .providers import (
    DEFAULT_COST_IN,
    DEFAULT_COST_OUT,
    PROVIDERS,
    looks_like_reasoning_model,
)

try:  # optional, but present in your environment
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

__all__ = ["PROVIDERS", "Settings", "load_settings", "keystore_path",
           "save_api_key", "forget_api_key"]


@dataclass
class Settings:
    provider: str = "fake"
    base_url: str = ""
    api_key: str = ""

    # Where the credential came from: "env", "store", "preset" or "".
    # Recorded rather than inferred because "the app is using a key I do not
    # remember giving it" is otherwise unanswerable, and because a UI that
    # cannot say which of three sources won cannot help you fix the wrong one.
    key_source: str = ""

    proposer_model: str = ""
    critic_model: str = ""
    # Blank means "reuse the critic model". The Arbiter runs at most once
    # per debate, so this is the one place where paying for a stronger
    # model is cheap: a single call on the hardest questions only.
    arbiter_model: str = ""

    # Two different temperatures on purpose.
    # The proposer is generating - a little creativity helps it explore.
    # The critic is judging - we want it boring, consistent, and repeatable.
    proposer_temperature: float = 0.4
    critic_temperature: float = 0.0
    # The Arbiter is passing judgement. Determinism matters even more here
    # than for the Critic: the same argument should get the same ruling.
    arbiter_temperature: float = 0.0

    # HOW HARD THE CRITIC GRADES. "calibrated" or "adversarial".
    #
    # The default prompt is tuned against aegis/calibration.py and measures
    # 100% balanced accuracy on the eight-case set: it passes good answers and
    # catches invented statistics, false mechanisms and non-answers. That
    # tuning is deliberately biased toward APPROVE, because the opposite
    # failure is worse - a critic that approves nothing makes every debate hit
    # the round cap and marks every answer contested, so the verdict stops
    # carrying information.
    #
    # "adversarial" raises the bar (see CRITIC_ADVERSARIAL_CLAUSE): unstated
    # load-bearing assumptions and underivable numbers become material, and
    # the critic must state the strongest counter-argument before approving.
    # It produces longer debates. It is NOT free, and the cost is measurable -
    # run `./dev.sh calibrate --strictness adversarial` and compare.
    critic_strictness: str = "calibrated"

    # When the debate deadlocks, send it to the Arbiter instead of
    # returning an unreviewed draft. Costs one extra call on hard
    # questions only. On by default because the alternative outcome -
    # handing over never-approved output - is the worst thing this
    # system can do.
    use_arbiter: bool = True

    max_rounds: int = 3
    max_tokens: int = 1200
    request_timeout_s: float = 120.0

    # Budget guard. Debates are cheap, but a stuck loop is not.
    # Set to 0.0 to disable.
    max_cost_usd: float = 0.50

    # WALL-CLOCK GUARD - the counterpart to max_cost_usd.
    #
    # On a paid API a runaway loop costs money, so dollars are the right
    # unit to bound. On local hardware - or a free tier - a runaway loop
    # costs NOTHING and therefore trips no budget guard at all; it just
    # quietly occupies your only GPU, or burns a daily rate limit, for as
    # long as it likes. The resource actually being consumed is time, so
    # that is what has to be bounded.
    #
    # This is the general lesson, not an Ollama detail: a guard denominated
    # in a currency the run does not spend is not a guard. When you change
    # execution substrate, re-ask what the scarce resource now is.
    # 0.0 disables.
    max_seconds: float = 0.0

    # Is inference happening on this machine? Set from the provider preset.
    # Cost accounting and the choice of guard both key off this.
    local: bool = False

    # Does this provider mandate a credential? Also from the preset.
    # Ollama does not; the cloud providers do.
    requires_key: bool = True

    # USD per 1M tokens for THIS provider, from the preset. Previously one
    # blended rate served every provider at once, which made the budget
    # guard wrong by up to 4x depending on who was serving. Approximate on
    # purpose - see the note in providers.py. Zero means "free", which also
    # means the budget guard cannot fire and max_seconds is what protects you.
    cost_in_per_1m: float = DEFAULT_COST_IN
    cost_out_per_1m: float = DEFAULT_COST_OUT

    # Ollama-specific: the KV-cache context window, in tokens. Exposed
    # because on a 4GB card it is a VRAM dial, not a nicety - context
    # allocation is what decides whether a model stays fully GPU-resident
    # or spills to the CPU and runs 3x slower. Cloud providers ignore it.
    num_ctx: int = 0  # 0 = let the server decide

    # Stream tokens as they are generated rather than waiting for the
    # whole completion. On a 50 tok/s local model a 400-token answer is
    # 8 seconds of dead air without this.
    stream: bool = True

    # REASONING MODELS need their own, much larger budget.
    #
    # A thinking model spends tokens on an internal monologue that counts
    # against max_tokens but never appears in the reply. Measured on
    # qwen3.5:4b: a critic prompt burned all 300 tokens thinking and
    # returned an EMPTY string, while reporting 300 completion tokens. At
    # 1500 it thought for ~600 tokens and then answered correctly.
    #
    # So `max_tokens` means two different things depending on the model:
    # "how long may the answer be" for gemma2, and "how long may the
    # thinking PLUS the answer be" for qwen3.5. One number cannot serve
    # both, which is why there are two.
    reasoning_max_tokens: int = 4096

    # KV-cache window for reasoning models, which must hold the prompt AND
    # everything generated - including thinking. A 4096 window with a
    # 4096-token budget cannot work: a 1000-token prompt leaves only 3000 for
    # a model that needs 2450 of thought before its first word. Sizing the
    # context from the *reply* length is the intuition that breaks here.
    reasoning_num_ctx: int = 8192

    # Models known to think. Filled from hostinfo.reasoning_models() for a
    # local server, which can be ASKED, and from a name heuristic for remote
    # ones, which cannot. Left empty by default because config must not probe
    # the network - sensing the environment and configuring it are different
    # jobs, and only the second belongs here.
    reasoning_models: set[str] = field(default_factory=set)

    def budget_for(self, model: str) -> int:
        """Token budget for one call, accounting for hidden thinking."""
        if model in self.reasoning_models:
            return max(self.reasoning_max_tokens, self.max_tokens)
        return self.max_tokens

    def ctx_for(self, model: str) -> int:
        """
        Context window for one call.

        Raised alongside the token budget, and for a reason that is easy to
        miss: num_ctx bounds prompt PLUS generation, so raising only the
        generation budget produces a model that is allowed to write more than
        it is allowed to remember.
        """
        if model in self.reasoning_models:
            return max(self.reasoning_num_ctx, self.num_ctx)
        return self.num_ctx

    # ---- Retrieval ------------------------------------------------------
    # Optional, and optional by design. Grounding is an enrichment: a debate
    # with no evidence must still run, so a missing key degrades to the fake
    # provider rather than raising. See aegis/tools.py.
    tinyfish_api_key: str = ""

    # Run a Researcher turn before the Proposer speaks.
    use_researcher: bool = False

    # How many snippets to put in front of the agents. THREE, and the limit
    # is hardware, not taste: the local context window is 4096 tokens, which
    # must also hold the topic, the previous answer, the open points and the
    # critique. Three snippets is ~150 tokens; ten would crowd out the thing
    # they are evidence for. Cloud models have room for more.
    research_results: int = 3

    transcript_dir: str = "runs"

    extra_headers: dict[str, str] = field(default_factory=dict)

    def redacted(self) -> dict[str, Any]:
        """Safe to print or write into a transcript."""
        data = asdict(self)
        # A set is not JSON-serialisable, and a transcript that cannot be
        # written is a transcript you do not have.
        data["reasoning_models"] = sorted(self.reasoning_models)
        if data.get("tinyfish_api_key"):
            data["tinyfish_api_key"] = f"...{self.tinyfish_api_key[-4:]}"
        if data.get("api_key"):
            data["api_key"] = f"...{self.api_key[-4:]}"
        return data

    @property
    def is_fake(self) -> bool:
        return self.provider == "fake"

    @property
    def is_free(self) -> bool:
        """
        Are these tokens costing anything? Drives which guard is load-bearing.

        Note this is a property of the SETTINGS, not of the provider: a paid
        provider serving a ':free' model id is free, and a free-tier provider
        can be pointed at a paid model in the same account.
        """
        from .providers import is_free_model

        if self.local or self.cost_in_per_1m <= 0:
            return True
        return all(is_free_model(m) for m in
                   (self.proposer_model, self.critic_model) if m)

    @property
    def same_model_both_roles(self) -> bool:
        """
        True when Proposer and Critic share weights.

        Worth surfacing rather than hiding: same-model self-critique is a
        known-weak pattern (a model tends to approve its own reasoning
        style). On 4GB of VRAM it is nonetheless the right default, because
        the alternative is a 7-12s model reload on every single turn. The
        UI displays this so the weakness stays visible instead of becoming
        an invisible assumption.
        """
        return bool(self.proposer_model) and self.proposer_model == self.critic_model


# ---------------------------------------------------------------------------
# Credentials
#
# Thin wrappers so that config stays the only module reading os.environ, and
# every caller gets the same store path without having to know the env var.
# ---------------------------------------------------------------------------


def keystore_path() -> str:
    return str(credentials.store_path(os.getenv("AEGIS_KEYSTORE", "")))


def save_api_key(provider: str, key: str) -> bool:
    return credentials.set_key(provider, key, keystore_path())


def forget_api_key(provider: str) -> bool:
    return credentials.forget_key(provider, keystore_path())


def _resolve_key(provider: str, preset: dict[str, Any]) -> tuple[str, str]:
    """
    (key, where it came from). Environment beats store; see the module note.
    """
    key_env = preset.get("key_env") or ""
    if key_env and os.getenv(key_env):
        return os.environ[key_env], "env"
    stored = credentials.get_key(provider, keystore_path())
    if stored:
        return stored, "store"
    return "", ""


def _detect_provider() -> str:
    """
    Which provider to use when nothing said.

    Only the ENVIRONMENT is consulted, never the key store, and only for
    providers that genuinely require a credential. Both exclusions are
    load-bearing:

      * A key saved from the UI must not silently become the default on the
        next start. Selecting a provider is a choice; remembering its key is
        not the same choice, and conflating them means a fresh clone's
        behaviour depends on which credentials happen to be lying around.

      * Ollama's key_env holds a placeholder the server ignores, so treating
        its presence as evidence of intent picked 'ollama' for anyone who
        had merely copied .env.example.

    Falls back to 'fake', so a fresh clone RUNS immediately instead of
    crashing on a missing credential. Failing loudly is good; failing before
    the user has seen the thing work even once is not.
    """
    for name, preset in PROVIDERS.items():
        if not preset.get("requires_key"):
            continue
        if preset.get("key_env") and os.getenv(preset["key_env"]):
            return name
    return "fake"


def load_settings(provider: str | None = None, **overrides: Any) -> Settings:
    """
    Build Settings from (in order of increasing priority):
        provider defaults  ->  key store  ->  environment  ->  explicit overrides
    """
    provider = provider or os.getenv("AEGIS_PROVIDER") or ""

    if not provider:
        provider = _detect_provider()

    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}. Known: {', '.join(PROVIDERS)}"
        )

    preset = PROVIDERS[provider]
    api_key, key_source = _resolve_key(provider, preset)

    # WHICH ENV VARS APPLY TO THIS PROVIDER, AND WHY IT IS NOT "ALL OF THEM"
    # ---------------------------------------------------------------------
    # AEGIS_PROPOSER_MODEL, AEGIS_CRITIC_MODEL and AEGIS_BASE_URL used to
    # override every provider unconditionally. With one cloud provider and
    # one local one that was merely untidy. With fourteen it is a bug: a
    # .env written for Ollama says `AEGIS_PROPOSER_MODEL=gemma2:2b`, and
    # switching the picker to Groq then sent Groq a model name it has never
    # heard of. The run died at the first turn with a 404 naming a model the
    # user had not chosen and could not see.
    #
    # Those variables are written NEXT TO an AEGIS_PROVIDER line, and they
    # mean "for that provider". So they apply when the resolved provider is
    # the one the environment was describing, and the preset's own defaults
    # apply otherwise. Explicitly asking for a provider the .env does not
    # name is a statement that you want that provider's own settings.
    env_provider = os.getenv("AEGIS_PROVIDER") or ""
    env_owns_provider = (not env_provider) or env_provider == provider

    def _env_model(var: str, default: str) -> str:
        return (os.getenv(var) or default) if env_owns_provider else default

    settings = Settings(
        provider=provider,
        base_url=_env_model("AEGIS_BASE_URL", "") or preset["base_url"],
        api_key=api_key,
        key_source=key_source,
        proposer_model=_env_model("AEGIS_PROPOSER_MODEL", preset["proposer"]),
        critic_model=_env_model("AEGIS_CRITIC_MODEL", preset["critic"]),
        arbiter_model=_env_model("AEGIS_ARBITER_MODEL", ""),
        critic_strictness=(os.getenv("AEGIS_CRITIC_STRICTNESS", "calibrated")
                           if os.getenv("AEGIS_CRITIC_STRICTNESS", "calibrated")
                           in ("calibrated", "adversarial") else "calibrated"),
        use_arbiter=os.getenv("AEGIS_USE_ARBITER", "1") not in ("0", "false", "False"),
        max_rounds=int(os.getenv("AEGIS_MAX_ROUNDS", "3")),
        max_cost_usd=float(os.getenv("AEGIS_MAX_COST_USD", "0.50")),
        transcript_dir=os.getenv("AEGIS_TRANSCRIPT_DIR", "runs"),
        local=bool(preset.get("local", False)),
        requires_key=bool(preset.get("requires_key", True)),
        cost_in_per_1m=float(preset.get("cost_in", DEFAULT_COST_IN)),
        cost_out_per_1m=float(preset.get("cost_out", DEFAULT_COST_OUT)),
        stream=os.getenv("AEGIS_STREAM", "1") not in ("0", "false", "False"),
        tinyfish_api_key=os.getenv("TINYFISH_API_KEY", ""),
        research_results=int(os.getenv("AEGIS_RESEARCH_RESULTS", "3")),
    )

    # Grounding defaults ON when a retrieval key exists, and off otherwise.
    #
    # Keyed on the credential rather than a separate flag because the two are
    # not independent: enabling research without a key would silently ground
    # the debate in the FAKE provider's canned snippets, which is far worse
    # than not grounding it at all - the agents would cite sources that do
    # not exist. AEGIS_USE_RESEARCHER overrides in either direction.
    _research_env = os.getenv("AEGIS_USE_RESEARCHER")
    if _research_env is not None:
        settings.use_researcher = _research_env not in ("0", "false", "False")
    else:
        settings.use_researcher = bool(settings.tinyfish_api_key)

    # ---- Local-inference tuning ------------------------------------------
    # Applied to real local inference only (not the fake provider, whose
    # defaults several tests read). These are not arbitrary: they come from
    # benchmarking gemma2:2b on this machine's GTX 1650 at ~50 tok/s.
    #
    #   max_tokens 1200 -> 500   1200 tokens is 24s of generation PER TURN,
    #                            and a 3-round debate is 6+ turns. That is a
    #                            two-and-a-half minute wait to see whether
    #                            the routing worked. Cloud latency hides
    #                            this; local latency is yours to pay.
    #
    #   max_seconds 0 -> 600     The guard that actually binds locally.
    #
    #   num_ctx 4096             Matches what Ollama already allocates for
    #                            this model, stated explicitly so the VRAM
    #                            budget is visible rather than implicit.
    if settings.local and provider != "fake":
        settings.max_tokens = int(os.getenv("AEGIS_MAX_TOKENS", "500"))
        settings.max_seconds = float(os.getenv("AEGIS_MAX_SECONDS", "600"))
        settings.num_ctx = int(os.getenv("AEGIS_NUM_CTX", "4096"))
        settings.reasoning_max_tokens = int(
            os.getenv("AEGIS_REASONING_MAX_TOKENS", "4096"))
        settings.reasoning_num_ctx = int(
            os.getenv("AEGIS_REASONING_NUM_CTX", "8192"))
        settings.request_timeout_s = float(os.getenv("AEGIS_TIMEOUT_S", "300"))
    else:
        settings.max_tokens = int(os.getenv("AEGIS_MAX_TOKENS", str(settings.max_tokens)))
        # A CLOUD run gets a wall-clock guard too, and for the same reason a
        # local one does: on a free tier the budget guard is denominated in a
        # currency the run does not spend. 900s is long enough that no honest
        # debate hits it and short enough that a wedged one does not run all
        # afternoon against your daily rate limit.
        _default_seconds = "900" if not settings.is_fake else "0"
        settings.max_seconds = float(os.getenv("AEGIS_MAX_SECONDS", _default_seconds))

    # A remote provider cannot be ASKED which of its models think, the way
    # Ollama can (hostinfo.model_capabilities). All that is available is the
    # id, so guess from it — the cost of guessing wrong in the cautious
    # direction is a larger token budget, and in the other direction it is an
    # empty answer. See providers.looks_like_reasoning_model.
    if not settings.local:
        settings.reasoning_models = {
            m for m in (settings.proposer_model, settings.critic_model,
                        settings.arbiter_model)
            if m and looks_like_reasoning_model(m)
        }

    # Ollama accepts any non-empty string; the OpenAI SDK requires one to
    # exist. Supplying it here rather than demanding it from the user's .env
    # is the difference between `--provider ollama` working out of the box
    # and failing on a credential that is never actually checked.
    if settings.local and not settings.requires_key and not settings.api_key \
            and provider != "fake":
        settings.api_key = "ollama-local"

    # OpenRouter asks for these for attribution. Harmless elsewhere.
    if provider == "openrouter":
        settings.extra_headers = {
            "HTTP-Referer": os.getenv("AEGIS_SITE_URL", "http://localhost"),
            "X-Title": "Aegis Orchestration Platform",
        }

    # The FAKE provider ignores model env vars, deliberately.
    #
    # `AEGIS_PROPOSER_MODEL` etc. override every provider, which is right
    # for real endpoints and wrong for this one. Once a .env named real
    # local models, offline runs started recording `gemma2:2b` in their
    # transcripts - a model that generated not one token of that output.
    # A transcript whose model field is a guess is worse than one with no
    # model field, because it will be believed.
    #
    # The offline path has to stay reproducible from a bare clone, which
    # means it cannot depend on the contents of a .env file.
    if provider == "fake":
        settings.proposer_model = preset["proposer"]
        settings.critic_model = preset["critic"]
        settings.arbiter_model = ""
        settings.api_key = ""
        settings.key_source = ""
        # ...and no retrieval either. SECOND TIME this exact mistake was made:
        # a credential in .env (TINYFISH_API_KEY) auto-enabled research, so
        # every test using the fake provider started issuing real HTTP
        # searches - the suite went from 5s to 34s and three cases failed on a
        # researcher turn they never expected.
        #
        # The rule generalises past both instances: NOTHING in .env may change
        # what the fake provider does. Its entire purpose is a path that
        # behaves identically on a bare clone, and a path whose behaviour
        # depends on which credentials happen to be lying around is not that.
        # Tests that want grounded fake runs set use_researcher on the
        # Settings object explicitly, which is visible at the call site.
        settings.use_researcher = False

    for key, value in overrides.items():
        if value is not None and hasattr(settings, key):
            setattr(settings, key, value)

    return settings
