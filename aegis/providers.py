"""
aegis.providers — the table of places tokens can come from.

Pure data and pure functions. No os.environ, no network, no I/O. config.py
turns a row of this table into a Settings; catalog.py asks the endpoint what
models it actually has. Keeping those three jobs apart is what lets a new
provider be a dict literal instead of a code change.

WHY ONE TABLE AND NOT A CLASS PER PROVIDER
------------------------------------------
Every entry below speaks the same dialect: OpenAI chat-completions. The only
things that genuinely differ are a URL, an env var, a default model pair,
and a price. Those are values, not behaviour, so they live in a table. The
moment a provider needs real branching logic it will need a class — and the
fact that none of the fourteen below do is the evidence that the abstraction
in llm.py is holding.

FIELD REFERENCE
---------------
    label         Human name for the UI. `openrouter` is an id, not a name.
    base_url      OpenAI-compatible root. Everything else follows from it.
    key_env       Environment variable holding the credential, if any.
    proposer      Default model for the agent that argues.
    critic        Default model for the agent that attacks it — a DIFFERENT
                  family wherever the provider serves one, because a model
                  asked to review its own reasoning style tends to like it.
    requires_key  Whether a credential is mandatory. Not cosmetic: a local
                  Ollama needs none, and treating that as an error once made
                  the entire local path unreachable.
    local         Whether tokens are generated on this machine. Drives cost
                  accounting and which resource guard actually protects you.
    free_tier     Whether you can run this today without paying. Drives one
                  badge in the UI and nothing else — it is a fact about the
                  provider's billing, not about the code.
    free_note     What the free tier actually gives you. Rate limits and
                  free credits are different promises and the difference
                  matters when a run stops halfway.
    console       Where to get a key. A provider picker that names a service
                  but not how to use it just relocates the dead end.
    cost_in/out   USD per 1M tokens, blended and APPROXIMATE. See below.
    models        Known-good model ids, used when the endpoint will not list
                  its own (or has not been asked yet). A starting point, not
                  an inventory — catalog.remote_models() gets the real one.

ABOUT THE PRICES
----------------
They are guard-rails, not billing. `max_cost_usd` stops a stuck loop, and
for that job an estimate within 2x of reality is entirely sufficient. What
is NOT sufficient is a single blended rate for every provider at once, which
is what this replaced: at 0.20/0.60 a DeepSeek run and a Kimi run reported
the same cost while differing by 4x, so the one number that was supposed to
bound spending was wrong for almost every provider it was applied to.

Providers with a free tier still carry a non-zero price on purpose. Pricing
a free tier at 0.00 disables the budget guard entirely (0 * anything never
reaches the limit), and free tiers are exactly where an unattended loop runs
longest. The guard should still be armed; see also Settings.max_seconds,
which is the guard that binds when tokens genuinely cost nothing.

Model ids drift. When one 404s, the fix is the picker's Refresh button —
catalog.remote_models() asks the endpoint — not a grep through this file.
"""

from __future__ import annotations

from typing import Any

# Blended fallback for the ~$0.20/$0.60 tier, used by any entry that does not
# state its own and by the legacy estimator in llm.py.
DEFAULT_COST_IN = 0.20
DEFAULT_COST_OUT = 0.60


PROVIDERS: dict[str, dict[str, Any]] = {
    # -- Local ------------------------------------------------------------
    "ollama": {
        "label": "Ollama (local)",
        # Local, or a remote GPU tunnelled to your machine. Same dialect.
        #
        # WHY ONE SMALL MODEL FOR BOTH ROLES, MEASURED NOT GUESSED
        # -------------------------------------------------------
        # This machine has a GTX 1650 with 4GB of VRAM. Benchmarked on it:
        #
        #   gemma2:2b   1.9GB resident -> 100% GPU        ~50 tok/s
        #   qwen3.5:4b  3.7GB resident -> 41% CPU/59% GPU ~15 tok/s
        #
        # The 4B model does not fit, so Ollama spills 41% of it to the CPU
        # and it runs 3.3x slower. Worse, alternating two models that cannot
        # co-reside in 4GB costs 7-12s of reload on EVERY turn, because each
        # one evicts the other. A two-agent debate does exactly that
        # alternation, so the pathological case is the normal case.
        #
        # Hence: one model, both roles, fully GPU-resident. The roles are
        # separated by system prompt and temperature, not by weights. This is
        # the one entry in the table where the cross-model rule is knowingly
        # broken, and the constraint is arithmetic rather than taste.
        "base_url": "http://localhost:11434/v1",
        "key_env": "OLLAMA_API_KEY",  # Ollama ignores it; the SDK wants something
        "proposer": "gemma2:2b",
        "critic": "gemma2:2b",
        "requires_key": False,
        "local": True,
        "free_tier": True,
        "free_note": "Free forever. Your GPU, your electricity, no rate limit.",
        "console": "https://ollama.com/download",
        "cost_in": 0.0,
        "cost_out": 0.0,
        "models": [],  # discovered from /api/tags — see hostinfo
    },

    # -- Offline ----------------------------------------------------------
    "fake": {
        "label": "Offline (no model)",
        # No network at all. Used by the tests and by the "try it before you
        # pay" path. See llm.FakeLLM.
        "base_url": "",
        "key_env": "",
        "proposer": "fake-proposer",
        "critic": "fake-critic",
        "requires_key": False,
        "local": True,
        "free_tier": True,
        "free_note": "Runs the whole graph with synthetic text. Costs nothing.",
        "console": "",
        "cost_in": 0.0,
        "cost_out": 0.0,
        "models": [],
    },

    # -- Aggregators ------------------------------------------------------
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        # Two different FAMILIES, both verified reachable on a free key.
        # Cross-family matters more here than raw capability: a critic that
        # shares a base model with the proposer tends to approve its own
        # reasoning style, which is the failure this whole system exists to
        # avoid. Checked by hand against the live listing — an earlier pair
        # (deepseek-chat-v3.1:free, glm-4.5-air:free) had been withdrawn from
        # OpenRouter entirely and 404'd on the first turn.
        # The critic is chosen for FORMAT ADHERENCE, not raw capability. Its
        # verdict is parsed, so a model that narrates its deliberation before
        # answering is useless here however smart it is: measured against the
        # real critic prompt, nemotron-3.5-lightning opened with "Here's a
        # thinking process:" and nex-n2.5-pro spent an entire 1200-token
        # budget reasoning and returned nothing at all. Both of these emit
        # `VERDICT:` as their first token, every time.
        "proposer": "nvidia/nemotron-3-super-120b-a12b:free",
        "critic": "poolside/laguna-s-2.1:free",
        "requires_key": True,
        "local": False,
        "free_tier": True,
        "free_note": "Models ending ':free' cost nothing; ~50 requests/day on "
                     "a new key, 1000/day once you top up $10 once. Individual "
                     "free models are rate-limited and some are gated — a 429 "
                     "or 403 means try another, not that the key is bad.",
        "console": "https://openrouter.ai/keys",
        "cost_in": DEFAULT_COST_IN,
        "cost_out": DEFAULT_COST_OUT,
        # A STARTING POINT, not an inventory. OpenRouter serves 400+ models and
        # rotates the free ones often; the picker's Refresh button asks the
        # endpoint for the real list. Everything here was reachable when written.
        "models": [
            "nvidia/nemotron-3-super-120b-a12b:free",
            "nex-agi/nex-n2.5-pro:free",
            "dots-studio/dots-3-note-preview:free",
            "poolside/laguna-s-2.1:free",
            "inclusionai/ling-3.0-flash-vl:free",
            "cohere/north-mini-code:free",
            "nvidia/nemotron-3.5-lightning:free",
            "deepseek/deepseek-chat",
            "openai/gpt-4.1-mini",
        ],
    },

    # -- Fast inference hosts (free tiers, rate-limited) -------------------
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "proposer": "llama-3.3-70b-versatile",
        "critic": "openai/gpt-oss-120b",
        "requires_key": True,
        "local": False,
        "free_tier": True,
        "free_note": "Free tier needs no card. Generous daily token cap, "
                     "low requests/minute — the limit you hit is rate, not cost.",
        "console": "https://console.groq.com/keys",
        "cost_in": 0.59,
        "cost_out": 0.79,
        "models": [
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "moonshotai/kimi-k2-instruct",
            "qwen/qwen3-32b",
            "deepseek-r1-distill-llama-70b",
            "llama-3.1-8b-instant",
        ],
    },
    "cerebras": {
        "label": "Cerebras",
        "base_url": "https://api.cerebras.ai/v1",
        "key_env": "CEREBRAS_API_KEY",
        "proposer": "llama-3.3-70b",
        "critic": "qwen-3-32b",
        "requires_key": True,
        "local": False,
        "free_tier": True,
        "free_note": "Free tier needs no card. The fastest tokens on this "
                     "list by a wide margin — a debate finishes in seconds.",
        "console": "https://cloud.cerebras.ai",
        "cost_in": 0.60,
        "cost_out": 0.60,
        "models": [
            "llama-3.3-70b",
            "qwen-3-32b",
            "gpt-oss-120b",
            "llama3.1-8b",
        ],
    },
    "sambanova": {
        "label": "SambaNova",
        "base_url": "https://api.sambanova.ai/v1",
        "key_env": "SAMBANOVA_API_KEY",
        "proposer": "Meta-Llama-3.3-70B-Instruct",
        "critic": "DeepSeek-V3-0324",
        "requires_key": True,
        "local": False,
        "free_tier": True,
        "free_note": "Free developer tier with per-minute rate limits.",
        "console": "https://cloud.sambanova.ai/apis",
        "cost_in": 0.60,
        "cost_out": 1.20,
        "models": [
            "Meta-Llama-3.3-70B-Instruct",
            "DeepSeek-V3-0324",
            "DeepSeek-R1-Distill-Llama-70B",
            "Qwen3-32B",
        ],
    },

    # -- Model labs, direct ------------------------------------------------
    "moonshot": {
        "label": "Moonshot (Kimi)",
        "base_url": "https://api.moonshot.ai/v1",
        "key_env": "MOONSHOT_API_KEY",
        "proposer": "kimi-k2-0905-preview",
        "critic": "kimi-k2-turbo-preview",
        "requires_key": True,
        "local": False,
        "free_tier": True,
        "free_note": "New accounts get trial credit. Kimi K2 is a 1T-parameter "
                     "MoE and an unusually sharp critic.",
        "console": "https://platform.moonshot.ai/console/api-keys",
        "cost_in": 0.60,
        "cost_out": 2.50,
        "models": [
            "kimi-k2-0905-preview",
            "kimi-k2-turbo-preview",
            "kimi-latest",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
        ],
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "key_env": "DEEPSEEK_API_KEY",
        "proposer": "deepseek-chat",
        # deepseek-reasoner thinks before it answers, which is exactly what a
        # critic should do — and why it gets the reasoning token budget. See
        # Settings.reasoning_max_tokens.
        "critic": "deepseek-reasoner",
        "requires_key": True,
        "local": False,
        "free_tier": False,
        "free_note": "Paid, but among the cheapest serious models available; "
                     "off-peak pricing drops it further.",
        "console": "https://platform.deepseek.com/api_keys",
        "cost_in": 0.27,
        "cost_out": 1.10,
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "zai": {
        "label": "Z.ai (GLM)",
        "base_url": "https://api.z.ai/api/paas/v4",
        "key_env": "ZAI_API_KEY",
        "proposer": "glm-4.5-flash",
        "critic": "glm-4.5-air",
        "requires_key": True,
        "local": False,
        "free_tier": True,
        "free_note": "glm-4.5-flash is free outright — no credits, no card.",
        "console": "https://z.ai/manage-apikey/apikey-list",
        "cost_in": 0.20,
        "cost_out": 1.10,
        "models": ["glm-4.5-flash", "glm-4.5-air", "glm-4.5", "glm-4.6"],
    },
    "mistral": {
        "label": "Mistral",
        "base_url": "https://api.mistral.ai/v1",
        "key_env": "MISTRAL_API_KEY",
        "proposer": "mistral-small-latest",
        "critic": "open-mistral-nemo",
        "requires_key": True,
        "local": False,
        "free_tier": True,
        "free_note": "Free 'Experiment' tier after phone verification.",
        "console": "https://console.mistral.ai/api-keys",
        "cost_in": 0.20,
        "cost_out": 0.60,
        "models": [
            "mistral-small-latest",
            "open-mistral-nemo",
            "mistral-large-latest",
            "magistral-small-latest",
        ],
    },
    "google": {
        "label": "Google AI Studio (Gemini)",
        # Gemini speaks its own dialect AND an OpenAI-compatible one. The
        # compatible endpoint is the whole reason this row is three lines of
        # data rather than a client library.
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "proposer": "gemini-2.5-flash",
        "critic": "gemini-2.0-flash",
        "requires_key": True,
        "local": False,
        "free_tier": True,
        "free_note": "Free tier needs no card. Note that free-tier prompts may "
                     "be used to improve the models — do not send anything private.",
        "console": "https://aistudio.google.com/apikey",
        "cost_in": 0.30,
        "cost_out": 2.50,
        "models": [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
        ],
    },

    # -- Everything else ---------------------------------------------------
    "together": {
        "label": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "key_env": "TOGETHER_API_KEY",
        "proposer": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "critic": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        "requires_key": True,
        "local": False,
        "free_tier": True,
        "free_note": "Model ids ending in '-Free' cost nothing, at low rate limits.",
        "console": "https://api.together.ai/settings/api-keys",
        "cost_in": 0.88,
        "cost_out": 0.88,
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
        ],
    },
    "nvidia": {
        "label": "NVIDIA NIM",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
        "proposer": "meta/llama-3.3-70b-instruct",
        "critic": "deepseek-ai/deepseek-r1",
        "requires_key": True,
        "local": False,
        "free_tier": True,
        "free_note": "1000 free inference credits on signup, no card.",
        "console": "https://build.nvidia.com",
        "cost_in": 0.20,
        "cost_out": 0.60,
        "models": [
            "meta/llama-3.3-70b-instruct",
            "deepseek-ai/deepseek-r1",
            "qwen/qwen2.5-coder-32b-instruct",
            "moonshotai/kimi-k2-instruct",
        ],
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "proposer": "gpt-4.1-mini",
        "critic": "o4-mini",
        "requires_key": True,
        "local": False,
        "free_tier": False,
        "free_note": "Paid only.",
        "console": "https://platform.openai.com/api-keys",
        "cost_in": 0.40,
        "cost_out": 1.60,
        "models": ["gpt-4.1-mini", "gpt-4.1", "o4-mini", "gpt-4o-mini"],
    },

    # -- Escape hatch ------------------------------------------------------
    "custom": {
        "label": "Custom endpoint",
        # The row that makes the table optional. Anything speaking the
        # OpenAI dialect — vLLM, LM Studio, llama.cpp, a company gateway, a
        # provider invented after this file was written — is reachable by
        # typing a URL, with no entry here and no code change. That this is
        # possible at all is the payoff for the single-dialect decision.
        "base_url": "",
        "key_env": "AEGIS_CUSTOM_API_KEY",
        "proposer": "",
        "critic": "",
        "requires_key": False,
        "local": False,
        "free_tier": False,
        "free_note": "Whatever you point it at. Set the base URL and models below.",
        "console": "",
        "cost_in": DEFAULT_COST_IN,
        "cost_out": DEFAULT_COST_OUT,
        "models": [],
    },
}


# ---------------------------------------------------------------------------
# Helpers — pure lookups over the table above
# ---------------------------------------------------------------------------


def preset(name: str) -> dict[str, Any]:
    """The row, or an empty dict. Callers must not mutate what they get."""
    return PROVIDERS.get(name, {})


def label(name: str) -> str:
    return str(PROVIDERS.get(name, {}).get("label") or name)


def is_local(name: str) -> bool:
    return bool(PROVIDERS.get(name, {}).get("local", False))


def is_free(name: str) -> bool:
    return bool(PROVIDERS.get(name, {}).get("free_tier", False))


def cloud_providers() -> list[str]:
    """Everything that talks to somebody else's GPU."""
    return [n for n, p in PROVIDERS.items() if not p.get("local", False)
            and n != "custom"]


def free_providers() -> list[str]:
    """Everything you can run today without paying."""
    return [n for n, p in PROVIDERS.items() if p.get("free_tier", False)]


def known_models(name: str) -> list[str]:
    return list(PROVIDERS.get(name, {}).get("models") or [])


# Substrings that mark a model as one that thinks before it answers.
#
# Inference from a NAME, and only for remote providers, because that is the
# only signal a cloud endpoint offers: Ollama can be asked directly
# (hostinfo.model_capabilities), an OpenAI-compatible /models listing cannot
# — it returns ids and nothing else. A wrong guess here is cheap in one
# direction and not the other: over-guessing spends a larger token budget
# than needed, under-guessing returns an EMPTY answer because the model spent
# the whole budget thinking. So the list errs toward guessing yes.
_REASONING_MARKERS = (
    "deepseek-r1", "deepseek-reasoner", "-r1", "r1-distill",
    "qwq", "qvq", "thinking", "reasoner", "reasoning",
    "o1-", "o3-", "o4-", "magistral", "glm-z1", "-think",
)


def looks_like_reasoning_model(model: str) -> bool:
    """
    Best-effort: does this model id belong to something that thinks?

    Deliberately a guess, and deliberately named as one. The reliable answer
    comes from the server (hostinfo.model_capabilities) and only Ollama
    offers it; for everyone else a name is all there is.
    """
    name = (model or "").lower()
    return any(marker in name for marker in _REASONING_MARKERS)


def reasoning_models(models: list[str]) -> set[str]:
    return {m for m in models if looks_like_reasoning_model(m)}


def is_free_model(model: str) -> bool:
    """
    Does this specific model id cost nothing?

    Two providers encode it in the id itself — OpenRouter suffixes ':free',
    Together suffixes '-free' — which is the only machine-readable statement
    of price available without a second API call.
    """
    name = (model or "").lower()
    return name.endswith(":free") or name.endswith("-free")
