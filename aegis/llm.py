"""
aegis.llm — the only module allowed to make a network call.

Everything else in Aegis talks to the `LLM` protocol below. That single
boundary buys us three things:

  1. Provider swaps are a config change, not a code change.
  2. Tests run with ZERO network and ZERO API key (see FakeLLM).
  3. Token/cost accounting happens in one place instead of being
     sprinkled through the agent code.

THE FAKE-FIRST PRINCIPLE
------------------------
FakeLLM is not an afterthought or a testing nicety. It is the thing that
lets you develop the ORCHESTRATION - the state machine, the routing, the
iteration cap, the UI - without spending a cent or waiting on latency.

Orchestration bugs and model-quality problems are completely different
categories of problem. Mixing them is why agent projects stall. With a
FakeLLM you can prove the machine is correct first, then plug in a real
model and evaluate only its output quality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from .config import Settings
from .providers import DEFAULT_COST_IN, DEFAULT_COST_OUT


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0

    # Time To First Token. Reported separately from total latency because
    # locally they measure two different things: ttft is dominated by model
    # LOADING and prompt processing, total latency by generation speed. A
    # turn that is slow because the model had to be swapped into VRAM and a
    # turn that is slow because the answer was long look identical in a
    # single latency number, and they call for opposite fixes.
    ttft_s: float = 0.0
    tokens_per_s: float = 0.0

    # A reasoning model's internal monologue, which arrives in its own
    # channel and is billed against the same token budget as the reply.
    # Captured rather than discarded: it is the only way to see WHY a
    # thinking model answered as it did, and the only way to explain a
    # turn that spent 600 tokens and said three sentences.
    reasoning: str = ""
    reasoning_tokens: int = 0

    # Why the model stopped: "stop" (it chose to), "length" (it hit the
    # budget), or "" when the server does not say. Load-bearing for
    # diagnosis, not decoration - a reasoning model that returns nothing
    # because it ran out of budget and one that returns nothing because it
    # thought itself to a standstill need opposite responses, and they are
    # indistinguishable from the token count alone.
    finish_reason: str = ""

    raw: Any = None


Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": ...}


# Called with each token as it is generated: on_token(agent_role, text).
TokenSink = Callable[[str, str], None]


class LLM(Protocol):
    """
    The entire surface area an agent is allowed to depend on.

    `agent` is the caller DECLARING its own role rather than leaving the
    LLM to infer it. That parameter exists because inferring role from
    prompt text has now caused three separate bugs in this project (see
    FakeLLM._is_critic): the proposer's prompt mentions the word "critic",
    the arbiter reuses the critic's model name, and so on. Identity is
    something the caller knows for certain; asking the callee to guess it
    from incidental substrings is inviting exactly those bugs.

    `on_token` receives tokens as they arrive, which is what makes live
    UI output possible. It is optional, and ignored by non-streaming
    implementations.
    """

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        agent: str = "",
        on_token: TokenSink | None = None,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Real client
# ---------------------------------------------------------------------------


class EmptyCompletionError(RuntimeError):
    """
    The model returned no visible text.

    Its own exception type because the cause is specific and the fix is
    specific: a reasoning model spent the whole token budget thinking. The
    alternative - returning "" and carrying on - is far worse than it
    sounds. An empty answer flows into the Critic, which correctly cannot
    approve it, so the debate runs to the round cap and produces a
    confident-looking "contested" verdict about nothing at all. The system
    LOOKS like it is working. Failing here converts a silent wrong answer
    into a loud, diagnosable one.
    """


class OpenAICompatibleLLM:
    """
    Works against ANY endpoint that speaks the OpenAI chat-completions
    dialect: OpenRouter, DeepSeek, Together, Fireworks, vLLM, Ollama.

    Note there is no provider-specific branching in here beyond headers.
    That is the whole point - your report flags vendor lock-in as a real
    risk and prescribes exactly this shape as the mitigation.
    """

    def __init__(self, settings: Settings) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'openai' package is required for live calls. "
                "Install it, or run with provider='fake'."
            ) from exc

        # Only providers that ACTUALLY require a credential are blocked.
        #
        # The previous version raised whenever api_key was empty, full stop.
        # That is correct for OpenRouter and DeepSeek and completely wrong
        # for a local Ollama, which authenticates nothing - and since it was
        # the constructor that raised, the entire local path was unreachable
        # before a single token was generated. The assumption "all providers
        # need a key" was invisible because the only two providers written
        # first both happened to satisfy it.
        if settings.requires_key and not settings.api_key:
            # Name the ACTUAL variable, read from the preset. Deriving it as
            # f"{provider.upper()}_API_KEY" was right for the two providers
            # that existed when it was written and wrong for several since:
            # Gemini's is GEMINI_API_KEY, not GOOGLE_API_KEY. An error that
            # confidently names a variable which does nothing sends people
            # off to debug their shell.
            from .providers import preset as _preset

            var = _preset(settings.provider).get("key_env") or "the provider's API key"
            raise RuntimeError(
                f"No API key for provider '{settings.provider}'. Paste one into "
                f"the sidebar's Connection panel, set {var} in your .env, or "
                f"switch the provider to 'fake' to run the graph offline."
            )

        self.settings = settings

        # The OpenAI SDK refuses to construct without SOME api_key, even
        # against a server that authenticates nothing - and the error it
        # raises talks about OPENAI_API_KEY, which is actively misleading
        # when you are pointed at localhost:11434. Supplying a placeholder
        # for providers that declare requires_key=False keeps that
        # transport detail from surfacing as a fake auth problem.
        #
        # Note this is NOT a way around the real check above: for a
        # provider that does require a key, that check has already raised.
        self._client = OpenAI(
            base_url=settings.base_url or None,
            api_key=settings.api_key or "not-required",
            timeout=settings.request_timeout_s,
            default_headers=settings.extra_headers or None,
        )

    # Extra body params the OpenAI schema has no field for. Ollama reads
    # `options`; every other provider ignores an unknown key. This is the
    # one concession to a specific server, and it is deliberately confined
    # to a single method so the rest of the class stays dialect-pure.
    def _extra_body(
        self, model: str, *, disable_thinking: bool = False,
    ) -> dict[str, Any] | None:
        ctx = self.settings.ctx_for(model)
        body: dict[str, Any] = {}
        if ctx > 0:
            body["options"] = {"num_ctx": ctx}
        if disable_thinking:
            # Ollama's OpenAI-compatible endpoint accepts this field for
            # thinking models. A direct retry is cheaper and more reliable
            # than repeatedly expanding the hidden-reasoning budget.
            body["reasoning_effort"] = "none"
        return body or None

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        agent: str = "",
        on_token: TokenSink | None = None,
    ) -> LLMResponse:
        # A thinking model needs a bigger budget than the caller asked for,
        # because part of it is spent where the caller cannot see. Resolved
        # here rather than at the call site so no agent has to know which
        # kind of model it is talking to.
        budget = max(max_tokens, self.settings.budget_for(model))

        response = self._complete_once(
            messages, model=model, temperature=temperature, max_tokens=budget,
            agent=agent, on_token=on_token,
        )

        # A local reasoning model can consume thousands of tokens privately
        # and still emit no answer. Retry once with thinking disabled so the
        # debate receives a usable turn instead of failing after minutes of
        # GPU time. The original attempt is retained in the telemetry below.
        if (self.settings.local and response.reasoning and not response.text
                and response.finish_reason == "length"):
            if on_token is not None:
                on_token(f"{agent}:thinking", "\n\nRetrying without hidden reasoning...\n")
            retry_budget = max_tokens
            retry = self._complete_once(
                messages, model=model, temperature=temperature,
                max_tokens=retry_budget, agent=agent, on_token=on_token,
                disable_thinking=True,
            )
            self._guard_empty(retry, retry_budget)
            return self._merge_retry(response, retry)

        # THE SAME FAILURE, REMOTELY — and no `local` flag to key off.
        #
        # Whether a model thinks before answering is something Ollama can be
        # ASKED (hostinfo.model_capabilities). A cloud /models listing returns
        # ids and nothing else, so config guesses from the name, and the guess
        # is wrong for every reasoning model whose id does not advertise it:
        # `nex-agi/nex-n2.5-pro` spent all 1200 tokens reasoning and returned
        # an empty string, exactly as an undetected local thinker used to.
        #
        # The name is a guess; THIS RESPONSE IS EVIDENCE. A reply that is all
        # reasoning and no text, stopped by the budget, is a model telling us
        # what it is. So record that on the settings — every later turn now
        # gets the larger budget up front — and retry this one with it.
        #
        # Retrying with a BIGGER BUDGET rather than with thinking disabled,
        # which is what the local path does: `reasoning_effort: none` is an
        # Ollama extension that most providers ignore, whereas a token budget
        # is honoured by all of them.
        if (not self.settings.local and response.reasoning and not response.text
                and response.finish_reason == "length"):
            retry_budget = max(self.settings.reasoning_max_tokens, budget * 2)
            if retry_budget > budget:
                self.settings.reasoning_models = (
                    set(self.settings.reasoning_models) | {model})
                if on_token is not None:
                    on_token(f"{agent}:thinking",
                             f"\n\n{model} reasons before answering — retrying "
                             f"with {retry_budget} tokens...\n")
                retry = self._complete_once(
                    messages, model=model, temperature=temperature,
                    max_tokens=retry_budget, agent=agent, on_token=on_token,
                )
                self._guard_empty(retry, retry_budget)
                return self._merge_retry(response, retry)

        self._guard_empty(response, budget)
        return response

    def _complete_once(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        agent: str,
        on_token: TokenSink | None,
        disable_thinking: bool = False,
    ) -> LLMResponse:
        # TRANSIENT UPSTREAM FAILURES ARE THE NORMAL CASE ON A FREE TIER.
        #
        # A free endpoint is oversubscribed by construction, and it says so
        # with 429 (rate limited) and 503 (overloaded). Both are statements
        # about the next few seconds, not about the request — and without a
        # retry, one of them anywhere in a six-turn debate destroys the whole
        # run. That is exactly what happened: round two died on "Upstream
        # error from Nvidia: Service temporarily overloaded" after the first
        # round had already completed successfully.
        #
        # Bounded and backed off, because the failure it must not cause is a
        # hammering loop against a provider that is already struggling. Only
        # retryable statuses are caught: a 401 or a 404 will fail identically
        # forever, and retrying it just delays a clear error.
        last: Exception | None = None
        for attempt in range(self._MAX_TRANSIENT_RETRIES + 1):
            try:
                return self._dispatch(
                    messages, model=model, temperature=temperature,
                    max_tokens=max_tokens, agent=agent, on_token=on_token,
                    disable_thinking=disable_thinking,
                )
            except Exception as exc:  # noqa: BLE001 - re-raised unless retryable
                if not _is_transient(exc) or attempt == self._MAX_TRANSIENT_RETRIES:
                    raise
                last = exc
                delay = self._TRANSIENT_BACKOFF_S * (2 ** attempt)
                if on_token is not None:
                    on_token(f"{agent}:thinking",
                             f"\n\n{model}: {type(exc).__name__} — retrying "
                             f"in {delay:.0f}s ({attempt + 1}/"
                             f"{self._MAX_TRANSIENT_RETRIES})...\n")
                time.sleep(delay)
        raise last  # unreachable; the loop either returns or raises

    # Three attempts total. A free model that is overloaded for more than
    # ~7 seconds is usually overloaded for minutes, and at that point failing
    # with a clear message beats silently stretching a debate to ten.
    _MAX_TRANSIENT_RETRIES = 2
    _TRANSIENT_BACKOFF_S = 2.0

    def _dispatch(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        agent: str,
        on_token: TokenSink | None,
        disable_thinking: bool = False,
    ) -> LLMResponse:
        if self.settings.stream:
            return self._complete_streaming(
                messages, model=model, temperature=temperature,
                max_tokens=max_tokens, agent=agent, on_token=on_token,
                disable_thinking=disable_thinking,
            )
        return self._complete_blocking(
            messages, model=model, temperature=temperature, max_tokens=max_tokens,
            disable_thinking=disable_thinking,
        )

    @staticmethod
    def _merge_retry(first: LLMResponse, retry: LLMResponse) -> LLMResponse:
        """Keep both attempts in the ledger while returning the usable reply."""
        total_out = first.tokens_out + retry.tokens_out
        total_latency = first.latency_s + retry.latency_s
        return LLMResponse(
            text=retry.text,
            model=retry.model,
            tokens_in=first.tokens_in + retry.tokens_in,
            tokens_out=total_out,
            latency_s=round(total_latency, 3),
            ttft_s=first.ttft_s,
            tokens_per_s=round(total_out / total_latency, 1) if total_latency else 0.0,
            cost_usd=first.cost_usd + retry.cost_usd,
            reasoning=(first.reasoning + "\n\n" + retry.reasoning).strip(),
            reasoning_tokens=first.reasoning_tokens + retry.reasoning_tokens,
            finish_reason=retry.finish_reason,
            raw=retry.raw,
        )

    @staticmethod
    def _guard_empty(response: LLMResponse, budget: int) -> None:
        """Refuse to hand back an invisible failure. See EmptyCompletionError."""
        if response.text:
            return
        if response.reasoning:
            # Two causes, two fixes, and the token count alone cannot tell
            # them apart. Ran out of room -> give it more. Stopped on its own
            # having produced only thought -> more room will not help; the
            # model is not reliably completing this task at this size.
            #
            # The first version of this message said "spent all N of its
            # BUDGET" in both cases, which pointed at raising the budget even
            # when the model had stopped 900 tokens short of it. A diagnostic
            # that names the wrong fix costs more than no diagnostic.
            ran_out = (response.finish_reason == "length"
                       or response.tokens_out >= budget)
            if ran_out:
                remedy = (f"It ran out of room. Raise AEGIS_REASONING_MAX_TOKENS "
                          f"(currently allowing {budget}) and "
                          f"AEGIS_REASONING_NUM_CTX with it, since context "
                          f"bounds prompt plus generation.")
            else:
                remedy = (f"It stopped on its own {budget - response.tokens_out} "
                          f"tokens short of the limit, so a larger budget will "
                          f"not help: the model thought itself to a standstill. "
                          f"Retry, or use a model that completes this task "
                          f"reliably at this size.")
            raise EmptyCompletionError(
                f"{response.model} produced no answer: {response.tokens_out} "
                f"tokens generated, all of it reasoning "
                f"(~{response.reasoning_tokens} thinking tokens), reply empty. "
                f"finish_reason={response.finish_reason or 'unreported'}, "
                f"budget={budget}.\n\n{remedy}"
            )
        raise EmptyCompletionError(
            f"{response.model} returned an empty response "
            f"({response.tokens_out} completion tokens, {budget}-token budget). "
            f"The model may not be loaded, or the prompt may have been refused."
        )

    def _complete_blocking(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        disable_thinking: bool = False,
    ) -> LLMResponse:
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=self._extra_body(model, disable_thinking=disable_thinking),
        )
        elapsed = time.perf_counter() - started

        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        tin = getattr(usage, "prompt_tokens", 0) or 0
        tout = getattr(usage, "completion_tokens", 0) or 0
        reasoning = _reasoning_of(choice.message)

        return LLMResponse(
            text=(choice.message.content or "").strip(),
            model=model,
            tokens_in=tin,
            tokens_out=tout,
            latency_s=round(elapsed, 3),
            ttft_s=round(elapsed, 3),  # indistinguishable without streaming
            tokens_per_s=round(tout / elapsed, 1) if elapsed >= 1e-3 else 0.0,
            cost_usd=self._price(tin, tout),
            reasoning=reasoning,
            reasoning_tokens=len(reasoning) // 4,
            finish_reason=getattr(choice, "finish_reason", "") or "",
            raw=response,
        )

    def _complete_streaming(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        agent: str,
        on_token: TokenSink | None,
        disable_thinking: bool = False,
    ) -> LLMResponse:
        """
        Same call, consumed incrementally.

        Streaming here is not a UI garnish. On this hardware a 500-token
        answer takes ~10 seconds, and a debate is six or more of those. A
        blocking client gives you a minute of silence and then a wall of
        text, which means that when something goes wrong - a model looping,
        a prompt misfiring - you find out at the end instead of at the
        second token. Watching the agents write IS the observability.

        `include_usage` asks the server for real token counts in a final
        chunk; Ollama honours it. If a server does not, we fall back to a
        character-based estimate rather than reporting a confident zero -
        a wrong-but-labelled number beats a plausible-looking lie.
        """
        started = time.perf_counter()
        # None, not 0.0, as the "not yet measured" sentinel. Time-to-first-
        # token can legitimately BE ~0 for a cached or stubbed response, and
        # with 0.0 doing double duty the code could not tell "no token yet"
        # from "a token arrived instantly" - which collapsed the generation
        # window to 1 microsecond and reported 190,000,000 tok/s. Any sentinel
        # inside the valid range of the thing it guards is a bug waiting for
        # the right input.
        ttft: float | None = None
        chunks: list[str] = []
        usage = None

        stream = self._client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=self._extra_body(model, disable_thinking=disable_thinking),
        )

        thoughts: list[str] = []
        finish_reason = ""
        for event in stream:
            if getattr(event, "usage", None):
                usage = event.usage
            if not event.choices:
                continue
            if getattr(event.choices[0], "finish_reason", None):
                finish_reason = event.choices[0].finish_reason
            delta = event.choices[0].delta

            # Reasoning arrives in its own channel and must NOT be mixed into
            # the answer text: the Critic's verdict is parsed from the first
            # line of the reply, and a paragraph of prepended musing would
            # push it out of reach. Streamed separately so a UI can show the
            # model thinking without corrupting what the parser sees.
            thought = _reasoning_of(delta)
            if thought:
                # Thinking counts as time-to-first-token. Measuring ttft from
                # the first CONTENT token instead made a reasoning model
                # report 237 tok/s: 113s of thought landed inside "ttft", so
                # the generation window shrank to the few seconds of visible
                # reply while the token count still included every thinking
                # token. ttft means "when did the model start working", and
                # thinking is working.
                if ttft is None:
                    ttft = time.perf_counter() - started
                thoughts.append(thought)
                if on_token is not None:
                    on_token(f"{agent}:thinking", thought)

            piece = delta.content
            if not piece:
                continue
            if ttft is None:
                ttft = time.perf_counter() - started
            chunks.append(piece)
            if on_token is not None:
                on_token(agent, piece)

        elapsed = time.perf_counter() - started
        text = "".join(chunks).strip()
        reasoning = "".join(thoughts).strip()

        tin = getattr(usage, "prompt_tokens", 0) or 0
        tout = getattr(usage, "completion_tokens", 0) or 0
        if tout == 0 and text:                       # server withheld usage
            tout = max(1, len(text) // 4)
        if tin == 0:
            tin = sum(len(m["content"]) for m in messages) // 4

        # Generation rate excludes time-to-first-token on purpose. TTFT is
        # model loading and prompt ingestion; including it would make a
        # cold start look like a slow GPU.
        #
        # But a window narrower than a millisecond is not a measurement, it
        # is a division by almost-zero - so fall back to total elapsed, and
        # then to "no rate at all" rather than emitting a number with no
        # physical meaning. A missing metric is honest; an absurd one gets
        # believed and quoted.
        ttft = ttft if ttft is not None else elapsed
        gen_window = elapsed - ttft
        if gen_window < 1e-3:
            gen_window = elapsed
        rate = round(tout / gen_window, 1) if gen_window >= 1e-3 else 0.0

        return LLMResponse(
            text=text,
            model=model,
            tokens_in=tin,
            tokens_out=tout,
            latency_s=round(elapsed, 3),
            ttft_s=round(ttft, 3),
            tokens_per_s=rate,
            cost_usd=self._price(tin, tout),
            reasoning=reasoning,
            reasoning_tokens=len(reasoning) // 4,
            finish_reason=finish_reason,
            raw=None,  # stream objects are consumed; nothing useful to keep
        )

    def _price(self, tokens_in: int, tokens_out: int) -> float:
        """
        Cost for this provider, at this provider's rates.

        LOCAL INFERENCE IS FREE, and saying otherwise is not a rounding
        error - it corrupts a safety mechanism. `max_cost_usd` is a guard
        the router checks BEFORE deciding whether to escalate (see
        route_after_critic). Charging phantom cents for tokens generated on
        your own GPU means that guard fires on money nobody spent, killing
        healthy runs. Report zero, and bound local runs with the resource
        they actually consume - time. See Settings.max_seconds.

        The rates come from the provider preset rather than from one blended
        constant. That constant was defensible while there were two cloud
        providers within 2x of each other; across fourteen it was wrong by up
        to an order of magnitude in both directions, which makes a budget
        guard denominated in it worse than no guard - it fires early on the
        cheap providers and late on the expensive ones.
        """
        if self.settings.local:
            return 0.0
        return _estimate_cost(
            tokens_in, tokens_out,
            cost_in_per_1m=self.settings.cost_in_per_1m,
            cost_out_per_1m=self.settings.cost_out_per_1m,
        )


# HTTP statuses worth trying again, and the ones that are not.
#
# 429 rate limited, 408 request timeout, and the 5xx family are all statements
# about the server's next few seconds. 401/403/404/400 are statements about
# the request and will fail identically forever.
_TRANSIENT_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


# A 429 that will NOT clear on its own within a retry window.
#
# Rate limits come in two kinds and they need opposite responses. A
# per-second or per-minute limit is exactly what backoff is for. A per-DAY
# quota is not: retrying it twice spends two more requests from a balance
# already at zero and delays the real message - which, helpfully, the
# provider already spells out ("Wait for the daily reset, or purchase
# credits"). Measured against OpenRouter's free tier, whose body reads
# 'Rate limit exceeded: free-models-per-day'.
#
# Note this matches on the message in the direction that is SAFE. The
# warning below about string matching is about retrying something that
# should not be retried; declining to retry costs at most one clear error
# arriving sooner.
_EXHAUSTED_QUOTA_MARKERS = ("per-day", "per day", "daily", "quota",
                            "insufficient_quota", "billing")


def _is_transient(exc: Exception) -> bool:
    """
    Is this worth another attempt?

    Read from the exception's `status_code` when the SDK supplies one, and
    from the text only as a fallback - matching on message strings is how you
    end up retrying a 404 because it happened to contain the word "timeout".
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        if status == 429 and any(m in str(exc).lower()
                                 for m in _EXHAUSTED_QUOTA_MARKERS):
            return False
        return status in _TRANSIENT_STATUSES

    # Connection-level failures never reach a status code but are exactly the
    # kind of thing that succeeds on a second attempt.
    name = type(exc).__name__
    if name == "RateLimitError":
        return not any(m in str(exc).lower() for m in _EXHAUSTED_QUOTA_MARKERS)
    if name in ("APIConnectionError", "APITimeoutError", "InternalServerError"):
        return True
    return False


def _reasoning_of(obj: Any) -> str:
    """
    Pull the thinking channel off a message or delta.

    Providers have not converged on a name for this: Ollama sends
    `reasoning`, others send `reasoning_content`. Checking both is cheaper
    than caring which server is on the other end, and returning "" for a
    model that has no such channel keeps every non-reasoning path unchanged.
    """
    for field in ("reasoning", "reasoning_content"):
        value = getattr(obj, field, None)
        if value:
            return str(value)
    return ""


# Fallback rates for a caller that names none. Per-provider rates live in
# providers.py; these exist so the function is still callable without a
# Settings. Deliberately an ESTIMATE, not a promise - it exists so a runaway
# loop trips the budget guard, not for billing.
_COST_IN_PER_1M = DEFAULT_COST_IN
_COST_OUT_PER_1M = DEFAULT_COST_OUT


def _estimate_cost(tokens_in: int, tokens_out: int, *,
                   cost_in_per_1m: float = _COST_IN_PER_1M,
                   cost_out_per_1m: float = _COST_OUT_PER_1M) -> float:
    return round(
        (tokens_in / 1_000_000) * cost_in_per_1m
        + (tokens_out / 1_000_000) * cost_out_per_1m,
        6,
    )


# ---------------------------------------------------------------------------
# Fake client
# ---------------------------------------------------------------------------


class FakeLLM:
    """
    A scripted stand-in for a real model. No network, no key, no cost.

    Two modes:

    * scripted  - you hand it a list of replies; it returns them in order.
                  Use this in tests to force an exact path through the
                  graph ("critic says REVISE twice, then APPROVE").

    * heuristic - no script given, so it synthesises plausible-looking
                  proposer/critic output based on the last message. Use
                  this to click around the Streamlit UI before you have
                  funded an API account.

    It records every call in `.calls`, which makes assertions like
    "the proposer was called exactly 3 times" trivial to write.
    """

    def __init__(self, script: Sequence[str] | None = None, latency_s: float = 0.0):
        self.script = list(script) if script else []
        self.latency_s = latency_s
        self.calls: list[dict[str, Any]] = []
        self._critic_calls = 0

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        agent: str = "",
        on_token: TokenSink | None = None,
    ) -> LLMResponse:
        self.calls.append({"model": model, "messages": list(messages), "agent": agent})

        if self.latency_s:
            time.sleep(self.latency_s)

        if self.script:
            text = self.script.pop(0)
        else:
            text = self._improvise(messages, model, agent)

        # Emit the text as pseudo-tokens so the streaming UI can be
        # developed and demonstrated with no model running at all. The
        # fake-first principle applies to the UI layer too: if live output
        # only works against a real model, you cannot debug the renderer
        # separately from the inference.
        if on_token is not None:
            for word in text.split(" "):
                on_token(agent, word + " ")

        return LLMResponse(
            text=text,
            model=model,
            tokens_in=sum(len(m["content"]) // 4 for m in messages),
            tokens_out=len(text) // 4,
            latency_s=self.latency_s,
            ttft_s=self.latency_s,
            tokens_per_s=0.0,
            cost_usd=0.0,
        )

    @staticmethod
    def _is_critic(system: str, model: str) -> bool:
        """
        Identify the calling agent.

        NOTE THE BUG THIS REPLACES. The first version asked
        `if "critic" in system.lower()`. But the PROPOSER's system prompt
        ends with "A Critic will attack your answer." - so the proposer
        matched, and the fake proposer started emitting verdicts.

        Lesson: never infer identity from an incidental substring. The
        word "critic" appearing somewhere in a prompt is content, not an
        identity marker. Match on a deliberate, unambiguous signal - here,
        the opening declaration of the system prompt, or the model name.
        """
        if FakeLLM._is_arbiter(system, model):
            return False
        if "critic" in model.lower():
            return True
        return system.strip().upper().startswith("YOU ARE THE CRITIC")

    @staticmethod
    def _is_arbiter(system: str, model: str) -> bool:
        """
        Checked BEFORE _is_critic, and that ordering is not cosmetic.

        The Arbiter defaults to reusing the CRITIC'S MODEL NAME (it runs
        once, so it does not warrant its own model by default). Under the
        old check, `"critic" in model` would have matched, and the fake
        Arbiter would have emitted `VERDICT: REVISE` instead of a ruling -
        the same class of misidentification bug as before, arriving by a
        new route.

        The lesson generalises: role detection based on overlapping
        signals needs an explicit priority order, and the most specific
        test must run first.
        """
        return system.strip().upper().startswith("YOU ARE THE ARBITER")

    def _improvise(
        self, messages: Sequence[Message], model: str, agent: str = ""
    ) -> str:
        """
        The declared `agent` role wins whenever it is supplied; the prompt
        sniffing below survives only as a fallback for callers that do not
        declare one (and for the tests that pin that behaviour).

        Every role-detection bug in this project's history came from
        guessing identity out of prompt text. The fix is not a smarter
        guess - it is to stop guessing. Prefer what the caller told you.
        """
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        last = messages[-1]["content"] if messages else ""

        if agent == "arbiter" or (not agent and FakeLLM._is_arbiter(system, model)):
            return (
                "RULING:\n"
                "- Scope objection: the Critic was right; the answer overreached.\n"
                "- Missing example: pedantic, the Proposer's position holds.\n\n"
                "FINAL ANSWER:\n"
                "[FakeLLM arbitrated answer] The defensible position, with the "
                "Critic's valid scope objection folded in and the pedantic "
                "objection discarded.\n\n"
                "REMAINING UNCERTAINTY:\n"
                "This ruling is synthetic; no model actually reasoned about it."
            )

        if agent == "critic" or (not agent and FakeLLM._is_critic(system, model)):
            # Approve on the SECOND look, so a default fake run exercises
            # the interesting path (revise -> revise -> approve) and then
            # terminates via the happy path rather than the iteration cap.
            #
            # This uses an explicit call counter rather than trying to
            # infer "have we revised yet?" from the prompt text. An earlier
            # version searched the critic's own prompt for the word
            # "REVISION" - which the critic never sees, since only the
            # proposer is told which revision it is on. That fake critic
            # would have revised forever. Count what you control; do not
            # infer what you can measure.
            self._critic_calls += 1
            if self._critic_calls >= 2:
                return (
                    "VERDICT: APPROVE\n"
                    "REASONS:\n"
                    "- The revision addressed the scope problem directly.\n"
                    "- Claims are now hedged appropriately.\n"
                )
            return (
                "VERDICT: REVISE\n"
                "REASONS:\n"
                "- The answer asserts more certainty than the evidence supports.\n"
                "- It does not address the strongest counter-argument.\n"
                "- No concrete example is given.\n"
            )

        return (
            f"[FakeLLM answer]\n\nOn the question of: {last[:160]}\n\n"
            "This is placeholder output produced without any network call. "
            "Its purpose is to prove the orchestration graph routes, loops, "
            "and terminates correctly. Swap in a real provider to judge "
            "answer quality."
        )


def build_llm(settings: Settings) -> LLM:
    """Factory. The graph asks for an LLM and does not care which it gets."""
    if settings.is_fake:
        return FakeLLM()
    return OpenAICompatibleLLM(settings)
