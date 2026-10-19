"""
Token usage and cost accounting for LLM calls.

Every call to the language model, whether a direct call or a step inside an
agent loop, reports how many tokens it used. Those counts are stored in the
audit log alongside a cost computed at the moment of the call. Storing the cost
rather than recomputing it later means a future price change never rewrites
what past runs actually cost.

Prices are per million tokens and configurable, because they depend on the
model in GROQ_MODEL. The defaults are Groq's published rates for
openai/gpt-oss-120b. Check https://groq.com/pricing when changing the model.

Groq bills cached prompt tokens at half the input rate, so cached tokens are
counted separately when the provider reports them.

This data is for the operator. It is exposed only through admin-key routes and
the Streamlit dashboard, never in the user-facing app.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()


def _price(name: str, default: str) -> float:
    try:
        value = float(os.getenv(name, default))
    except ValueError:
        value = float(default)
    return max(0.0, value)


INPUT_PRICE_PER_MILLION = _price("LLM_INPUT_PRICE_PER_MILLION", "0.15")
OUTPUT_PRICE_PER_MILLION = _price("LLM_OUTPUT_PRICE_PER_MILLION", "0.60")
CACHED_INPUT_DISCOUNT = 0.5

# Groq's published rates (USD per million tokens, input and output) for models
# a single agent can be moved to, e.g. PREDICTOR_MODEL. GROQ_MODEL itself is
# priced from the two settings above, so its rate stays configurable. A model
# not listed here is priced like GROQ_MODEL. Check https://groq.com/pricing.
KNOWN_PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}


def prices_for(model: Optional[str] = None) -> tuple:
    """(input, output) USD per million tokens for the model that made a call."""
    if model and model != os.getenv("GROQ_MODEL") and model in KNOWN_PRICES:
        return KNOWN_PRICES[model]
    return INPUT_PRICE_PER_MILLION, OUTPUT_PRICE_PER_MILLION


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    model: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.prompt_tokens, self.completion_tokens, self.cached_tokens,
                        self.model)

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cached_tokens + other.cached_tokens,
            self.model or other.model,
        )


def cost_usd(prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0,
             model: Optional[str] = None) -> float:
    """Cost of one call in US dollars, at the rate of the model that made it."""
    input_price, output_price = prices_for(model)
    cached = max(0, min(cached_tokens, prompt_tokens))
    uncached = prompt_tokens - cached
    dollars = (
        uncached * input_price
        + cached * input_price * CACHED_INPUT_DISCOUNT
        + completion_tokens * output_price
    ) / 1_000_000
    return round(dollars, 8)


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def usage_from_openai(usage: Any, model: Optional[str] = None) -> Optional[Usage]:
    """Read the `usage` block of an OpenAI-compatible chat completion."""
    if usage is None:
        return None
    details = _get(usage, "prompt_tokens_details")
    return Usage(
        prompt_tokens=_int(_get(usage, "prompt_tokens")),
        completion_tokens=_int(_get(usage, "completion_tokens")),
        cached_tokens=_int(_get(details, "cached_tokens")),
        model=model,
    )


def usage_from_messages(messages: list) -> Optional[Usage]:
    """Sum the usage LangChain attaches to each AI message in an agent run.

    Each step of an agent loop is a separate model call, and each one resends
    the conversation so far. Summing per-message usage gives the true total.
    """
    total = None
    for message in messages or []:
        meta = getattr(message, "usage_metadata", None)
        if not meta:
            continue
        details = _get(meta, "input_token_details")
        step = Usage(
            prompt_tokens=_int(_get(meta, "input_tokens")),
            completion_tokens=_int(_get(meta, "output_tokens")),
            cached_tokens=_int(_get(details, "cache_read")),
        )
        total = step if total is None else total + step
    return total
