"""
Runtime configuration.

AGENT_MODE controls how /brand, /generate and /feedback execute:

  AGENT_MODE=1  (default)  Run the ReAct agents. The model chooses its own
                           tools, and the response includes agent_trace.
                           Costs roughly 3-6 Groq calls per request.

  AGENT_MODE=0             Skip the agent loop and make one direct LLM call.
                           Same endpoints, same output fields, agent_trace
                           comes back empty. Much faster and much easier on
                           the Groq rate limit.

GROQ_MODEL sets the model used by every LLM call and every agent. It is
required: the model name is never written into the source.

Set both in .env. Nothing else in the project needs to change.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# Default is on: the agent path is the one worth demonstrating.
AGENT_MODE = _as_bool(os.getenv("AGENT_MODE", "1"))

# The Groq model used by every LLM call and every agent.
#
# Required, with no default: the model name lives only in .env, never in the
# source. A provider deprecation is then a one-line config change.
#
# Deliberately not auto-selected either — tool-calling reliability varies
# between models, and the audit log records the model per call, so past runs
# stay reproducible.
#
# Check https://console.groq.com/docs/deprecations for current models.
GROQ_MODEL = os.getenv("GROQ_MODEL")

if not GROQ_MODEL:
    raise RuntimeError(
        "GROQ_MODEL is not set. Copy .env.template to .env and set it, "
        "e.g. GROQ_MODEL=openai/gpt-oss-120b — see "
        "https://console.groq.com/docs/deprecations for current models."
    )


# Output token cap for each structured LLM call.
#
# gpt-oss is a reasoning model: its hidden reasoning counts against this cap
# as well as the visible answer. At 2048, the longer content prompts (v4, v5)
# often spent the whole budget reasoning and returned nothing, which Groq
# reports as json_validate_failed. On Groq's free tier the prompt plus this cap
# must also fit inside the 8K tokens-per-minute limit, so it is not set higher.
def _as_int(value, default: int, low: int, high: int) -> int:
    try:
        return min(high, max(low, int(value)))
    except (TypeError, ValueError):
        return default


LLM_MAX_TOKENS = _as_int(os.getenv("LLM_MAX_TOKENS"), 4096, 256, 32768)

# How much a gpt-oss model reasons before answering: low, medium or high.
# Unset means Groq's default. Lower effort spends fewer output tokens, so it is
# cheaper and faster; whether quality holds is a question for evals/run_eval.
_effort = (os.getenv("LLM_REASONING_EFFORT") or "").strip().lower()
LLM_REASONING_EFFORT = _effort if _effort in {"low", "medium", "high"} else None


# Model for the engagement predictor only. Empty means GROQ_MODEL, like every
# other call. Scoring a draft is a narrower job than writing one, so a smaller,
# cheaper model may do it as well; evals/compare_predictor.py measures whether
# it does before you switch.
PREDICTOR_MODEL = (os.getenv("PREDICTOR_MODEL") or "").strip() or None
