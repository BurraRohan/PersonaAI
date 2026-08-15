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