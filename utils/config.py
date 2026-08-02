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

Set it in .env. Nothing else in the project needs to change.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# Default is on: the agent path is the one worth demonstrating.
AGENT_MODE = _as_bool(os.getenv("AGENT_MODE", "1"))
