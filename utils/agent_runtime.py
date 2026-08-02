"""
Agent runtime helpers.

The ReAct agents built with `create_react_agent` return a message list.
These helpers:
  * build the LLM used by every agent,
  * run an agent and capture which tools it decided to call,
  * pull structured results back out of ToolMessages.

Capturing the trace is what makes the agent layer demonstrable: the API can
return the exact sequence of tool calls the model chose, instead of us just
claiming tool use happened.
"""

import json
import logging
import os
from typing import Any, Optional

from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

AGENT_MODEL = "llama-3.3-70b-versatile"
AGENT_TEMPERATURE = 0.3
AGENT_RECURSION_LIMIT = 12


def build_agent_llm() -> ChatGroq:
    """Single place where the agent LLM is configured."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.template to .env and fill it in."
        )
    return ChatGroq(
        api_key=api_key,
        model_name=AGENT_MODEL,
        temperature=AGENT_TEMPERATURE,
    )


def _message_kind(message: Any) -> str:
    """Return 'ai' / 'tool' / 'human' for a LangChain message object."""
    explicit = getattr(message, "type", None)
    if explicit:
        return explicit
    return message.__class__.__name__.replace("Message", "").lower()


def extract_trace(messages: list) -> list[dict]:
    """Flatten an agent message list into a readable tool-call trace.

    Produces entries like:
        {"step": 1, "action": "tool_call",   "tool": "check_existing_profile", "args": {...}}
        {"step": 2, "action": "tool_result", "tool": "check_existing_profile", "result": "..."}
    """
    trace: list[dict] = []
    step = 0

    for message in messages:
        kind = _message_kind(message)

        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            step += 1
            trace.append({
                "step": step,
                "action": "tool_call",
                "tool": call.get("name"),
                "args": call.get("args", {}),
            })

        if kind == "tool":
            step += 1
            content = str(getattr(message, "content", ""))
            trace.append({
                "step": step,
                "action": "tool_result",
                "tool": getattr(message, "name", None),
                "result": content[:400],
            })

    return trace


def find_tool_result(messages: list, tool_name: str, key: str) -> Optional[Any]:
    """Search ToolMessages (most recent first) for `tool_name` and return `key`.

    Used to recover IDs the agent's tools wrote to the database, so the API
    response is built from ground truth rather than from the model's prose.
    """
    for message in reversed(messages):
        if _message_kind(message) != "tool":
            continue
        if getattr(message, "name", None) != tool_name:
            continue
        try:
            payload = json.loads(str(message.content))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and key in payload:
            return payload[key]
    return None


def run_agent(agent, instruction: str) -> list:
    """Invoke a compiled ReAct agent and return its message list."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": instruction}]},
        config={"recursion_limit": AGENT_RECURSION_LIMIT},
    )
    messages = result.get("messages", [])
    logger.info(
        "[agent] completed with %d messages, %d trace entries",
        len(messages), len(extract_trace(messages)),
    )
    return messages
