"""
Per-request context for cost attribution.

One user action can trigger several LLM calls: an agent loop makes a call per
step, and its tools make calls of their own. Each request gets an id, and every
audit-log row written while serving it carries that id and the endpoint, so the
total cost of an action can be summed.

Context variables follow the request into FastAPI's thread pool and into the
agent's tool calls, so nothing has to be passed through function arguments.
"""

import re
import uuid
from contextvars import ContextVar
from typing import Optional

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
endpoint_var: ContextVar[Optional[str]] = ContextVar("endpoint", default=None)

_ID_SEGMENT = re.compile(r"/\d+(?=/|$)")


def normalise_path(method: str, path: str) -> str:
    """'/posts/12/review' -> 'POST /posts/{id}/review', so ids group together."""
    return f"{method} {_ID_SEGMENT.sub('/{id}', path)}"[:100]


class RequestContextMiddleware:
    """Plain ASGI middleware: tag each HTTP request with an id and endpoint."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        rid = request_id_var.set(str(uuid.uuid4()))
        ep = endpoint_var.set(normalise_path(scope.get("method", ""), scope.get("path", "")))
        try:
            await self.app(scope, receive, send)
        finally:
            request_id_var.reset(rid)
            endpoint_var.reset(ep)
