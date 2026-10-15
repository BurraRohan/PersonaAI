"""
HTTP hardening applied to every response and request.

Security headers
    The UI stores the API key in localStorage, so a successful script injection
    would be able to read it. The Content-Security-Policy is the second line of
    defence behind output escaping: scripts may only come from this origin, and
    inline scripts, inline styles and framing are refused outright.

    Swagger (/docs) and ReDoc load their bundles from a CDN and use an inline
    bootstrap script, so they get a looser policy of their own. They contain no
    stored data, only the API schema.

Request size
    Field-level limits in the schemas reject oversized values, but only after
    the whole body has been read and parsed. The size cap stops a large body
    before any of that work happens.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

MAX_BODY_BYTES = 64 * 1024

APP_CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    "img-src 'self' data:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])

DOCS_CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    "img-src 'self' data: https://fastapi.tiangolo.com https://cdn.redoc.ly",
    "connect-src 'self'",
    "object-src 'none'",
    "frame-ancestors 'none'",
])

DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        path = request.url.path
        response.headers["Content-Security-Policy"] = (
            DOCS_CSP if path.startswith(DOCS_PATHS) else APP_CSP
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"

        # API responses carry profile data; keep them out of shared caches.
        if not path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store"

        return response


class BodySizeLimitMiddleware:
    """Plain ASGI middleware: counts body bytes as they arrive.

    Works for both declared Content-Length and chunked uploads, and never
    buffers more than the cap.
    """

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    return await _reject(send, 413, f"Request body exceeds {self.max_bytes // 1024} KB.")
            except ValueError:
                return await _reject(send, 400, "Invalid Content-Length.")

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await _reject(send, 413, f"Request body exceeds {self.max_bytes // 1024} KB.")


class _BodyTooLarge(Exception):
    pass


async def _reject(send, status: int, detail: str):
    response = JSONResponse(status_code=status, content={"detail": detail})
    await send({"type": "http.response.start", "status": status,
                "headers": [(k.encode(), v.encode()) for k, v in response.headers.items()]})
    await send({"type": "http.response.body", "body": response.body})
