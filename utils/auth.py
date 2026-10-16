"""
API key authentication.

Two keys, two audiences:

  API_KEY        The app's own key. Every user-facing route requires it, and the
                 app refuses to start without it rather than falling back to a
                 value that is public in the source code.

  ADMIN_API_KEY  The operator's key, for routes that expose token usage and
                 cost. Optional: if it is not set, those routes are switched
                 off instead of falling back to API_KEY, so holding the app key
                 never grants access to operator data.
"""

import logging
import os
import secrets

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError(
        "API_KEY is not set. Copy .env.template to .env and set API_KEY before starting. "
        "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
    )


# Registering the scheme is what puts the "Authorize" button in Swagger at /docs.
security = HTTPBearer(
    auto_error=False,
    description="Paste the API_KEY value from your .env file.",
)


def check_key(candidate: str) -> bool:
    """Constant-time comparison, so timing does not leak the key."""
    return secrets.compare_digest(candidate, API_KEY)


async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> None:
    """FastAPI dependency: require a valid `Authorization: Bearer <key>` header."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format. Use: Bearer <api-key>",
        )

    if not check_key(credentials.credentials):
        client = request.client.host if request.client else "unknown"
        logger.warning("Invalid API key attempt from %s", client)
        raise HTTPException(status_code=403, detail="Invalid API key")


# ── Operator (admin) access ─────────────────────────────────────

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY") or None

# Reusing the app key would let every user of the UI read operator data, so a
# reused key is treated as unset: the operator routes stay off until it differs.
if ADMIN_API_KEY and secrets.compare_digest(ADMIN_API_KEY, API_KEY):
    logger.warning(
        "ADMIN_API_KEY is the same as API_KEY. Operator routes are disabled until "
        "ADMIN_API_KEY is set to a different value."
    )
    ADMIN_API_KEY = None

# A separate scheme gives Swagger's Authorize dialog its own box for this key,
# and Swagger then sends the right key to each route automatically.
admin_security = HTTPBearer(
    auto_error=False,
    scheme_name="AdminKey",
    description="Paste the ADMIN_API_KEY value from your .env file. Operator routes only.",
)


async def verify_admin_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(admin_security),
) -> None:
    """FastAPI dependency for operator routes: usage, cost and audit data."""
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Operator routes are disabled. Set ADMIN_API_KEY in .env to enable them.",
        )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Admin key required")
    if not secrets.compare_digest(credentials.credentials, ADMIN_API_KEY):
        client = request.client.host if request.client else "unknown"
        logger.warning("Invalid admin key attempt from %s", client)
        raise HTTPException(status_code=403, detail="Invalid admin key")
