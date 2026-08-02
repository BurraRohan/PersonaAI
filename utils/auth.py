"""
API key authentication.

The key is required at import time: a missing API_KEY stops the app from
starting rather than silently falling back to a value that is public in the
source code.
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
