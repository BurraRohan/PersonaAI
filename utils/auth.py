"""
API Key Authentication – simple bearer token auth for all endpoints.
"""

import os
import logging

from dotenv import load_dotenv
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

load_dotenv()

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

API_KEY = os.getenv("API_KEY", "dev-key-change-me")


async def verify_api_key(request: Request) -> None:
    """Dependency that verifies the API key from the Authorization header.
    Skips auth for health check, metrics, docs, and static files."""

    # Skip auth for these paths
    skip_paths = ["/health", "/metrics", "/docs", "/openapi.json", "/redoc", "/"]
    if request.url.path in skip_paths or request.url.path.startswith("/static"):
        return

    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    # Expect "Bearer <key>"
    parts = auth_header.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization format. Use: Bearer <api-key>")

    if parts[1] != API_KEY:
        logger.warning("Invalid API key attempt from %s", request.client.host)
        raise HTTPException(status_code=403, detail="Invalid API key")
