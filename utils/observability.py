"""
Observability – Prometheus metrics and structured logging setup.
"""

import logging
import sys

from prometheus_fastapi_instrumentator import Instrumentator


def setup_logging():
    """Configure structured logging for all modules."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def setup_prometheus(app):
    """Attach Prometheus metrics instrumentation to the FastAPI app.

    Exposes /metrics endpoint with:
    - http_requests_total (counter)
    - http_request_duration_seconds (histogram)
    - http_request_size_bytes (summary)
    - http_response_size_bytes (summary)
    """
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/health", "/docs", "/openapi.json"],
    )
    instrumentator.instrument(app)

    # /metrics lists every route with its traffic and latency, which maps the
    # API for anyone who can reach it. It requires the API key by default;
    # Prometheus can send one with `authorization: credentials`. Set
    # METRICS_PUBLIC=1 only when the endpoint sits on a private network.
    import os
    from fastapi import Depends
    from utils.auth import verify_api_key

    public = os.getenv("METRICS_PUBLIC", "0").strip().lower() in {"1", "true", "yes"}
    dependencies = [] if public else [Depends(verify_api_key)]

    instrumentator.expose(
        app,
        endpoint="/metrics",
        include_in_schema=True,
        tags=["Observability"],
        dependencies=dependencies,
    )
