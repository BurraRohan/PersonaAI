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
    instrumentator.expose(app, endpoint="/metrics", include_in_schema=True, tags=["Observability"])
