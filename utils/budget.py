"""
Daily spending limit for LLM calls.

Cost is recorded for every call (see utils/costs.py). This module turns that
record into a control: once today's spend crosses a limit, the app spends less.

Two thresholds, both relative to DAILY_BUDGET_USD:

  soft limit  (BUDGET_SOFT_LIMIT_PERCENT, default 80%)
      Agent loops are paused. /brand, /generate and /feedback take the direct
      path instead: one LLM call rather than several, so the app keeps working
      at a fraction of the cost. Responses report execution_mode
      "budget_direct".

  hard limit  (100%)
      Routes that call the LLM return 429 until the day resets at midnight UTC.
      Everything else (dashboard, history, review, engagement) keeps working.

The message users see never mentions money or the limit. Spend figures stay
on the operator routes and the Streamlit dashboard.

Leave DAILY_BUDGET_USD empty or 0 to switch the guard off.

One request is checked before it runs, so the request that crosses the limit
still completes. The overshoot is at most one request's cost.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

load_dotenv()

logger = logging.getLogger(__name__)


def _read_budget() -> Optional[float]:
    raw = (os.getenv("DAILY_BUDGET_USD") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.warning("DAILY_BUDGET_USD=%r is not a number; budget guard is off.", raw)
        return None
    return value if value > 0 else None


def _read_soft_percent() -> int:
    try:
        value = int(os.getenv("BUDGET_SOFT_LIMIT_PERCENT", "80"))
    except ValueError:
        value = 80
    return min(100, max(1, value))


DAILY_BUDGET_USD = _read_budget()
SOFT_LIMIT_PERCENT = _read_soft_percent()

BLOCKED_MESSAGE = (
    "The daily AI usage limit has been reached. Generation resumes at "
    "midnight UTC; everything else keeps working."
)


def _day_start(now: datetime) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def spent_today(db: Optional[Session] = None) -> float:
    """Sum of recorded LLM cost since midnight UTC."""
    from database.db import SessionLocal
    from database.models import AuditLog

    own = db is None
    db = db or SessionLocal()
    try:
        since = _day_start(datetime.now(timezone.utc))
        total = (
            db.query(func.sum(AuditLog.cost_usd))
            .filter(AuditLog.created_at >= since)
            .scalar()
        )
        return float(total or 0.0)
    finally:
        if own:
            db.close()


def budget_status(db: Optional[Session] = None) -> dict:
    """Current state of the guard, for the operator."""
    now = datetime.now(timezone.utc)
    resets_at = (_day_start(now) + timedelta(days=1)).isoformat()

    if DAILY_BUDGET_USD is None:
        return {"enabled": False, "state": "off", "resets_at": resets_at}

    spent = spent_today(db)
    percent = spent / DAILY_BUDGET_USD * 100
    if percent >= 100:
        state = "blocked"
    elif percent >= SOFT_LIMIT_PERCENT:
        state = "agents_paused"
    else:
        state = "ok"

    return {
        "enabled": True,
        "state": state,
        "daily_limit_usd": DAILY_BUDGET_USD,
        "soft_limit_percent": SOFT_LIMIT_PERCENT,
        "spent_today_usd": round(spent, 6),
        "percent_used": round(percent, 1),
        "resets_at": resets_at,
    }


def agents_allowed() -> bool:
    """False once today's spend passes the soft limit."""
    if DAILY_BUDGET_USD is None:
        return True
    return budget_status()["state"] == "ok"


def enforce_budget() -> None:
    """FastAPI dependency for routes that call the LLM: 429 at the hard limit."""
    if DAILY_BUDGET_USD is None:
        return
    status = budget_status()
    if status["state"] != "blocked":
        return

    now = datetime.now(timezone.utc)
    retry_after = int((_day_start(now) + timedelta(days=1) - now).total_seconds()) + 1
    logger.warning("Daily budget reached (%.4f of %.4f USD); LLM route refused.",
                   status["spent_today_usd"], DAILY_BUDGET_USD)
    raise HTTPException(status_code=429, detail=BLOCKED_MESSAGE,
                        headers={"Retry-After": str(retry_after)})
