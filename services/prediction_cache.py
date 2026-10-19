"""
Reuse an engagement prediction when nothing that shapes it has changed.

Scoring the same draft twice is common: a reviewer re-opens a post, scores it
again after reading the tips, or two people check the same draft. Each repeat
would otherwise be a fresh Groq call with a slightly different answer.

A prediction is reused only when all of these match the earlier call:

  * the draft text (trailing spaces and blank edges ignored, line breaks kept,
    because paragraph breaks change the readability score)
  * the brand profile fields the predictor reads: tone, themes, positioning
  * the profile's engagement history
  * the model and the predictor prompt version

Editing the profile, logging new engagement, switching PREDICTOR_MODEL or
activating another predictor prompt therefore always produces a fresh score.

Entries expire after PREDICTION_CACHE_HOURS (default 24). Set it to 0 to turn
caching off.

Only the /predict route uses this. The evaluation scripts call the predictor
directly, so they always measure the model rather than the cache.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

load_dotenv()

logger = logging.getLogger(__name__)


def _read_hours() -> float:
    try:
        value = float(os.getenv("PREDICTION_CACHE_HOURS", "24"))
    except ValueError:
        value = 24.0
    return max(0.0, value)


CACHE_HOURS = _read_hours()


def _normalise(text: str) -> str:
    return "\n".join(line.rstrip() for line in (text or "").strip().splitlines())


def predictor_setup(db: Session) -> tuple:
    """(model, prompt version) the predictor would use right now."""
    from services import llm_service

    model = llm_service.PREDICTOR_MODEL or llm_service.MODEL
    _, version = llm_service.get_active_prompt(db, "predictor")
    return model, version


def cache_key(draft: str, brand_context: dict, history: list,
              model: str, prompt_version: Optional[int]) -> str:
    payload = {
        "draft": _normalise(draft),
        "brand": {k: brand_context.get(k)
                  for k in ("tone", "content_themes", "positioning_summary")},
        "history": history,
        "model": model,
        "prompt_version": prompt_version,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cutoff() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=CACHE_HOURS)


def lookup(db: Session, key: str) -> Optional[dict]:
    """The stored prediction for this key, if caching is on and it is fresh."""
    if CACHE_HOURS <= 0:
        return None
    from database.models import PredictionCache

    try:
        row = (db.query(PredictionCache)
               .filter(PredictionCache.cache_key == key,
                       PredictionCache.created_at >= _cutoff())
               .first())
        if row is None:
            return None
        row.hit_count = (row.hit_count or 0) + 1
        row.last_hit_at = datetime.now(timezone.utc)
        db.commit()
        return json.loads(row.result_json)
    except Exception as exc:  # a cache problem must never break scoring
        logger.warning("Prediction cache lookup failed: %s", exc)
        db.rollback()
        return None


def store(db: Session, key: str, result: dict, model: str,
          prompt_version: Optional[int]) -> None:
    """Save a fresh prediction, with what it cost, and clear expired entries."""
    if CACHE_HOURS <= 0:
        return
    from database.models import AuditLog, PredictionCache
    from utils.request_context import request_id_var

    try:
        cost = None
        request_id = request_id_var.get()
        if request_id:
            cost = (db.query(AuditLog.cost_usd)
                    .filter(AuditLog.request_id == request_id,
                            AuditLog.agent_name == "predictor")
                    .order_by(AuditLog.id.desc())
                    .limit(1).scalar())

        db.query(PredictionCache).filter(PredictionCache.created_at < _cutoff()) \
            .delete(synchronize_session="fetch")
        # Replace an expired entry with the same key rather than collide with it.
        db.query(PredictionCache).filter(PredictionCache.cache_key == key) \
            .delete(synchronize_session="fetch")
        db.add(PredictionCache(cache_key=key, result_json=json.dumps(result),
                               model=model, prompt_version=prompt_version,
                               cost_usd=cost))
        db.commit()
    except IntegrityError:
        db.rollback()  # another request stored the same key first; that is fine
    except Exception as exc:
        logger.warning("Prediction cache store failed: %s", exc)
        db.rollback()


def stats(db: Session) -> dict:
    """Cache size, hits and money saved, for the operator."""
    from database.models import PredictionCache

    entries, hits, saved = db.query(
        func.count(PredictionCache.id),
        func.coalesce(func.sum(PredictionCache.hit_count), 0),
        func.coalesce(func.sum(PredictionCache.hit_count * PredictionCache.cost_usd), 0.0),
    ).one()
    return {
        "enabled": CACHE_HOURS > 0,
        "hours": CACHE_HOURS,
        "entries": int(entries or 0),
        "hits": int(hits or 0),
        "saved_usd": round(float(saved or 0.0), 6),
    }
