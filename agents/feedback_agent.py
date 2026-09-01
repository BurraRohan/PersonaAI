"""
Feedback Agent – ReAct agent that analyses engagement history and recommends
strategy changes.

The statistics tool does real arithmetic in Python rather than asking the model
to compute averages, so the numbers the agent reasons over are correct.
"""

import json
import logging

from fastapi import HTTPException
from sqlalchemy.orm import Session
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from database.models import BrandProfile, Post
from schemas.schemas import FeedbackRequest, FeedbackResponse
from services.llm_service import generate_feedback
from utils.agent_runtime import build_agent_llm, run_agent, extract_trace, find_tool_result
from utils.config import AGENT_MODE

logger = logging.getLogger(__name__)


def _build_engagement_history(posts) -> list:
    """Flatten posts and their engagement rows into a list of dicts."""
    history = []
    for post in posts:
        for eng in post.engagements:
            history.append({
                "post_id": post.id,
                "topic": post.topic,
                "likes": eng.likes,
                "comments": eng.comments,
                "shares": eng.shares,
                "created_at": eng.created_at.isoformat() if eng.created_at else None,
            })
    return history


def _load_history(user_id: int) -> list:
    from database.db import SessionLocal
    db = SessionLocal()
    try:
        posts = (
            db.query(Post)
            .filter(Post.user_id == user_id)
            .order_by(Post.created_at.desc())
            .all()
        )
        return _build_engagement_history(posts)
    finally:
        db.close()


# ── Tools ───────────────────────────────────────────────────────

@tool
def fetch_engagement_history(user_id: int) -> str:
    """Fetch every logged engagement record for a user's posts.

    Returns JSON with an "entries" count and a "history" list of records
    containing topic, likes, comments and shares.
    """
    history = _load_history(user_id)
    return json.dumps({"entries": len(history), "history": history})


@tool
def compute_engagement_stats(user_id: int) -> str:
    """Compute averages and identify the best and worst performing posts.

    Returns JSON with avg_likes, avg_comments, avg_shares, and the topic and
    total engagement of the best and worst post. Computed in Python, so these
    numbers are exact.
    """
    history = _load_history(user_id)
    if not history:
        return json.dumps({"error": "No engagement data logged for this user."})

    total = len(history)

    def score(entry):
        return entry["likes"] + entry["comments"] + entry["shares"]

    best = max(history, key=score)
    worst = min(history, key=score)

    return json.dumps({
        "entries": total,
        "avg_likes": round(sum(h["likes"] for h in history) / total, 1),
        "avg_comments": round(sum(h["comments"] for h in history) / total, 1),
        "avg_shares": round(sum(h["shares"] for h in history) / total, 1),
        "best_post": {"topic": best["topic"], "total_engagement": score(best)},
        "worst_post": {"topic": worst["topic"], "total_engagement": score(worst)},
    })


@tool
def generate_strategy_feedback(user_id: int, brand_summary: str) -> str:
    """Produce the final written feedback for a user from their engagement history.

    Returns JSON with performance_summary and improvement_recommendation.
    """
    history = _load_history(user_id)
    if not history:
        return json.dumps({"error": "No engagement data logged for this user."})
    result = generate_feedback(history, brand_summary)
    return json.dumps(result)


FEEDBACK_AGENT_TOOLS = [
    fetch_engagement_history,
    compute_engagement_stats,
    generate_strategy_feedback,
]

FEEDBACK_AGENT_PROMPT = """You are a LinkedIn personal-branding coach agent.

Analyse the user's engagement data and produce strategic feedback. Call one
tool at a time:

1. Call fetch_engagement_history with the user id.
2. Call compute_engagement_stats with the user id to get exact averages and the
   best and worst performing posts.
3. Call generate_strategy_feedback with the user id and the brand summary you
   were given.
4. Reply with a one-sentence confirmation.

Never estimate averages yourself. Always take them from compute_engagement_stats."""


def get_feedback_agent():
    """Build the compiled ReAct feedback agent."""
    return create_react_agent(
        build_agent_llm(),
        FEEDBACK_AGENT_TOOLS,
        prompt=FEEDBACK_AGENT_PROMPT,
    )


# ── Public entry point (used by the API) ────────────────────────

def get_feedback(db: Session, request: FeedbackRequest) -> FeedbackResponse:
    """Generate feedback for a user by running the ReAct agent."""
    profile = db.query(BrandProfile).filter(BrandProfile.id == request.user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Brand profile not found.")

    # Only approved posts count. Rejected drafts were deliberately not used,
    # and pending ones have not been reviewed yet, so neither should shape
    # strategy advice.
    posts = (
        db.query(Post)
        .filter(Post.user_id == request.user_id, Post.status == "approved")
        .all()
    )
    if not posts:
        raise HTTPException(
            status_code=400,
            detail=(
                "No approved posts found for this user. Generate content via "
                "POST /generate, then approve it in the Evaluate tab."
            ),
        )

    history = _build_engagement_history(posts)
    if not history:
        raise HTTPException(
            status_code=400,
            detail="No engagement data found. Log engagement first via POST /engagement.",
        )

    history.sort(key=lambda h: h["created_at"] or "", reverse=True)
    brand_summary = f"{profile.name} – {profile.role} in {profile.industry}. Tone: {profile.tone}"

    if not AGENT_MODE:
        logger.info("AGENT_MODE=0 — direct path for user_id=%d", request.user_id)
        result = generate_feedback(history, brand_summary, db=db)
        return FeedbackResponse(
            user_id=request.user_id,
            total_posts=len(posts),
            performance_summary=result.get("performance_summary", ""),
            improvement_recommendation=result.get("improvement_recommendation", ""),
            execution_mode="direct",
            agent_trace=[],
        )

    logger.info("Running feedback agent for user_id=%d", request.user_id)

    instruction = (
        f"Analyse engagement and give feedback for brand profile id {request.user_id}.\n"
        f"brand_summary: {brand_summary}"
    )

    trace: list = []
    try:
        messages = run_agent(get_feedback_agent(), instruction)
        trace = extract_trace(messages)

        summary = find_tool_result(messages, "generate_strategy_feedback", "performance_summary")
        recommendation = find_tool_result(
            messages, "generate_strategy_feedback", "improvement_recommendation"
        )

        if summary and recommendation:
            return FeedbackResponse(
                user_id=request.user_id,
                total_posts=len(posts),
                performance_summary=summary,
                improvement_recommendation=recommendation,
                execution_mode="agent",
                agent_trace=trace,
            )

        logger.warning("Feedback agent returned no structured feedback; using direct path.")
    except Exception as exc:
        logger.warning("Feedback agent failed (%s); using direct path.", exc)
        trace.append({"step": len(trace) + 1, "action": "agent_error", "error": str(exc)[:300]})

    llm_result = generate_feedback(history, brand_summary, db=db)
    return FeedbackResponse(
        user_id=request.user_id,
        total_posts=len(posts),
        performance_summary=llm_result.get("performance_summary", ""),
        improvement_recommendation=llm_result.get("improvement_recommendation", ""),
        execution_mode="fallback_direct",
        agent_trace=trace,
    )