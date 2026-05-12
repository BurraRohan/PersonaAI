"""
Feedback Agent – LangChain agent that analyses engagement history
and produces strategic recommendations using tools.
"""

import json
import logging

from fastapi import HTTPException
from sqlalchemy.orm import Session
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from database.models import BrandProfile, Post, Engagement
from schemas.schemas import FeedbackRequest, FeedbackResponse
from services.llm_service import generate_feedback

logger = logging.getLogger(__name__)


# ── LangChain Tools ─────────────────────────────────────────────

@tool
def fetch_engagement_history(user_id: str) -> str:
    """Fetch engagement history for all posts by a given user ID.
    Returns a JSON list of engagement records."""
    from database.db import SessionLocal
    db = SessionLocal()
    try:
        posts = (
            db.query(Post)
            .filter(Post.user_id == int(user_id))
            .order_by(Post.created_at.desc())
            .all()
        )
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
        return json.dumps({"total_posts": len(posts), "history": history})
    finally:
        db.close()


@tool
def compute_engagement_stats(history_json: str) -> str:
    """Compute average engagement statistics from engagement history.
    Input must be a JSON string with a 'history' list."""
    data = json.loads(history_json)
    history = data.get("history", [])

    if not history:
        return json.dumps({"error": "No engagement data available"})

    total = len(history)
    avg_likes = sum(h["likes"] for h in history) / total
    avg_comments = sum(h["comments"] for h in history) / total
    avg_shares = sum(h["shares"] for h in history) / total

    best_post = max(history, key=lambda h: h["likes"] + h["comments"] + h["shares"])
    worst_post = min(history, key=lambda h: h["likes"] + h["comments"] + h["shares"])

    return json.dumps({
        "total_entries": total,
        "avg_likes": round(avg_likes, 1),
        "avg_comments": round(avg_comments, 1),
        "avg_shares": round(avg_shares, 1),
        "best_post_topic": best_post["topic"],
        "worst_post_topic": worst_post["topic"],
    })


@tool
def generate_strategy_feedback(input_data: str) -> str:
    """Generate AI-powered strategy feedback using engagement history and brand context.
    Input must be a JSON string with 'engagement_history' and 'brand_summary' keys."""
    data = json.loads(input_data)
    result = generate_feedback(data["engagement_history"], data["brand_summary"])
    return json.dumps(result)


# ── Agent Setup ─────────────────────────────────────────────────

FEEDBACK_AGENT_PROMPT = """You are a LinkedIn personal branding coach agent.

Your job is to analyse a user's engagement data and provide strategic feedback.
Follow this process:
1. Fetch the engagement history for the user
2. Compute engagement statistics to understand performance
3. Generate strategic feedback using the data and brand context
4. Return a comprehensive feedback summary"""


def get_feedback_agent():
    """Create and return the feedback agent."""
    import os
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
    )

    tools = [fetch_engagement_history, compute_engagement_stats, generate_strategy_feedback]
    agent = create_react_agent(llm, tools, prompt=FEEDBACK_AGENT_PROMPT)

    return agent


# ── Helper ──────────────────────────────────────────────────────

def _build_engagement_history(posts) -> list[dict]:
    """Flatten posts + engagements into a list of dicts for analysis."""
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


# ── Public Function (backward-compatible) ───────────────────────

def get_feedback(db: Session, request: FeedbackRequest) -> FeedbackResponse:
    """Generate feedback for a user based on their engagement history."""
    logger.info("Generating feedback for user_id=%d", request.user_id)

    profile = db.query(BrandProfile).filter(BrandProfile.id == request.user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Brand profile not found.")

    posts = (
        db.query(Post)
        .filter(Post.user_id == request.user_id)
        .order_by(Post.created_at.desc())
        .all()
    )

    if not posts:
        raise HTTPException(
            status_code=400,
            detail="No posts found for this user. Generate content first via POST /generate.",
        )

    history = _build_engagement_history(posts)

    if not history:
        raise HTTPException(
            status_code=400,
            detail="No engagement data found. Log engagement first via POST /engagement.",
        )

    history.sort(key=lambda h: h["created_at"] or "", reverse=True)

    brand_summary = f"{profile.name} – {profile.role} in {profile.industry}. Tone: {profile.tone}"
    llm_result = generate_feedback(history, brand_summary, db=db)

    return FeedbackResponse(
        user_id=request.user_id,
        total_posts=len(posts),
        performance_summary=llm_result.get("performance_summary", ""),
        improvement_recommendation=llm_result.get("improvement_recommendation", ""),
    )