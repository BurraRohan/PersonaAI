"""
Content Agent – ReAct agent that writes a LinkedIn post aligned to a brand profile.

Tools let the agent look up the brand, check what has already been posted so it
does not repeat a topic, and then generate and save the post.
"""

import json
import logging

from fastapi import HTTPException
from sqlalchemy.orm import Session
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from database.models import BrandProfile, Post
from schemas.schemas import GenerateRequest, GenerateResponse
from services.llm_service import generate_linkedin_post
from utils.agent_runtime import build_agent_llm, run_agent, extract_trace, find_tool_result
from utils.config import AGENT_MODE

logger = logging.getLogger(__name__)


def _brand_context(profile: BrandProfile) -> dict:
    """Shape a BrandProfile row into the dict the prompt builder expects."""
    return {
        "tone": profile.tone,
        "content_themes": json.loads(profile.content_themes) if profile.content_themes else [],
        "positioning_summary": profile.positioning_summary,
        "do_guidelines": json.loads(profile.do_guidelines) if profile.do_guidelines else [],
        "dont_guidelines": json.loads(profile.dont_guidelines) if profile.dont_guidelines else [],
    }


# ── Tools ───────────────────────────────────────────────────────

@tool
def fetch_brand_profile(user_id: int) -> str:
    """Fetch the stored brand profile for a brand profile id.

    Returns JSON with tone, content_themes, positioning_summary and guidelines.
    """
    from database.db import SessionLocal
    db = SessionLocal()
    try:
        profile = db.query(BrandProfile).filter(BrandProfile.id == user_id).first()
        if not profile:
            return json.dumps({"error": f"No brand profile with id {user_id}"})
        return json.dumps(_brand_context(profile))
    finally:
        db.close()


@tool
def list_recent_topics(user_id: int) -> str:
    """List the topics this user has already posted about, newest first.

    Use this to avoid repeating a topic that was covered recently.
    """
    from database.db import SessionLocal
    db = SessionLocal()
    try:
        posts = (
            db.query(Post)
            .filter(Post.user_id == user_id)
            .order_by(Post.created_at.desc())
            .limit(10)
            .all()
        )
        return json.dumps({
            "count": len(posts),
            "topics": [p.topic for p in posts],
        })
    finally:
        db.close()


@tool
def create_and_save_post(user_id: int, topic: str) -> str:
    """Write a LinkedIn post on the given topic, aligned to the user's brand, and save it.

    Returns JSON with post_id, a short preview, and the suggested hashtags.
    """
    from database.db import SessionLocal
    db = SessionLocal()
    try:
        profile = db.query(BrandProfile).filter(BrandProfile.id == user_id).first()
        if not profile:
            return json.dumps({"error": f"No brand profile with id {user_id}"})

        result = generate_linkedin_post(_brand_context(profile), topic, db=db)
        hashtags = result.get("suggested_hashtags", [])

        post = Post(
            user_id=user_id,
            topic=topic,
            content=result.get("post_content", ""),
            hashtags=json.dumps(hashtags),
        )
        db.add(post)
        db.commit()
        db.refresh(post)

        return json.dumps({
            "post_id": post.id,
            "preview": post.content[:200],
            "suggested_hashtags": hashtags,
        })
    finally:
        db.close()


CONTENT_AGENT_TOOLS = [fetch_brand_profile, list_recent_topics, create_and_save_post]

CONTENT_AGENT_PROMPT = """You are a LinkedIn content creation agent.

Your job is to publish one post that fits the user's brand. Call one tool at a time:

1. Call fetch_brand_profile with the user id, so you know the tone and themes.
2. Call list_recent_topics with the user id, to see what has already been covered.
3. Call create_and_save_post with the user id and the topic you were given.
4. Reply with a one-sentence confirmation that includes the saved post id.

If list_recent_topics shows the exact same topic was already posted, still write
the post, but mention the overlap in your final reply."""


def get_content_agent():
    """Build the compiled ReAct content agent."""
    return create_react_agent(
        build_agent_llm(),
        CONTENT_AGENT_TOOLS,
        prompt=CONTENT_AGENT_PROMPT,
    )


# ── Deterministic fallback ──────────────────────────────────────

def _generate_post_directly(db: Session, request: GenerateRequest) -> Post:
    profile = db.query(BrandProfile).filter(BrandProfile.id == request.user_id).first()
    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Brand profile not found. Create one via POST /brand first.",
        )

    result = generate_linkedin_post(_brand_context(profile), request.topic, db=db)
    post = Post(
        user_id=request.user_id,
        topic=request.topic,
        content=result.get("post_content", ""),
        hashtags=json.dumps(result.get("suggested_hashtags", [])),
    )
    db.add(post)
    db.commit()
    db.refresh(post)
    return post


def _to_response(post: Post, trace: list, mode: str) -> GenerateResponse:
    try:
        hashtags = json.loads(post.hashtags) if post.hashtags else []
    except json.JSONDecodeError:
        hashtags = []
    return GenerateResponse(
        post_id=post.id,
        post_content=post.content,
        suggested_hashtags=hashtags,
        execution_mode=mode,
        agent_trace=trace,
    )


# ── Public entry point (used by the API) ────────────────────────

def generate_post(db: Session, request: GenerateRequest) -> GenerateResponse:
    """Generate a LinkedIn post by running the ReAct agent."""
    profile = db.query(BrandProfile).filter(BrandProfile.id == request.user_id).first()
    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Brand profile not found. Create one via POST /brand first.",
        )

    if not AGENT_MODE:
        logger.info("AGENT_MODE=0 — direct path for user_id=%d", request.user_id)
        post = _generate_post_directly(db, request)
        return _to_response(post, [], "direct")

    logger.info("Running content agent for user_id=%d", request.user_id)

    instruction = (
        f"Write and save a LinkedIn post for brand profile id {request.user_id}.\n"
        f"topic: {request.topic}"
    )

    trace: list = []
    try:
        messages = run_agent(get_content_agent(), instruction)
        trace = extract_trace(messages)
        post_id = find_tool_result(messages, "create_and_save_post", "post_id")

        if post_id:
            post = db.query(Post).filter(Post.id == post_id).first()
            if post:
                logger.info("Content agent saved post id=%d", post.id)
                return _to_response(post, trace, "agent")

        logger.warning("Content agent did not persist a post; using direct path.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Content agent failed (%s); using direct path.", exc)
        trace.append({"step": len(trace) + 1, "action": "agent_error", "error": str(exc)[:300]})

    post = _generate_post_directly(db, request)
    return _to_response(post, trace, "fallback_direct")
