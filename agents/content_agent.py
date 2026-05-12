"""
Content Agent – LangChain agent with tools for LinkedIn post generation.
Uses ReAct pattern to fetch brand profile and generate aligned content.
"""

import json
import logging

from fastapi import HTTPException
from sqlalchemy.orm import Session
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from database.models import BrandProfile, Post
from schemas.schemas import GenerateRequest, GenerateResponse
from services.llm_service import generate_linkedin_post

logger = logging.getLogger(__name__)


# ── LangChain Tools ─────────────────────────────────────────────

@tool
def fetch_brand_profile(user_id: str) -> str:
    """Fetch the brand profile for a given user ID from the database.
    Returns the brand context as a JSON string."""
    from database.db import SessionLocal
    db = SessionLocal()
    try:
        profile = db.query(BrandProfile).filter(BrandProfile.id == int(user_id)).first()
        if not profile:
            return json.dumps({"error": "Brand profile not found"})
        return json.dumps({
            "tone": profile.tone,
            "content_themes": json.loads(profile.content_themes) if profile.content_themes else [],
            "positioning_summary": profile.positioning_summary,
            "do_guidelines": json.loads(profile.do_guidelines) if profile.do_guidelines else [],
            "dont_guidelines": json.loads(profile.dont_guidelines) if profile.dont_guidelines else [],
        })
    finally:
        db.close()


@tool
def create_linkedin_post(input_data: str) -> str:
    """Generate a LinkedIn post using the brand profile and topic.
    Input must be a JSON string with 'brand_profile' and 'topic' keys."""
    data = json.loads(input_data)
    result = generate_linkedin_post(data["brand_profile"], data["topic"])
    return json.dumps(result)


@tool
def save_post(post_data: str) -> str:
    """Save the generated post to the database.
    Input must be a JSON string with: user_id, topic, content, hashtags."""
    from database.db import SessionLocal
    data = json.loads(post_data)
    db = SessionLocal()
    try:
        post = Post(
            user_id=data["user_id"],
            topic=data["topic"],
            content=data["content"],
            hashtags=json.dumps(data.get("hashtags", [])),
        )
        db.add(post)
        db.commit()
        db.refresh(post)
        return json.dumps({"post_id": post.id, "status": "saved"})
    finally:
        db.close()


# ── Agent Setup ─────────────────────────────────────────────────

CONTENT_AGENT_PROMPT = """You are a LinkedIn content creation agent.

Your job is to generate a LinkedIn post aligned with the user's brand profile.
Follow this process:
1. Fetch the brand profile for the given user ID
2. Generate a LinkedIn post using the brand profile and topic
3. Save the post to the database
4. Return the generated post content"""


def get_content_agent():
    """Create and return the content generation agent."""
    import os
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
    )

    tools = [fetch_brand_profile, create_linkedin_post, save_post]
    agent = create_react_agent(llm, tools, prompt=CONTENT_AGENT_PROMPT)

    return agent


# ── Public Function (backward-compatible) ───────────────────────

def generate_post(db: Session, request: GenerateRequest) -> GenerateResponse:
    """Fetch brand profile, generate a LinkedIn post, and store it."""
    logger.info("Generating post for user_id=%d, topic=%s", request.user_id, request.topic)

    profile = db.query(BrandProfile).filter(BrandProfile.id == request.user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Brand profile not found. Create one via POST /brand first.")

    brand_context = {
        "tone": profile.tone,
        "content_themes": json.loads(profile.content_themes) if profile.content_themes else [],
        "positioning_summary": profile.positioning_summary,
        "do_guidelines": json.loads(profile.do_guidelines) if profile.do_guidelines else [],
        "dont_guidelines": json.loads(profile.dont_guidelines) if profile.dont_guidelines else [],
    }

    result = generate_linkedin_post(brand_context, request.topic, db=db)

    hashtags = result.get("suggested_hashtags", [])
    post = Post(
        user_id=request.user_id,
        topic=request.topic,
        content=result.get("post_content", ""),
        hashtags=json.dumps(hashtags),
    )
    db.add(post)
    db.commit()
    db.refresh(post)

    logger.info("Post created with id=%d", post.id)

    return GenerateResponse(
        post_id=post.id,
        post_content=post.content,
        suggested_hashtags=hashtags,
    )