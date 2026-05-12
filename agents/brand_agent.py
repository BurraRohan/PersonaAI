"""
Brand Agent – LangChain agent with tools for brand profile creation.
Uses ReAct pattern: the agent reasons about what tools to call.
"""

import json
import logging

from sqlalchemy.orm import Session
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from database.models import BrandProfile
from schemas.schemas import BrandRequest, BrandResponse
from services.llm_service import generate_brand_profile

logger = logging.getLogger(__name__)


# ── LangChain Tools ─────────────────────────────────────────────

@tool
def check_existing_profile(name: str) -> str:
    """Check if a brand profile already exists for the given name.
    Returns existing profile info or 'not found'."""
    from database.db import SessionLocal
    db = SessionLocal()
    try:
        profile = db.query(BrandProfile).filter(BrandProfile.name == name).first()
        if profile:
            return json.dumps({
                "exists": True,
                "id": profile.id,
                "name": profile.name,
                "role": profile.role,
                "tone": profile.tone,
            })
        return json.dumps({"exists": False})
    finally:
        db.close()


@tool
def generate_brand_strategy(profile_data: str) -> str:
    """Generate a brand strategy using LLM given profile data as JSON string.
    Input must be a JSON string with: name, role, industry, goals, preferred_tone."""
    data = json.loads(profile_data)
    result = generate_brand_profile(
        name=data["name"],
        role=data["role"],
        industry=data["industry"],
        goals=data["goals"],
        preferred_tone=data["preferred_tone"],
    )
    return json.dumps(result)


@tool
def save_brand_profile(profile_json: str) -> str:
    """Save the brand profile to the database. Input is a JSON string with all profile fields."""
    from database.db import SessionLocal
    data = json.loads(profile_json)
    db = SessionLocal()
    try:
        profile = BrandProfile(
            name=data["name"],
            role=data["role"],
            industry=data["industry"],
            goals=data["goals"],
            preferred_tone=data["preferred_tone"],
            tone=data.get("tone", ""),
            content_themes=json.dumps(data.get("content_themes", [])),
            positioning_summary=data.get("positioning_summary", ""),
            do_guidelines=json.dumps(data.get("do_guidelines", [])),
            dont_guidelines=json.dumps(data.get("dont_guidelines", [])),
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
        return json.dumps({"id": profile.id, "status": "saved"})
    finally:
        db.close()


# ── Agent Setup ─────────────────────────────────────────────────

BRAND_AGENT_PROMPT = """You are a personal branding strategist agent.

Your job is to create a structured LinkedIn brand profile for a user.
Follow this process:
1. First check if a profile already exists for this person
2. If not, generate a brand strategy using their details
3. Save the generated profile to the database
4. Return the final profile information"""


def get_brand_agent():
    """Create and return the brand agent with tools."""
    import os
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
    )

    tools = [check_existing_profile, generate_brand_strategy, save_brand_profile]
    agent = create_react_agent(llm, tools, prompt=BRAND_AGENT_PROMPT)

    return agent


# ── Public Function (backward-compatible) ───────────────────────

def create_brand_profile(db: Session, request: BrandRequest) -> BrandResponse:
    """Create a brand profile via LLM and persist it."""
    logger.info("Creating brand profile for: %s", request.name)

    result = generate_brand_profile(
        name=request.name,
        role=request.role,
        industry=request.industry,
        goals=request.goals,
        preferred_tone=request.preferred_tone,
        db=db,
    )

    profile = BrandProfile(
        name=request.name,
        role=request.role,
        industry=request.industry,
        goals=request.goals,
        preferred_tone=request.preferred_tone,
        tone=result.get("tone", ""),
        content_themes=json.dumps(result.get("content_themes", [])),
        positioning_summary=result.get("positioning_summary", ""),
        do_guidelines=json.dumps(result.get("do_guidelines", [])),
        dont_guidelines=json.dumps(result.get("dont_guidelines", [])),
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)

    logger.info("Brand profile created with id=%d", profile.id)

    return BrandResponse(
        id=profile.id,
        name=profile.name,
        role=profile.role,
        industry=profile.industry,
        goals=profile.goals,
        preferred_tone=profile.preferred_tone,
        tone=profile.tone,
        content_themes=json.loads(profile.content_themes) if profile.content_themes else [],
        positioning_summary=profile.positioning_summary,
        do_guidelines=json.loads(profile.do_guidelines) if profile.do_guidelines else [],
        dont_guidelines=json.loads(profile.dont_guidelines) if profile.dont_guidelines else [],
        created_at=profile.created_at,
    )