"""
Brand Agent – ReAct agent that builds a LinkedIn brand profile using tools.

The agent decides which tools to call. `create_brand_profile` is the entry
point used by the API and runs the agent for real; if the agent fails to
persist a profile, it falls back to a deterministic single-call path so the
endpoint never returns a 500 during a demo.
"""

import json
import logging

from sqlalchemy.orm import Session
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from database.models import BrandProfile
from schemas.schemas import BrandRequest, BrandResponse
from services.llm_service import generate_brand_profile
from utils.agent_runtime import build_agent_llm, run_agent, extract_trace, find_tool_result
from utils.config import AGENT_MODE

logger = logging.getLogger(__name__)


# ── Tools ───────────────────────────────────────────────────────

@tool
def check_existing_profile(name: str) -> str:
    """Check whether a brand profile already exists for the given person's name.

    Returns JSON with "exists": true/false, plus the profile id and role if found.
    """
    from database.db import SessionLocal
    db = SessionLocal()
    try:
        profile = (
            db.query(BrandProfile)
            .filter(BrandProfile.name == name)
            .order_by(BrandProfile.id.desc())
            .first()
        )
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
def generate_brand_strategy(
    name: str, role: str, industry: str, goals: str, preferred_tone: str
) -> str:
    """Generate a LinkedIn brand strategy for a professional.

    Returns JSON containing tone, content_themes, positioning_summary,
    do_guidelines and dont_guidelines. Pass this JSON straight to
    save_brand_profile as the strategy_json argument.
    """
    result = generate_brand_profile(
        name=name,
        role=role,
        industry=industry,
        goals=goals,
        preferred_tone=preferred_tone,
    )
    return json.dumps(result)


@tool
def save_brand_profile(
    name: str, role: str, industry: str, goals: str,
    preferred_tone: str, strategy_json: str,
) -> str:
    """Persist a brand profile to the database.

    strategy_json must be the JSON string returned by generate_brand_strategy.
    Returns JSON containing the new profile "id".
    """
    from database.db import SessionLocal

    try:
        strategy = json.loads(strategy_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "strategy_json was not valid JSON"})

    db = SessionLocal()
    try:
        profile = BrandProfile(
            name=name,
            role=role,
            industry=industry,
            goals=goals,
            preferred_tone=preferred_tone,
            tone=strategy.get("tone", ""),
            content_themes=json.dumps(strategy.get("content_themes", [])),
            positioning_summary=strategy.get("positioning_summary", ""),
            do_guidelines=json.dumps(strategy.get("do_guidelines", [])),
            dont_guidelines=json.dumps(strategy.get("dont_guidelines", [])),
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
        return json.dumps({"id": profile.id, "status": "saved"})
    finally:
        db.close()


BRAND_AGENT_TOOLS = [check_existing_profile, generate_brand_strategy, save_brand_profile]

BRAND_AGENT_PROMPT = """You are a personal-branding strategist agent.

Your job is to create a structured LinkedIn brand profile. Work through these
steps, calling one tool at a time:

1. Call check_existing_profile with the person's name.
2. Call generate_brand_strategy with all five details about the person.
3. Call save_brand_profile, passing the same five details plus the exact JSON
   string returned by generate_brand_strategy as strategy_json.
4. Reply with a one-sentence confirmation that includes the saved profile id.

Do not skip the save step. Do not invent a profile id."""


def get_brand_agent():
    """Build the compiled ReAct brand agent."""
    return create_react_agent(
        build_agent_llm(),
        BRAND_AGENT_TOOLS,
        prompt=BRAND_AGENT_PROMPT,
    )


# ── Deterministic fallback ──────────────────────────────────────

def _create_profile_directly(db: Session, request: BrandRequest) -> BrandProfile:
    """Single LLM call plus save. Used when the agent path does not persist."""
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
    return profile


def _to_response(profile: BrandProfile, trace: list, mode: str) -> BrandResponse:
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
        execution_mode=mode,
        agent_trace=trace,
    )


# ── Public entry point (used by the API) ────────────────────────

def create_brand_profile(db: Session, request: BrandRequest) -> BrandResponse:
    """Create a brand profile by running the ReAct agent."""
    if not AGENT_MODE:
        # AGENT_MODE=0: one direct LLM call instead of the agent loop.
        logger.info("AGENT_MODE=0 — direct path for %s", request.name)
        profile = _create_profile_directly(db, request)
        return _to_response(profile, [], "direct")

    logger.info("Running brand agent for: %s", request.name)

    instruction = (
        "Create a LinkedIn brand profile for this person.\n"
        f"name: {request.name}\n"
        f"role: {request.role}\n"
        f"industry: {request.industry}\n"
        f"goals: {request.goals}\n"
        f"preferred_tone: {request.preferred_tone}"
    )

    trace: list = []
    try:
        messages = run_agent(get_brand_agent(), instruction)
        trace = extract_trace(messages)
        profile_id = find_tool_result(messages, "save_brand_profile", "id")

        if profile_id:
            profile = db.query(BrandProfile).filter(BrandProfile.id == profile_id).first()
            if profile:
                logger.info("Brand agent saved profile id=%d", profile.id)
                return _to_response(profile, trace, "agent")

        logger.warning("Brand agent did not persist a profile; using direct path.")
    except Exception as exc:
        logger.warning("Brand agent failed (%s); using direct path.", exc)
        trace.append({"step": len(trace) + 1, "action": "agent_error", "error": str(exc)[:300]})

    profile = _create_profile_directly(db, request)
    return _to_response(profile, trace, "fallback_direct")
