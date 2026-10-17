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
from langchain.agents import create_agent

from database.models import BrandProfile
from schemas.schemas import BrandRequest, BrandResponse
from services.llm_service import generate_brand_profile
from utils.agent_runtime import build_agent_llm, run_agent, extract_trace, find_tool_result
from utils.budget import agents_allowed
from utils.config import AGENT_MODE

logger = logging.getLogger(__name__)


# ── Tools ───────────────────────────────────────────────────────

def build_brand_tools(request: BrandRequest) -> list:
    """Return the brand agent's tools, bound to one validated request.

    The person's details come from the request, never from the model. If the
    model retyped them into tool arguments, one slipped character in a name
    would go straight into the saved strategy. The generated strategy is also
    held here between tools instead of being copied through the model, which
    keeps it intact and spends fewer output tokens.
    """
    details = {
        "name": request.name,
        "role": request.role,
        "industry": request.industry,
        "goals": request.goals,
        "preferred_tone": request.preferred_tone,
    }
    state: dict = {}

    @tool
    def check_existing_profile() -> str:
        """Check whether a brand profile already exists for this person.

        Returns JSON with "exists": true/false, plus the profile id and role if found.
        """
        from database.db import SessionLocal
        db = SessionLocal()
        try:
            profile = (
                db.query(BrandProfile)
                .filter(BrandProfile.name == details["name"])
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
    def generate_brand_strategy() -> str:
        """Generate a LinkedIn brand strategy for this person.

        Returns JSON containing tone, content_themes, positioning_summary,
        do_guidelines and dont_guidelines. The strategy is kept for
        save_brand_profile, so there is nothing to pass on.
        """
        state["strategy"] = generate_brand_profile(**details)
        return json.dumps(state["strategy"])

    @tool
    def save_brand_profile() -> str:
        """Save this person's brand profile with the strategy just generated.

        Call generate_brand_strategy first. Returns JSON containing the new
        profile "id".
        """
        if "saved_id" in state:
            # A repeated save must not create a duplicate profile.
            return json.dumps({"id": state["saved_id"], "status": "saved"})
        strategy = state.get("strategy")
        if not strategy:
            return json.dumps({"error": "No strategy yet. Call generate_brand_strategy first."})

        from database.db import SessionLocal
        db = SessionLocal()
        try:
            profile = BrandProfile(
                **details,
                tone=strategy.get("tone", ""),
                content_themes=json.dumps(strategy.get("content_themes", [])),
                positioning_summary=strategy.get("positioning_summary", ""),
                do_guidelines=json.dumps(strategy.get("do_guidelines", [])),
                dont_guidelines=json.dumps(strategy.get("dont_guidelines", [])),
            )
            db.add(profile)
            db.commit()
            db.refresh(profile)
            state["saved_id"] = profile.id
            return json.dumps({"id": profile.id, "status": "saved"})
        finally:
            db.close()

    return [check_existing_profile, generate_brand_strategy, save_brand_profile]


BRAND_AGENT_PROMPT = """You are a personal-branding strategist agent.

Your job is to create a structured LinkedIn brand profile. The person's details
are already attached to your tools, so the tools take no arguments. Work
through these steps, calling one tool at a time:

1. Call check_existing_profile.
2. Call generate_brand_strategy.
3. Call save_brand_profile.
4. Reply with a one-sentence confirmation that includes the saved profile id.

Do not skip the save step. Do not invent a profile id."""


def get_brand_agent(request: BrandRequest):
    """Build the compiled ReAct brand agent for one request."""
    return create_agent(
        build_agent_llm(),
        build_brand_tools(request),
        system_prompt=BRAND_AGENT_PROMPT,
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
    if not AGENT_MODE or not agents_allowed():
        # One direct LLM call instead of the agent loop: either AGENT_MODE=0,
        # or today's spend has passed the budget's soft limit.
        mode = "direct" if not AGENT_MODE else "budget_direct"
        logger.info("%s path for %s", mode, request.name)
        profile = _create_profile_directly(db, request)
        return _to_response(profile, [], mode)

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
        messages = run_agent(get_brand_agent(request), instruction, "brand")
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
        trace.append({"step": len(trace) + 1, "action": "agent_error", "error": type(exc).__name__})

    profile = _create_profile_directly(db, request)
    return _to_response(profile, trace, "fallback_direct")
