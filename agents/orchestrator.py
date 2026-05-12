"""
Orchestrator – LangGraph StateGraph that chains brand → content → feedback agents.
Implements the planner-executor pattern from Unit 2.
"""

import json
import logging
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, END

from database.db import SessionLocal
from database.models import BrandProfile, Post, Engagement
from services.llm_service import generate_brand_profile, generate_linkedin_post, generate_feedback

logger = logging.getLogger(__name__)


# ── Shared State ────────────────────────────────────────────────

class WorkflowState(TypedDict):
    # Input
    name: str
    role: str
    industry: str
    goals: str
    preferred_tone: str
    topic: str

    # Intermediate state
    brand_profile_id: Optional[int]
    brand_context: Optional[dict]
    post_id: Optional[int]
    post_content: Optional[str]
    suggested_hashtags: Optional[List[str]]

    # Output
    feedback_summary: Optional[str]
    workflow_steps: List[str]
    error: Optional[str]


# ── Node Functions ──────────────────────────────────────────────

def brand_node(state: WorkflowState) -> WorkflowState:
    """Node 1: Create brand profile using the brand agent."""
    logger.info("[Orchestrator] Running brand_node for %s", state["name"])
    db = SessionLocal()
    try:
        result = generate_brand_profile(
            name=state["name"],
            role=state["role"],
            industry=state["industry"],
            goals=state["goals"],
            preferred_tone=state["preferred_tone"],
            db=db,
        )

        profile = BrandProfile(
            name=state["name"],
            role=state["role"],
            industry=state["industry"],
            goals=state["goals"],
            preferred_tone=state["preferred_tone"],
            tone=result.get("tone", ""),
            content_themes=json.dumps(result.get("content_themes", [])),
            positioning_summary=result.get("positioning_summary", ""),
            do_guidelines=json.dumps(result.get("do_guidelines", [])),
            dont_guidelines=json.dumps(result.get("dont_guidelines", [])),
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)

        state["brand_profile_id"] = profile.id
        state["brand_context"] = result
        state["workflow_steps"] = state.get("workflow_steps", []) + [
            f"Brand profile created (id={profile.id})"
        ]
        logger.info("[Orchestrator] Brand profile created: id=%d", profile.id)

    except Exception as e:
        state["error"] = f"Brand creation failed: {str(e)}"
        state["workflow_steps"] = state.get("workflow_steps", []) + [
            f"Brand creation failed: {str(e)}"
        ]
    finally:
        db.close()

    return state


def content_node(state: WorkflowState) -> WorkflowState:
    """Node 2: Generate LinkedIn post using the content agent."""
    if state.get("error"):
        return state

    logger.info("[Orchestrator] Running content_node for topic: %s", state["topic"])
    db = SessionLocal()
    try:
        result = generate_linkedin_post(
            brand_profile=state["brand_context"],
            topic=state["topic"],
            db=db,
        )

        post = Post(
            user_id=state["brand_profile_id"],
            topic=state["topic"],
            content=result.get("post_content", ""),
            hashtags=json.dumps(result.get("suggested_hashtags", [])),
        )
        db.add(post)
        db.commit()
        db.refresh(post)

        state["post_id"] = post.id
        state["post_content"] = post.content
        state["suggested_hashtags"] = result.get("suggested_hashtags", [])
        state["workflow_steps"] = state.get("workflow_steps", []) + [
            f"LinkedIn post generated (id={post.id})"
        ]
        logger.info("[Orchestrator] Post created: id=%d", post.id)

    except Exception as e:
        state["error"] = f"Content generation failed: {str(e)}"
        state["workflow_steps"] = state.get("workflow_steps", []) + [
            f"Content generation failed: {str(e)}"
        ]
    finally:
        db.close()

    return state


def feedback_node(state: WorkflowState) -> WorkflowState:
    """Node 3: Generate feedback based on the brand and generated content."""
    if state.get("error"):
        return state

    logger.info("[Orchestrator] Running feedback_node")
    db = SessionLocal()
    try:
        # Build engagement-like data from the just-generated post
        history = [{
            "post_id": state["post_id"],
            "topic": state["topic"],
            "likes": 0,
            "comments": 0,
            "shares": 0,
            "created_at": None,
        }]

        # Also fetch any prior engagement data for this user
        posts = (
            db.query(Post)
            .filter(Post.user_id == state["brand_profile_id"])
            .all()
        )
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

        brand_summary = f"{state['name']} – {state['role']} in {state['industry']}"
        result = generate_feedback(history, brand_summary, db=db)

        state["feedback_summary"] = result.get("performance_summary", "") + " " + result.get("improvement_recommendation", "")
        state["workflow_steps"] = state.get("workflow_steps", []) + [
            "Feedback generated based on engagement history"
        ]

    except Exception as e:
        # Feedback is optional — don't block the workflow
        state["feedback_summary"] = f"Feedback generation skipped: {str(e)}"
        state["workflow_steps"] = state.get("workflow_steps", []) + [
            f"Feedback skipped: {str(e)}"
        ]
    finally:
        db.close()

    return state


def should_continue(state: WorkflowState) -> str:
    """Router: decide whether to continue or stop on error."""
    if state.get("error"):
        return END
    return "next"


# ── Build the Graph ─────────────────────────────────────────────

def build_orchestration_graph() -> StateGraph:
    """Construct the LangGraph StateGraph for the full PersonaAI workflow."""

    graph = StateGraph(WorkflowState)

    # Add nodes
    graph.add_node("brand", brand_node)
    graph.add_node("content", content_node)
    graph.add_node("feedback", feedback_node)

    # Set entry point
    graph.set_entry_point("brand")

    # Add edges: brand → content → feedback → END
    graph.add_edge("brand", "content")
    graph.add_edge("content", "feedback")
    graph.add_edge("feedback", END)

    return graph.compile()


# ── Public Function ─────────────────────────────────────────────

def run_full_workflow(
    name: str, role: str, industry: str, goals: str,
    preferred_tone: str, topic: str,
) -> dict:
    """Execute the full orchestrated workflow: brand → content → feedback."""

    logger.info("[Orchestrator] Starting full workflow for %s", name)

    workflow = build_orchestration_graph()

    initial_state: WorkflowState = {
        "name": name,
        "role": role,
        "industry": industry,
        "goals": goals,
        "preferred_tone": preferred_tone,
        "topic": topic,
        "brand_profile_id": None,
        "brand_context": None,
        "post_id": None,
        "post_content": None,
        "suggested_hashtags": None,
        "feedback_summary": None,
        "workflow_steps": [],
        "error": None,
    }

    final_state = workflow.invoke(initial_state)

    logger.info("[Orchestrator] Workflow complete. Steps: %s", final_state["workflow_steps"])

    return final_state
