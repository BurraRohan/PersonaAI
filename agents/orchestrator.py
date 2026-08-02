"""
Orchestrator – LangGraph StateGraph chaining brand → content → feedback.

The graph uses conditional edges, so routing is a real decision made at runtime
rather than a fixed line of nodes:

    brand ──▶ (error?) ──▶ END
      │
      └──▶ content ──▶ (error?) ──▶ END
              │
              └──▶ check_history ──▶ (has engagement data?) ──▶ feedback ──▶ END
                                     └──▶ (no data) ────────────────────────▶ END

The check_history node exists because feedback on a brand new profile is
meaningless: there is nothing to analyse, and asking the model for a
data-driven summary of zero data invites it to invent numbers.
"""

import json
import logging
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, END

from database.db import SessionLocal
from database.models import BrandProfile, Post
from services.llm_service import (
    generate_brand_profile,
    generate_linkedin_post,
    generate_feedback,
)

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
    engagement_history: List[dict]

    # Output
    feedback_summary: Optional[str]
    feedback_available: bool
    workflow_steps: List[str]
    error: Optional[str]


def _step(state: WorkflowState, message: str) -> None:
    state["workflow_steps"] = state.get("workflow_steps", []) + [message]


# ── Nodes ───────────────────────────────────────────────────────

def brand_node(state: WorkflowState) -> WorkflowState:
    """Create the brand profile."""
    logger.info("[Orchestrator] brand_node for %s", state["name"])
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
        _step(state, f"Brand profile created (id={profile.id})")
        logger.info("[Orchestrator] brand profile id=%d", profile.id)

    except Exception as exc:
        state["error"] = f"Brand creation failed: {exc}"
        _step(state, f"Brand creation failed: {exc}")
    finally:
        db.close()

    return state


def content_node(state: WorkflowState) -> WorkflowState:
    """Generate and store the LinkedIn post."""
    logger.info("[Orchestrator] content_node topic=%s", state["topic"])
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
        _step(state, f"LinkedIn post generated (id={post.id})")
        logger.info("[Orchestrator] post id=%d", post.id)

    except Exception as exc:
        state["error"] = f"Content generation failed: {exc}"
        _step(state, f"Content generation failed: {exc}")
    finally:
        db.close()

    return state


def check_history_node(state: WorkflowState) -> WorkflowState:
    """Load any real engagement data for this profile.

    A brand new profile has none, which is what routes the graph past feedback.
    """
    logger.info("[Orchestrator] check_history_node")
    db = SessionLocal()
    try:
        posts = (
            db.query(Post)
            .filter(Post.user_id == state["brand_profile_id"])
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

        state["engagement_history"] = history
        _step(state, f"Engagement history checked ({len(history)} record(s) found)")

    except Exception as exc:
        state["engagement_history"] = []
        _step(state, f"Engagement history check failed: {exc}")
    finally:
        db.close()

    return state


def feedback_node(state: WorkflowState) -> WorkflowState:
    """Generate feedback. Only reached when real engagement data exists."""
    logger.info("[Orchestrator] feedback_node")
    db = SessionLocal()
    try:
        brand_summary = f"{state['name']} – {state['role']} in {state['industry']}"
        result = generate_feedback(state["engagement_history"], brand_summary, db=db)

        state["feedback_summary"] = " ".join(filter(None, [
            result.get("performance_summary", ""),
            result.get("improvement_recommendation", ""),
        ])).strip()
        state["feedback_available"] = True
        _step(state, "Feedback generated from engagement history")

    except Exception as exc:
        # Feedback is a bonus, never a blocker.
        state["feedback_summary"] = None
        state["feedback_available"] = False
        _step(state, f"Feedback skipped: {exc}")
    finally:
        db.close()

    return state


def skip_feedback_node(state: WorkflowState) -> WorkflowState:
    """Explicit cold-start branch: say so instead of inventing an analysis."""
    logger.info("[Orchestrator] skip_feedback_node (no engagement data)")
    state["feedback_summary"] = (
        "No engagement data has been logged yet, so there is nothing to analyse. "
        "Publish this post, record its likes, comments and shares via POST /engagement, "
        "then request feedback."
    )
    state["feedback_available"] = False
    _step(state, "Feedback skipped: no engagement data yet (cold start)")
    return state


# ── Routers (conditional edges) ─────────────────────────────────

def route_after_brand(state: WorkflowState) -> str:
    """Stop the graph if the brand step failed."""
    return "error" if state.get("error") else "continue"


def route_after_content(state: WorkflowState) -> str:
    """Stop the graph if the content step failed."""
    return "error" if state.get("error") else "continue"


def route_after_history(state: WorkflowState) -> str:
    """Only run feedback when there is real engagement data to reason about."""
    return "has_data" if state.get("engagement_history") else "no_data"


# ── Graph ───────────────────────────────────────────────────────

def build_orchestration_graph():
    """Construct and compile the PersonaAI workflow graph."""
    graph = StateGraph(WorkflowState)

    graph.add_node("brand", brand_node)
    graph.add_node("content", content_node)
    graph.add_node("check_history", check_history_node)
    graph.add_node("feedback", feedback_node)
    graph.add_node("skip_feedback", skip_feedback_node)

    graph.set_entry_point("brand")

    graph.add_conditional_edges(
        "brand",
        route_after_brand,
        {"continue": "content", "error": END},
    )
    graph.add_conditional_edges(
        "content",
        route_after_content,
        {"continue": "check_history", "error": END},
    )
    graph.add_conditional_edges(
        "check_history",
        route_after_history,
        {"has_data": "feedback", "no_data": "skip_feedback"},
    )

    graph.add_edge("feedback", END)
    graph.add_edge("skip_feedback", END)

    return graph.compile()


def render_graph_mermaid() -> str:
    """Return a Mermaid diagram of the compiled graph (handy for the report)."""
    return build_orchestration_graph().get_graph().draw_mermaid()


# ── Public Function ─────────────────────────────────────────────

def run_full_workflow(
    name: str, role: str, industry: str, goals: str,
    preferred_tone: str, topic: str,
) -> dict:
    """Execute the orchestrated workflow: brand → content → history → feedback."""
    logger.info("[Orchestrator] starting workflow for %s", name)

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
        "engagement_history": [],
        "feedback_summary": None,
        "feedback_available": False,
        "workflow_steps": [],
        "error": None,
    }

    final_state = workflow.invoke(initial_state)
    logger.info("[Orchestrator] complete. Steps: %s", final_state.get("workflow_steps"))
    return final_state
