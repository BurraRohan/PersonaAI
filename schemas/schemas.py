"""
Pydantic schemas for request validation and response serialization.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

# Agent execution metadata attached to responses so the tool-calling trace
# is visible to callers instead of only living in the server logs.
AgentTrace = List[Dict[str, Any]]


# ── Brand Endpoint ──────────────────────────────────────────────

class BrandRequest(BaseModel):
    name: str = Field(..., json_schema_extra={"example": "Alice Johnson"})
    role: str = Field(..., json_schema_extra={"example": "ML Engineer"})
    industry: str = Field(..., json_schema_extra={"example": "AI/ML"})
    goals: str = Field(..., json_schema_extra={"example": "Thought leadership and community building"})
    preferred_tone: str = Field(..., json_schema_extra={"example": "Professional yet approachable"})


class BrandResponse(BaseModel):
    id: int
    name: str
    role: str
    industry: str
    goals: str
    preferred_tone: str
    tone: Optional[str] = None
    content_themes: Optional[List[str]] = None
    positioning_summary: Optional[str] = None
    do_guidelines: Optional[List[str]] = None
    dont_guidelines: Optional[List[str]] = None
    created_at: datetime
    execution_mode: str = "agent"
    agent_trace: AgentTrace = []

    model_config = ConfigDict(from_attributes=True)


# ── Content Generation Endpoint ─────────────────────────────────

class GenerateRequest(BaseModel):
    user_id: int = Field(..., json_schema_extra={"example": 1})
    topic: str = Field(..., json_schema_extra={"example": "Why fine-tuning matters more than prompt engineering"})


class GenerateResponse(BaseModel):
    post_id: int
    post_content: str
    suggested_hashtags: Optional[List[str]] = None
    execution_mode: str = "agent"
    agent_trace: AgentTrace = []

    model_config = ConfigDict(from_attributes=True)


# ── Engagement Tracking Endpoint ────────────────────────────────

class EngagementRequest(BaseModel):
    post_id: int = Field(..., json_schema_extra={"example": 1})
    likes: int = Field(0, ge=0, json_schema_extra={"example": 42})
    comments: int = Field(0, ge=0, json_schema_extra={"example": 7})
    shares: int = Field(0, ge=0, json_schema_extra={"example": 3})


class EngagementResponse(BaseModel):
    id: int
    post_id: int
    likes: int
    comments: int
    shares: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ── Feedback Endpoint ───────────────────────────────────────────

class FeedbackRequest(BaseModel):
    user_id: int = Field(..., json_schema_extra={"example": 1})


class FeedbackResponse(BaseModel):
    user_id: int
    total_posts: int
    performance_summary: str
    improvement_recommendation: str
    execution_mode: str = "agent"
    agent_trace: AgentTrace = []


# ── Orchestration Endpoint (LangGraph) ──────────────────────────

class OrchestrateRequest(BaseModel):
    name: str = Field(..., json_schema_extra={"example": "Alice Johnson"})
    role: str = Field(..., json_schema_extra={"example": "ML Engineer"})
    industry: str = Field(..., json_schema_extra={"example": "AI/ML"})
    goals: str = Field(..., json_schema_extra={"example": "Thought leadership"})
    preferred_tone: str = Field(..., json_schema_extra={"example": "Professional yet approachable"})
    topic: str = Field(..., json_schema_extra={"example": "Why fine-tuning matters more than prompt engineering"})


class OrchestrateResponse(BaseModel):
    brand_profile_id: int
    brand_tone: Optional[str] = None
    post_id: int
    post_content: str
    suggested_hashtags: Optional[List[str]] = None
    feedback_summary: Optional[str] = None
    feedback_available: bool = False
    workflow_steps: List[str] = []


# ── Prompt Versioning ───────────────────────────────────────────

class PromptTemplateResponse(BaseModel):
    id: int
    agent_name: str
    version: int
    template: str
    is_active: bool
    description: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class PredictRequest(BaseModel):
    user_id: int = Field(..., json_schema_extra={"example": 1})
    draft_content: str = Field(..., json_schema_extra={"example": "Here's why every engineer should learn about LLMs..."})


class PredictResponse(BaseModel):
    overall_score: int
    predicted_likes: str
    predicted_comments: str
    predicted_shares: str
    brand_alignment: int
    hook_strength: int
    readability: int
    call_to_action: int
    improvement_tips: str

