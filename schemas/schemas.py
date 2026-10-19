"""
Pydantic schemas for request validation and response serialization.
"""

from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# Agent execution metadata attached to responses so the tool-calling trace
# is visible to callers instead of only living in the server logs.
AgentTrace = List[Dict[str, Any]]

# ── Input limits ────────────────────────────────────────────────
# Every request field is bounded. Unbounded text would let one request bloat
# the database or send a multi-megabyte prompt to a paid LLM API; unbounded
# integers overflow SQLite and crash the request. Short fields match their
# database column sizes.

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Role = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Short = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Medium = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
Long = Annotated[str, StringConstraints(max_length=2000)]
Topic = Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=300)]
PostText = Annotated[str, StringConstraints(max_length=10_000)]
ListItem = Annotated[str, StringConstraints(max_length=300)]
Hashtag = Annotated[str, StringConstraints(max_length=60)]

ID = Annotated[int, Field(ge=1, le=2_147_483_647)]
Count = Annotated[int, Field(ge=0, le=100_000_000)]


# ── Brand Endpoint ──────────────────────────────────────────────

class BrandRequest(BaseModel):
    name: Name = Field(..., json_schema_extra={"example": "Alice Johnson"})
    role: Role = Field(..., json_schema_extra={"example": "ML Engineer"})
    industry: Short = Field(..., json_schema_extra={"example": "AI/ML"})
    goals: Medium = Field(..., json_schema_extra={"example": "Thought leadership and community building"})
    preferred_tone: Short = Field(..., json_schema_extra={"example": "Professional yet approachable"})


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
    user_id: ID = Field(..., json_schema_extra={"example": 1})
    topic: Topic = Field(..., json_schema_extra={"example": "Why fine-tuning matters more than prompt engineering"})


class GenerateResponse(BaseModel):
    post_id: int
    post_content: str
    suggested_hashtags: Optional[List[str]] = None
    execution_mode: str = "agent"
    agent_trace: AgentTrace = []

    model_config = ConfigDict(from_attributes=True)


# ── Engagement Tracking Endpoint ────────────────────────────────

class EngagementRequest(BaseModel):
    post_id: ID = Field(..., json_schema_extra={"example": 1})
    likes: Count = Field(0, json_schema_extra={"example": 42})
    comments: Count = Field(0, json_schema_extra={"example": 7})
    shares: Count = Field(0, json_schema_extra={"example": 3})


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
    user_id: ID = Field(..., json_schema_extra={"example": 1})


class FeedbackResponse(BaseModel):
    user_id: int
    total_posts: int
    performance_summary: str
    improvement_recommendation: str
    execution_mode: str = "agent"
    agent_trace: AgentTrace = []


# ── Orchestration Endpoint (LangGraph) ──────────────────────────

class OrchestrateRequest(BaseModel):
    name: Name = Field(..., json_schema_extra={"example": "Alice Johnson"})
    role: Role = Field(..., json_schema_extra={"example": "ML Engineer"})
    industry: Short = Field(..., json_schema_extra={"example": "AI/ML"})
    goals: Medium = Field(..., json_schema_extra={"example": "Thought leadership"})
    preferred_tone: Short = Field(..., json_schema_extra={"example": "Professional yet approachable"})
    topic: Topic = Field(..., json_schema_extra={"example": "Why fine-tuning matters more than prompt engineering"})


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
    user_id: ID = Field(..., json_schema_extra={"example": 1})
    # Either score a saved post by id (which then becomes approvable), or paste
    # free text for a scratch draft that is not in the database.
    post_id: Optional[ID] = Field(None, json_schema_extra={"example": 1})
    draft_content: Optional[PostText] = Field(
        None,
        json_schema_extra={"example": "Here's why every engineer should learn about LLMs..."},
    )


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
    # Present only when a saved post was scored. The UI shows approve/reject
    # controls when post_id is not None.
    post_id: Optional[int] = None
    status: Optional[str] = None
    # True when an unchanged request was answered from the prediction cache.
    cached: bool = False


class ReviewRequest(BaseModel):
    """Approve or reject a generated post."""

    decision: Annotated[str, StringConstraints(max_length=20)] = Field(..., json_schema_extra={"example": "approve"})
    note: Optional[Long] = Field(
        None, json_schema_extra={"example": "Too generic, off-brand"}
    )


class ReviewResponse(BaseModel):
    post_id: int
    status: str
    reviewed_at: Optional[datetime] = None
    review_note: Optional[str] = None


class PendingPost(BaseModel):
    post_id: int
    topic: str
    content: str
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)



class PostEditRequest(BaseModel):
    """Reviewer edits to a pending post. Send either field, or both."""

    content: Optional[PostText] = Field(None, json_schema_extra={"example": "Edited post text."})
    hashtags: Optional[List[Hashtag]] = Field(None, max_length=15, json_schema_extra={"example": ["cloud", "finops"]})


class BrandEditRequest(BaseModel):
    """Partial update of a brand profile. Omitted fields are left unchanged."""

    name: Optional[Annotated[str, StringConstraints(max_length=100)]] = None
    role: Optional[Annotated[str, StringConstraints(max_length=200)]] = None
    industry: Optional[Annotated[str, StringConstraints(max_length=100)]] = None
    goals: Optional[Annotated[str, StringConstraints(max_length=1000)]] = None
    preferred_tone: Optional[Annotated[str, StringConstraints(max_length=100)]] = None
    tone: Optional[Long] = None
    content_themes: Optional[List[ListItem]] = Field(None, max_length=20)
    positioning_summary: Optional[Long] = None
    do_guidelines: Optional[List[ListItem]] = Field(None, max_length=20)
    dont_guidelines: Optional[List[ListItem]] = Field(None, max_length=20)
