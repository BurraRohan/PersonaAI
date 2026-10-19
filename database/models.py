"""
SQLAlchemy ORM models for PersonaAI.
Tables: brand_profiles, posts, engagements, prompt_templates, audit_logs,
prediction_cache.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Float
from sqlalchemy.orm import relationship
from database.db import Base


class BrandProfile(Base):
    """Stores a user's personal brand strategy profile."""

    __tablename__ = "brand_profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    role = Column(String(200), nullable=False)
    industry = Column(String(100), nullable=False)
    goals = Column(Text, nullable=False)
    preferred_tone = Column(String(100), nullable=False)

    # LLM-generated brand strategy fields
    tone = Column(Text, nullable=True)
    content_themes = Column(Text, nullable=True)
    positioning_summary = Column(Text, nullable=True)
    do_guidelines = Column(Text, nullable=True)
    dont_guidelines = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    posts = relationship("Post", back_populates="brand_profile")


class Post(Base):
    """Stores generated LinkedIn posts."""

    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("brand_profiles.id"), nullable=False)
    topic = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    hashtags = Column(Text, nullable=True)

    # Human-in-the-loop review state: "pending", "approved" or "rejected".
    # Generated posts start pending; nothing enters the engagement pipeline
    # until a person has looked at it and decided.
    status = Column(String(20), nullable=False, default="pending", index=True)
    reviewed_at = Column(DateTime, nullable=True)
    review_note = Column(Text, nullable=True)

    # Set the first time a reviewer edits the post. `content` then holds the
    # human-edited text and this keeps what the model originally wrote, so
    # there is always a record of how much the reviewer changed.
    original_content = Column(Text, nullable=True)
    edited_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    brand_profile = relationship("BrandProfile", back_populates="posts")
    engagements = relationship("Engagement", back_populates="post")


class Engagement(Base):
    """Stores engagement metrics for a post."""

    __tablename__ = "engagements"

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), nullable=False)
    likes = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    shares = Column(Integer, default=0)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    post = relationship("Post", back_populates="engagements")


# ── Fix 5: Prompt Versioning ───────────────────────────────────

class PromptTemplate(Base):
    """Stores versioned prompt templates for each agent."""

    __tablename__ = "prompt_templates"

    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String(50), nullable=False, index=True)  # brand, content, feedback
    version = Column(Integer, nullable=False, default=1)
    template = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    description = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AuditLog(Base):
    """Logs every LLM call for traceability and debugging."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    trace_id = Column(String(36), nullable=False, index=True)
    agent_name = Column(String(50), nullable=False)
    prompt_version = Column(Integer, nullable=True)
    input_summary = Column(Text, nullable=True)
    output_summary = Column(Text, nullable=True)
    model = Column(String(100), nullable=True)
    latency_ms = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="success")  # success, error, retry
    error_message = Column(Text, nullable=True)

    # Token usage and cost, captured from the provider's response. Cost is
    # stored at the time of the call so a later price change does not rewrite
    # history. All of this is operator data: admin routes and Streamlit only.
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    cached_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)

    # Groups every LLM call made while serving one HTTP request, so the cost of
    # a whole action (an agent run plus the calls its tools make) can be summed.
    request_id = Column(String(36), nullable=True, index=True)
    endpoint = Column(String(100), nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PredictionCache(Base):
    """A stored engagement prediction, reused while its inputs are unchanged.

    The key is a hash of everything that shapes a prediction: the draft text,
    the brand profile fields the predictor reads, the engagement history, the
    model and the predictor prompt version. Change any of them and the key
    changes, so a stale score is never served.
    """

    __tablename__ = "prediction_cache"

    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String(64), nullable=False, unique=True, index=True)
    result_json = Column(Text, nullable=False)
    model = Column(String(100), nullable=True)
    prompt_version = Column(Integer, nullable=True)
    # What the original call cost, so hits can be counted as money saved.
    cost_usd = Column(Float, nullable=True)
    hit_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    last_hit_at = Column(DateTime, nullable=True)
