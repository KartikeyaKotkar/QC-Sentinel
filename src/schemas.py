from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BugReport(BaseModel):
    bug_id: str | None = Field(default=None, description="Unique bug identifier, e.g. GODOT-12345")
    title: str = Field(..., min_length=1, max_length=500)
    description: str | None = Field(default=None)
    subsystem: str | None = Field(default="Core")
    labels: list[str] = Field(default_factory=list)
    state: str | None = Field(default="closed")

    @field_validator("bug_id", mode="before")
    @classmethod
    def normalize_bug_id(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return f"TRIAGE-{uuid.uuid4().hex[:8].upper()}"
        return str(v).strip()

    @field_validator("description", mode="before")
    @classmethod
    def normalize_description(cls, v):
        if v is None:
            return ""
        return str(v)

    @field_validator("subsystem", mode="before")
    @classmethod
    def normalize_subsystem(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return "Core"
        return str(v).strip()

    @field_validator("labels", mode="before")
    @classmethod
    def normalize_labels(cls, v):
        if v is None:
            return []
        return v


class RetrievedBug(BaseModel):
    bug_id: str
    title: str
    description: str
    subsystem: str
    distance: float
    metadata: dict = Field(default_factory=dict)


class TriageResponse(BaseModel):
    is_duplicate: bool
    duplicate_of: str | None = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    root_cause_hypothesis: str
    suggested_workaround: str
    retrieved_context: list[RetrievedBug] = Field(default_factory=list)


class IngestPayload(BaseModel):
    bugs: list[BugReport]


class IngestResponse(BaseModel):
    ingested: int
    collection_count: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    chroma: Literal["connected", "disconnected"]
    model: str
    llm: str
    collection_count: int | None = None
