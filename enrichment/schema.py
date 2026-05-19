"""Pydantic models for structured lead enrichment output."""
from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class Lead(BaseModel):
    """Input lead from CSV."""
    name: str
    website: Optional[str] = None
    designation: Optional[str] = None
    company: Optional[str] = None  # optional, derived from website if absent
    extra: dict = Field(default_factory=dict)  # any additional CSV columns

    @field_validator("name")
    @classmethod
    def name_must_be_present(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Lead name is required")
        return v.strip()


class ScoreBreakdown(BaseModel):
    """Sub-scores rolling up to the final 0-100 lead score."""
    relevance: int = Field(ge=0, le=100, description="ICP fit on industry/role/size")
    urgency: int = Field(ge=0, le=100, description="Buying signals, recent triggers")
    time_to_close: int = Field(
        ge=0, le=100,
        description="Likelihood of fast close (higher = faster)"
    )


class EnrichmentResult(BaseModel):
    """Structured output Claude returns per lead."""
    # Identity (echo back so we can join in the UI)
    name: str
    company: Optional[str] = None
    website: Optional[str] = None
    designation: Optional[str] = None

    # Account intelligence
    company_summary: str = Field(description="2-4 sentences: what the company does")
    person_summary: str = Field(description="2-3 sentences: likely priorities/pain")
    signals: List[str] = Field(
        default_factory=list,
        description="Recent newsworthy signals (funding, launches, hires, etc.)"
    )

    # Scoring
    score: int = Field(ge=0, le=100, description="Overall lead score")
    score_breakdown: ScoreBreakdown
    justification: str = Field(description="2-3 sentence rationale for the score")

    # Provenance
    sources: List[str] = Field(default_factory=list, description="URLs cited")
    confidence: str = Field(
        default="medium",
        description="low | medium | high — how grounded the result is"
    )
    error: Optional[str] = None  # populated only if enrichment failed
