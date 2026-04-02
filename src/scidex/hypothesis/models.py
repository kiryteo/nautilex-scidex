"""Pydantic models for hypothesis generation — hypotheses, reports, and gap analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Valid hypothesis types
HYPOTHESIS_TYPES = {
    "gap_filling",
    "analogy_transfer",
    "contradiction_resolution",
    "combination",
    "extension",
}


class Hypothesis(BaseModel):
    """A generated scientific hypothesis with scoring metadata."""

    id: str
    statement: str
    hypothesis_type: str
    supporting_evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    novelty_score: float = 0.0
    testability_score: float = 0.0
    source_papers: list[str] = Field(default_factory=list)
    evidence_summary: dict = Field(default_factory=dict)
    composite_score: float = 0.0
    generated_at: str = ""


class HypothesisReport(BaseModel):
    """Aggregated output from a hypothesis generation run."""

    topic: str
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    knowledge_gaps: list[str] = Field(default_factory=list)
    summary: str = ""


class GapAnalysis(BaseModel):
    """Result of knowledge-graph gap detection for a topic."""

    topic: str
    well_studied: list[str] = Field(default_factory=list)
    understudied: list[str] = Field(default_factory=list)
    connections_missing: list[tuple[str, str]] = Field(default_factory=list)
    suggested_directions: list[str] = Field(default_factory=list)
