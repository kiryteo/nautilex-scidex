"""Pydantic models for the knowledge accumulation system."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeLayer(str, Enum):
    RAW = "raw"  # Papers, abstracts, metadata
    DOMAIN = "domain"  # Extracted entities, relationships
    OUTCOMES = "outcomes"  # Hypotheses, experiment results
    META = "meta"  # Patterns across research sessions


class KnowledgeEntry(BaseModel):
    """A single piece of accumulated knowledge."""

    id: str
    layer: KnowledgeLayer
    content: str
    source: str  # paper_id, hypothesis_id, experiment_id, etc.
    tags: list[str] = Field(default_factory=list)
    created_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeSnapshot(BaseModel):
    """State of accumulated knowledge."""

    entries: list[KnowledgeEntry] = Field(default_factory=list)
    total_papers: int = 0
    total_entities: int = 0
    total_hypotheses: int = 0
    total_experiments: int = 0
    sessions: list[dict] = Field(default_factory=list)
