"""Pydantic models for the knowledge graph — entities, relationships, and stats."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Valid entity types
ENTITY_TYPES = {
    "author",
    "paper",
    "institution",
    "method",
    "dataset",
    "gene",
    "protein",
    "disease",
    "compound",
    "venue",
}

# Valid relationship types
RELATIONSHIP_TYPES = {
    "WROTE",
    "CITES",
    "USES_METHOD",
    "USES_DATASET",
    "AFFILIATED_WITH",
    "PUBLISHED_IN",
    "RELATED_TO",
}


class Entity(BaseModel):
    """A node in the knowledge graph."""

    id: str
    name: str
    entity_type: str
    properties: dict = Field(default_factory=dict)
    mentions: list[str] = Field(default_factory=list)


class Relationship(BaseModel):
    """A directed edge in the knowledge graph."""

    source_id: str
    target_id: str
    rel_type: str
    properties: dict = Field(default_factory=dict)
    evidence: str = ""
    confidence: float = 1.0


class KnowledgeGraphStats(BaseModel):
    """Summary statistics for a knowledge graph."""

    num_nodes: int
    num_edges: int
    node_types: dict[str, int]
    edge_types: dict[str, int]
