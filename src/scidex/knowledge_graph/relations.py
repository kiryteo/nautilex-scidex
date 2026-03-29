"""Build relationships between entities extracted from papers.

Two strategies:
1. Structural: author→WROTE→paper, paper→CITES→paper, etc. from metadata.
2. Co-occurrence: entities mentioned in the same sentence get RELATED_TO edges.
"""

from __future__ import annotations

import re
import hashlib
from typing import Sequence

from scidex.knowledge_graph.models import Entity, Relationship
from scidex.knowledge_graph.extractor import _make_id


class RelationshipBuilder:
    """Build relationships from paper metadata and entity co-occurrence."""

    # ------------------------------------------------------------------
    # Structural relationships from metadata
    # ------------------------------------------------------------------

    def build_from_paper(self, paper_metadata: dict, entities: list[Entity]) -> list[Relationship]:
        """Build structural relationships from paper metadata.

        Creates:
        - author WROTE paper
        - paper CITES paper (from references)
        - paper PUBLISHED_IN venue
        - author AFFILIATED_WITH institution
        """
        relationships: list[Relationship] = []
        paper_id = paper_metadata.get("paperId", paper_metadata.get("paper_id", ""))
        if not paper_id:
            return relationships

        # Index entities by type for fast lookup
        entity_by_type: dict[str, list[Entity]] = {}
        for e in entities:
            entity_by_type.setdefault(e.entity_type, []).append(e)

        # author WROTE paper
        for author_entity in entity_by_type.get("author", []):
            relationships.append(
                Relationship(
                    source_id=author_entity.id,
                    target_id=paper_id,
                    rel_type="WROTE",
                    evidence=f"{author_entity.name} authored the paper",
                )
            )

        # paper PUBLISHED_IN venue
        for venue_entity in entity_by_type.get("venue", []):
            relationships.append(
                Relationship(
                    source_id=paper_id,
                    target_id=venue_entity.id,
                    rel_type="PUBLISHED_IN",
                    evidence=f"Published in {venue_entity.name}",
                )
            )

        # paper CITES paper (from references in metadata)
        references = paper_metadata.get("references", [])
        for ref in references:
            if isinstance(ref, dict):
                ref_id = ref.get("paperId", ref.get("paper_id", ""))
            elif isinstance(ref, str):
                ref_id = ref
            else:
                continue
            if ref_id:
                relationships.append(
                    Relationship(
                        source_id=paper_id,
                        target_id=ref_id,
                        rel_type="CITES",
                        evidence="Citation from references",
                    )
                )

        # author AFFILIATED_WITH institution
        for author in paper_metadata.get("authors", []):
            if isinstance(author, dict):
                affiliations = author.get("affiliations", [])
                author_name = author.get("name", "")
                author_id_val = author.get("authorId", "")
                if not author_id_val and author_name:
                    author_id_val = _make_id("author", author_name)
                for aff in affiliations:
                    if isinstance(aff, str) and aff:
                        inst_id = _make_id("institution", aff)
                        relationships.append(
                            Relationship(
                                source_id=author_id_val,
                                target_id=inst_id,
                                rel_type="AFFILIATED_WITH",
                                evidence=f"{author_name} affiliated with {aff}",
                            )
                        )

        return relationships

    # ------------------------------------------------------------------
    # Co-occurrence-based relationships
    # ------------------------------------------------------------------

    def build_from_cooccurrence(self, entities: list[Entity], text: str) -> list[Relationship]:
        """Build RELATED_TO edges for entities co-occurring in the same sentence.

        Splits text into sentences, then checks which entities are mentioned
        in each sentence. Pairs of entities in the same sentence get a
        RELATED_TO edge.
        """
        if not text or len(entities) < 2:
            return []

        # Split into sentences (simple heuristic)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        relationships: list[Relationship] = []
        seen_pairs: set[tuple[str, str]] = set()

        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Find which entities are mentioned in this sentence
            present: list[Entity] = []
            for entity in entities:
                if entity.name.lower() in sentence_lower:
                    present.append(entity)

            # Create pairwise RELATED_TO edges
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    e1, e2 = present[i], present[j]
                    if e1.id == e2.id:
                        continue
                    pair = tuple(sorted([e1.id, e2.id]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    relationships.append(
                        Relationship(
                            source_id=e1.id,
                            target_id=e2.id,
                            rel_type="RELATED_TO",
                            evidence=sentence.strip()[:200],
                            confidence=0.7,
                        )
                    )

        return relationships
