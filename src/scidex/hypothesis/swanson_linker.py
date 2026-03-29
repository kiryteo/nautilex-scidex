"""Swanson ABC literature-based discovery.

Implements Don Swanson's model: given a starting entity A, find co-occurring
entities B, then for each B find co-occurring entities C that do NOT co-occur
with A.  The A→B→C chains suggest novel indirect relationships.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

import networkx as nx

from scidex.knowledge_graph.graph import KnowledgeGraph
from scidex.hypothesis.models import Hypothesis

logger = logging.getLogger(__name__)


def _hypothesis_id(prefix: str, *parts: str) -> str:
    """Deterministic short hash for a hypothesis."""
    raw = ":".join([prefix, *parts])
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


class SwansonLinker:
    """Discover novel A→B→C hypothesis chains via Swanson's ABC model."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_links(
        self,
        kg: KnowledgeGraph,
        source_entity: str,
        max_hops: int = 2,
    ) -> list[Hypothesis]:
        """Find novel indirect connections from *source_entity*.

        Args:
            kg: Knowledge graph to traverse.
            source_entity: Name of the starting entity (A).
            max_hops: Maximum chain length (default 2 → classic A-B-C).

        Returns:
            List of ``Hypothesis`` objects, sorted by confidence descending.
        """
        G = kg._graph
        undirected = kg.undirected

        # Resolve A
        a_id = kg._find_node_by_name(source_entity)
        if a_id is None:
            logger.warning(f"Source entity '{source_entity}' not found in graph.")
            return []

        a_name = G.nodes[a_id].get("name", a_id)

        # Direct neighbours of A (the B set)
        a_neighbors = set(undirected.neighbors(a_id))
        if not a_neighbors:
            return []

        # For each B, find its neighbours C that are NOT neighbours of A
        chains: list[dict] = []
        for b_id in a_neighbors:
            b_name = G.nodes[b_id].get("name", b_id)
            b_type = G.nodes[b_id].get("entity_type", "unknown")

            # Skip trivial intermediaries (same author, same venue)
            if b_type in {"author", "venue"}:
                continue

            b_neighbors = set(undirected.neighbors(b_id))
            c_candidates = b_neighbors - a_neighbors - {a_id}

            for c_id in c_candidates:
                c_type = G.nodes[c_id].get("entity_type", "unknown")
                if c_type in {"author", "venue"}:
                    continue

                c_name = G.nodes[c_id].get("name", c_id)

                # Score: count how many B intermediaries link A to this C
                chains.append(
                    {
                        "a_id": a_id,
                        "b_id": b_id,
                        "c_id": c_id,
                        "a_name": a_name,
                        "b_name": b_name,
                        "c_name": c_name,
                        "b_type": b_type,
                        "c_type": c_type,
                    }
                )

        # Aggregate: group by C, count unique B intermediaries
        c_groups: dict[str, list[dict]] = {}
        for chain in chains:
            c_groups.setdefault(chain["c_id"], []).append(chain)

        hypotheses: list[Hypothesis] = []
        for c_id, group in c_groups.items():
            c_name = group[0]["c_name"]
            unique_bs = {ch["b_id"] for ch in group}
            b_names = [G.nodes[b].get("name", b) for b in unique_bs]

            # Confidence scales with number of intermediaries
            n_intermediaries = len(unique_bs)
            confidence = min(1.0, 0.3 + 0.15 * n_intermediaries)

            # Novelty: based on graph distance (higher distance → more novel)
            try:
                dist = nx.shortest_path_length(undirected, a_id, c_id)
            except nx.NetworkXNoPath:
                dist = max_hops + 1
            novelty = min(1.0, dist / (max_hops + 1))

            # Collect evidence strings
            evidence_parts = [
                f"{a_name} co-occurs with {b} which co-occurs with {c_name}" for b in b_names
            ]

            hypothesis = Hypothesis(
                id=_hypothesis_id("swanson", a_name, c_name),
                statement=(f"{a_name} may be related to {c_name} through {', '.join(b_names)}"),
                hypothesis_type="combination",
                supporting_evidence=evidence_parts,
                confidence=confidence,
                novelty_score=novelty,
                testability_score=0.5,  # default; refined by LLM later
                source_papers=[],
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
            hypotheses.append(hypothesis)

        # Sort by confidence descending
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses
