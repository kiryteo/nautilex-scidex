"""Cross-domain analogy engine.

Finds methods or techniques applied in one domain that could transfer to
another.  Uses graph structure: methods connected to entities in domain A
but not in domain B, where the two domains share structural similarities.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from datetime import datetime, timezone

import networkx as nx

from scidex.knowledge_graph.graph import KnowledgeGraph
from scidex.hypothesis.models import Hypothesis

logger = logging.getLogger(__name__)

_DOMAIN_TYPES = {"disease", "gene", "protein", "compound"}


def _hypothesis_id(prefix: str, *parts: str) -> str:
    raw = ":".join([prefix, *parts])
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


class AnalogyEngine:
    """Discover cross-domain analogy hypotheses from graph structure."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_analogies(
        self,
        kg: KnowledgeGraph,
        source_method: str,
    ) -> list[Hypothesis]:
        """Find domains where *source_method* could be applied by analogy.

        Looks for domain entities connected to the method's current domains
        that are NOT yet connected to the method itself.

        Args:
            kg: Knowledge graph.
            source_method: Name of the method/technique to search from.

        Returns:
            List of ``Hypothesis`` objects (type ``analogy_transfer``).
        """
        G = kg._graph
        undirected = kg.undirected

        # Resolve method node
        method_id = kg._find_node_by_name(source_method)
        if method_id is None:
            logger.warning(f"Method '{source_method}' not found in graph.")
            return []

        method_name = G.nodes[method_id].get("name", method_id)

        # Domains currently connected to the method
        connected_domains = self._connected_domains(undirected, G, method_id)
        if not connected_domains:
            return []

        # Candidate domains: connected to at least one connected domain
        # but NOT directly connected to the method
        candidates = self._candidate_domains(undirected, G, method_id, connected_domains)

        hypotheses: list[Hypothesis] = []
        for candidate_id, shared_domains in candidates.items():
            cand_name = G.nodes[candidate_id].get("name", candidate_id)
            cand_type = G.nodes[candidate_id].get("entity_type", "unknown")

            shared_names = [G.nodes[d].get("name", d) for d in shared_domains]

            # Score: similarity = fraction of method's domains that are shared
            similarity = len(shared_domains) / max(len(connected_domains), 1)
            novelty = 1.0 - similarity  # less similar context → more novel

            evidence = [
                f"{method_name} is used for {G.nodes[d].get('name', d)} "
                f"which is connected to {cand_name}"
                for d in shared_domains
            ]

            h = Hypothesis(
                id=_hypothesis_id("analogy", method_name, cand_name),
                statement=(
                    f"{method_name} could be applied to {cand_name} — "
                    f"it is already used in related areas: {', '.join(shared_names)}"
                ),
                hypothesis_type="analogy_transfer",
                supporting_evidence=evidence,
                confidence=min(1.0, 0.3 + 0.2 * len(shared_domains)),
                novelty_score=novelty,
                testability_score=0.6,  # methods are generally testable
                source_papers=[],
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
            hypotheses.append(h)

        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _connected_domains(undirected: nx.Graph, G: nx.DiGraph, method_id: str) -> set[str]:
        """Return domain-type node IDs directly connected to *method_id*."""
        return {
            n
            for n in undirected.neighbors(method_id)
            if G.nodes[n].get("entity_type", "") in _DOMAIN_TYPES
        }

    @staticmethod
    def _candidate_domains(
        undirected: nx.Graph,
        G: nx.DiGraph,
        method_id: str,
        connected_domains: set[str],
    ) -> dict[str, list[str]]:
        """Find domain nodes that share neighbours with connected domains
        but are not directly connected to the method.

        Returns:
            Dict mapping candidate node ID → list of shared domain IDs.
        """
        method_neighbors = set(undirected.neighbors(method_id))
        candidates: dict[str, list[str]] = defaultdict(list)

        for domain_id in connected_domains:
            for neighbor in undirected.neighbors(domain_id):
                if neighbor == method_id:
                    continue
                if neighbor in method_neighbors:
                    continue
                if G.nodes[neighbor].get("entity_type", "") not in _DOMAIN_TYPES:
                    continue
                if neighbor in connected_domains:
                    continue
                candidates[neighbor].append(domain_id)

        return dict(candidates)
