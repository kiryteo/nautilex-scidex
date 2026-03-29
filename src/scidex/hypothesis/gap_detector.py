"""Detect knowledge gaps by analyzing graph structure and connectivity.

Uses node degree, betweenness centrality, connected-component analysis, and
shared-neighbor heuristics to surface under-explored areas and missing links.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx

from scidex.knowledge_graph.graph import KnowledgeGraph
from scidex.hypothesis.models import GapAnalysis

logger = logging.getLogger(__name__)

# Minimum degree to consider an entity "well studied"
_WELL_STUDIED_DEGREE = 3

# Maximum degree to flag as "understudied" (inclusive)
_UNDERSTUDIED_MAX_DEGREE = 1


class GapDetector:
    """Analyse a KnowledgeGraph to find research gaps and bridge opportunities."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_gaps(self, kg: KnowledgeGraph, topic: str) -> GapAnalysis:
        """Run gap analysis on the knowledge graph for a given topic.

        Args:
            kg: The knowledge graph to analyse.
            topic: A descriptive topic string (used mainly for labelling).

        Returns:
            A ``GapAnalysis`` summarising well-studied entities,
            understudied entities, missing connections, and suggested
            research directions.
        """
        G = kg._graph
        undirected = kg.undirected

        well_studied = self._find_well_studied(G)
        understudied = self._find_understudied(G)
        missing = self._find_missing_connections(undirected)
        cross_method = self._find_cross_method_gaps(G, undirected)
        bridges = self.find_bridge_opportunities(kg)

        # Build suggested research directions from the detected gaps
        directions: list[str] = []
        for e1, e2, reason in bridges[:5]:
            directions.append(f"Investigate link between {e1} and {e2}: {reason}")
        for e1, e2 in missing[:5]:
            n1 = G.nodes[e1].get("name", e1)
            n2 = G.nodes[e2].get("name", e2)
            directions.append(f"Explore potential relationship between {n1} and {n2}")
        for method, domain in cross_method[:5]:
            directions.append(f"Apply {method} to {domain}")

        return GapAnalysis(
            topic=topic,
            well_studied=[G.nodes[n].get("name", n) for n in well_studied],
            understudied=[G.nodes[n].get("name", n) for n in understudied],
            connections_missing=[
                (G.nodes[a].get("name", a), G.nodes[b].get("name", b)) for a, b in missing
            ],
            suggested_directions=directions,
        )

    def find_bridge_opportunities(self, kg: KnowledgeGraph) -> list[tuple[str, str, str]]:
        """Find entity pairs that could bridge disconnected clusters.

        Returns:
            List of (entity1_name, entity2_name, reasoning) tuples.
        """
        G = kg._graph
        undirected = kg.undirected

        # Get connected components
        components = list(nx.connected_components(undirected))
        if len(components) < 2:
            # Graph is fully connected; fall back to community-based bridges
            return self._community_bridges(undirected, G)

        # For each pair of components, pick high-centrality nodes as bridge candidates
        bridges: list[tuple[str, str, str]] = []
        centrality = nx.degree_centrality(undirected)

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                c1 = components[i]
                c2 = components[j]

                # Pick node with highest centrality from each component
                best1 = max(c1, key=lambda n: centrality.get(n, 0))
                best2 = max(c2, key=lambda n: centrality.get(n, 0))

                name1 = G.nodes[best1].get("name", best1)
                name2 = G.nodes[best2].get("name", best2)
                type1 = G.nodes[best1].get("entity_type", "unknown")
                type2 = G.nodes[best2].get("entity_type", "unknown")

                reason = (
                    f"{name1} ({type1}) and {name2} ({type2}) are in disconnected "
                    f"clusters of size {len(c1)} and {len(c2)}"
                )
                bridges.append((name1, name2, reason))

        return bridges

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _find_well_studied(G: nx.DiGraph) -> list[str]:
        """Return node IDs with degree >= threshold."""
        return [n for n, d in G.degree() if d >= _WELL_STUDIED_DEGREE]

    @staticmethod
    def _find_understudied(G: nx.DiGraph) -> list[str]:
        """Return node IDs with very low degree."""
        return [
            n
            for n, d in G.degree()
            if d <= _UNDERSTUDIED_MAX_DEGREE
            and G.nodes[n].get("entity_type", "") not in {"venue", "paper"}
        ]

    @staticmethod
    def _find_missing_connections(undirected: nx.Graph) -> list[tuple[str, str]]:
        """Find entity pairs that share neighbours but have no direct edge.

        This is a lightweight "link prediction" heuristic based on common
        neighbours: if two nodes share at least 2 neighbours but are not
        directly connected, they are candidates for a missing link.
        """
        missing: list[tuple[str, str]] = []
        nodes = list(undirected.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                if undirected.has_edge(n1, n2):
                    continue
                common = len(set(undirected.neighbors(n1)) & set(undirected.neighbors(n2)))
                if common >= 2:
                    missing.append((n1, n2))
        return missing

    @staticmethod
    def _find_cross_method_gaps(
        G: nx.DiGraph,
        undirected: nx.Graph,
    ) -> list[tuple[str, str]]:
        """Find methods used in one domain but not another.

        Looks for method nodes connected to one disease/gene but not others
        that share related entities.

        Returns:
            List of (method_name, domain_name) pairs.
        """
        method_nodes = [n for n, d in G.nodes(data=True) if d.get("entity_type") == "method"]
        domain_nodes = [
            n
            for n, d in G.nodes(data=True)
            if d.get("entity_type") in {"disease", "gene", "protein"}
        ]

        if not method_nodes or not domain_nodes:
            return []

        # Map each method → set of domains it is connected to
        method_domains: dict[str, set[str]] = defaultdict(set)
        for m in method_nodes:
            for neighbor in undirected.neighbors(m):
                if neighbor in set(domain_nodes):
                    method_domains[m].add(neighbor)

        # For each method, find domains NOT yet connected
        gaps: list[tuple[str, str]] = []
        domain_set = set(domain_nodes)
        for m, connected_domains in method_domains.items():
            missing_domains = domain_set - connected_domains
            for d in missing_domains:
                # Only suggest if the domain shares a neighbour with a connected domain
                connected_neighbors = set()
                for cd in connected_domains:
                    connected_neighbors |= set(undirected.neighbors(cd))
                if d in connected_neighbors:
                    m_name = G.nodes[m].get("name", m)
                    d_name = G.nodes[d].get("name", d)
                    gaps.append((m_name, d_name))
        return gaps

    @staticmethod
    def _community_bridges(undirected: nx.Graph, G: nx.DiGraph) -> list[tuple[str, str, str]]:
        """When the graph is connected, use greedy modularity communities
        to identify potential inter-community bridges."""
        try:
            communities = list(nx.community.greedy_modularity_communities(undirected))
        except Exception:
            return []

        if len(communities) < 2:
            return []

        bridges: list[tuple[str, str, str]] = []
        centrality = nx.degree_centrality(undirected)

        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):
                c1 = communities[i]
                c2 = communities[j]
                best1 = max(c1, key=lambda n: centrality.get(n, 0))
                best2 = max(c2, key=lambda n: centrality.get(n, 0))

                name1 = G.nodes[best1].get("name", best1)
                name2 = G.nodes[best2].get("name", best2)

                reason = (
                    f"{name1} and {name2} are central nodes in separate "
                    f"communities of size {len(c1)} and {len(c2)}"
                )
                bridges.append((name1, name2, reason))
        return bridges
