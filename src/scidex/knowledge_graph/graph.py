"""NetworkX-backed knowledge graph with query, path-finding, and serialization.

Stores entities as nodes and relationships as directed edges in a DiGraph.
Provides convenience methods for ingesting papers end-to-end.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx

from scidex.knowledge_graph.models import Entity, Relationship, KnowledgeGraphStats
from scidex.knowledge_graph.extractor import EntityExtractor
from scidex.knowledge_graph.relations import RelationshipBuilder

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """In-memory knowledge graph backed by networkx.DiGraph."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._undirected: nx.Graph | None = None
        self._extractor = EntityExtractor()
        self._builder = RelationshipBuilder()

    # ------------------------------------------------------------------
    # Cached undirected view
    # ------------------------------------------------------------------

    @property
    def undirected(self) -> nx.Graph:
        """Return a cached undirected copy of the graph.

        Invalidated automatically when nodes or edges are added.
        """
        if self._undirected is None:
            self._undirected = self._graph.to_undirected()
        return self._undirected

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> None:
        """Add an entity as a graph node."""
        self._graph.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity.entity_type,
            properties=entity.properties,
            mentions=entity.mentions,
        )
        self._undirected = None

    def add_relationship(self, rel: Relationship) -> None:
        """Add a relationship as a directed edge.

        Creates placeholder nodes if source/target are not yet in the graph.
        """
        # Ensure both endpoints exist
        if rel.source_id not in self._graph:
            self._graph.add_node(
                rel.source_id, name=rel.source_id, entity_type="unknown", properties={}, mentions=[]
            )
        if rel.target_id not in self._graph:
            self._graph.add_node(
                rel.target_id, name=rel.target_id, entity_type="unknown", properties={}, mentions=[]
            )

        self._graph.add_edge(
            rel.source_id,
            rel.target_id,
            rel_type=rel.rel_type,
            properties=rel.properties,
            evidence=rel.evidence,
            confidence=rel.confidence,
        )
        self._undirected = None

    def add_paper(self, paper_metadata: dict) -> None:
        """Convenience: extract entities, build relations, add everything.

        Args:
            paper_metadata: Dict with keys like paperId, title, abstract,
                authors, venue, references, etc.
        """
        # Extract entities from metadata
        entities = self._extractor.extract_from_paper(paper_metadata)

        # Extract entities from abstract
        abstract = paper_metadata.get("abstract", "")
        if abstract:
            abstract_entities = self._extractor.extract_from_abstract(abstract)
            entities.extend(abstract_entities)

        # Add all entities
        for entity in entities:
            self.add_entity(entity)

        # Build and add structural relationships
        relationships = self._builder.build_from_paper(paper_metadata, entities)
        for rel in relationships:
            self.add_relationship(rel)

        # Build and add co-occurrence relationships from abstract
        if abstract:
            cooccurrence_rels = self._builder.build_from_cooccurrence(entities, abstract)
            for rel in cooccurrence_rels:
                self.add_relationship(rel)

        logger.info(
            f"Added paper '{paper_metadata.get('title', '?')[:50]}' — "
            f"{len(entities)} entities, {len(relationships)} relationships"
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_entity(self, name: str) -> dict:
        """Find an entity by name and return it with all connections.

        Searches by name (case-insensitive). Returns the first match with
        its incoming and outgoing edges.

        Returns:
            Dict with keys: entity, incoming, outgoing. Empty dict if not found.
        """
        # Find node by name
        target_id = None
        for node_id, data in self._graph.nodes(data=True):
            if data.get("name", "").lower() == name.lower():
                target_id = node_id
                break

        if target_id is None:
            return {}

        node_data = self._graph.nodes[target_id]
        incoming = []
        for src, _, edge_data in self._graph.in_edges(target_id, data=True):
            incoming.append(
                {
                    "source_id": src,
                    "source_name": self._graph.nodes[src].get("name", src),
                    **edge_data,
                }
            )

        outgoing = []
        for _, tgt, edge_data in self._graph.out_edges(target_id, data=True):
            outgoing.append(
                {
                    "target_id": tgt,
                    "target_name": self._graph.nodes[tgt].get("name", tgt),
                    **edge_data,
                }
            )

        return {
            "entity": {
                "id": target_id,
                "name": node_data.get("name", ""),
                "entity_type": node_data.get("entity_type", ""),
                "properties": node_data.get("properties", {}),
            },
            "incoming": incoming,
            "outgoing": outgoing,
        }

    def find_path(self, entity1: str, entity2: str) -> list[str]:
        """Find shortest path between two entities (by name).

        Treats graph as undirected for path finding. Returns list of entity
        names along the path.

        Returns:
            List of entity names from entity1 to entity2, or empty list if
            no path exists.
        """
        id1 = self._find_node_by_name(entity1)
        id2 = self._find_node_by_name(entity2)
        if id1 is None or id2 is None:
            return []

        try:
            path_ids = nx.shortest_path(self.undirected, id1, id2)
        except nx.NetworkXNoPath:
            return []

        return [self._graph.nodes[nid].get("name", nid) for nid in path_ids]

    def get_subgraph(self, entity_id: str, depth: int = 2) -> dict:
        """Get local neighborhood of an entity as JSON-serializable dict.

        Does BFS from entity_id up to given depth.

        Returns:
            Dict with keys: nodes (list), edges (list).
        """
        if entity_id not in self._graph:
            return {"nodes": [], "edges": []}

        # BFS to collect nearby nodes
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(entity_id, 0)])
        visited.add(entity_id)

        while queue:
            current, d = queue.popleft()
            if d >= depth:
                continue
            # Get all neighbors (both directions)
            neighbors = set(self._graph.successors(current)) | set(
                self._graph.predecessors(current)
            )
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))

        # Build subgraph data
        nodes = []
        for nid in visited:
            data = self._graph.nodes[nid]
            nodes.append(
                {
                    "id": nid,
                    "name": data.get("name", ""),
                    "entity_type": data.get("entity_type", ""),
                    "properties": data.get("properties", {}),
                }
            )

        edges = []
        for src, tgt, data in self._graph.edges(data=True):
            if src in visited and tgt in visited:
                edges.append(
                    {
                        "source_id": src,
                        "target_id": tgt,
                        "rel_type": data.get("rel_type", ""),
                        "evidence": data.get("evidence", ""),
                        "confidence": data.get("confidence", 1.0),
                    }
                )

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> KnowledgeGraphStats:
        """Get summary statistics for the graph."""
        node_types: dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            t = data.get("entity_type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1

        edge_types: dict[str, int] = {}
        for _, _, data in self._graph.edges(data=True):
            t = data.get("rel_type", "unknown")
            edge_types[t] = edge_types.get(t, 0) + 1

        return KnowledgeGraphStats(
            num_nodes=self._graph.number_of_nodes(),
            num_edges=self._graph.number_of_edges(),
            node_types=node_types,
            edge_types=edge_types,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        """Serialize the entire graph to a JSON-serializable dict."""
        nodes = []
        for nid, data in self._graph.nodes(data=True):
            nodes.append(
                {
                    "id": nid,
                    "name": data.get("name", ""),
                    "entity_type": data.get("entity_type", ""),
                    "properties": data.get("properties", {}),
                    "mentions": data.get("mentions", []),
                }
            )

        edges = []
        for src, tgt, data in self._graph.edges(data=True):
            edges.append(
                {
                    "source_id": src,
                    "target_id": tgt,
                    "rel_type": data.get("rel_type", ""),
                    "properties": data.get("properties", {}),
                    "evidence": data.get("evidence", ""),
                    "confidence": data.get("confidence", 1.0),
                }
            )

        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_json(cls, data: dict) -> KnowledgeGraph:
        """Deserialize a knowledge graph from JSON dict."""
        kg = cls()
        for node in data.get("nodes", []):
            kg.add_entity(
                Entity(
                    id=node["id"],
                    name=node["name"],
                    entity_type=node.get("entity_type", "unknown"),
                    properties=node.get("properties", {}),
                    mentions=node.get("mentions", []),
                )
            )

        for edge in data.get("edges", []):
            kg.add_relationship(
                Relationship(
                    source_id=edge["source_id"],
                    target_id=edge["target_id"],
                    rel_type=edge.get("rel_type", "RELATED_TO"),
                    properties=edge.get("properties", {}),
                    evidence=edge.get("evidence", ""),
                    confidence=edge.get("confidence", 1.0),
                )
            )

        return kg

    def save(self, path: str | Path) -> None:
        """Save graph to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeGraph:
        """Load graph from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_json(data)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_node_by_name(self, name: str) -> str | None:
        """Find a node ID by its name (case-insensitive)."""
        for node_id, data in self._graph.nodes(data=True):
            if data.get("name", "").lower() == name.lower():
                return node_id
        return None
