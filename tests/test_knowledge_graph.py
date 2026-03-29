"""Tests for the KnowledgeGraph class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scidex.knowledge_graph.models import Entity, Relationship, KnowledgeGraphStats
from scidex.knowledge_graph.graph import KnowledgeGraph


@pytest.fixture
def kg():
    return KnowledgeGraph()


@pytest.fixture
def sample_paper():
    return {
        "paperId": "p1",
        "title": "CRISPR-based gene therapy for breast cancer",
        "abstract": (
            "We applied CRISPR to edit BRCA1 in breast cancer cell lines. "
            "Results show that TP53 expression was restored."
        ),
        "year": 2024,
        "citationCount": 42,
        "authors": [
            {"name": "Alice Smith", "authorId": "a1"},
            {"name": "Bob Jones", "authorId": "a2"},
        ],
        "venue": "Nature Methods",
        "references": [{"paperId": "ref1"}, {"paperId": "ref2"}],
    }


# ---------------------------------------------------------------------------
# add_entity / add_relationship
# ---------------------------------------------------------------------------


class TestAddOperations:
    def test_add_entity(self, kg):
        e = Entity(id="e1", name="BRCA1", entity_type="gene")
        kg.add_entity(e)
        stats = kg.get_statistics()
        assert stats.num_nodes == 1
        assert stats.node_types["gene"] == 1

    def test_add_relationship(self, kg):
        e1 = Entity(id="a1", name="Alice", entity_type="author")
        e2 = Entity(id="p1", name="Paper", entity_type="paper")
        kg.add_entity(e1)
        kg.add_entity(e2)
        rel = Relationship(source_id="a1", target_id="p1", rel_type="WROTE")
        kg.add_relationship(rel)
        stats = kg.get_statistics()
        assert stats.num_edges == 1
        assert stats.edge_types["WROTE"] == 1

    def test_add_relationship_creates_missing_nodes(self, kg):
        """Relationship endpoints should be auto-created if missing."""
        rel = Relationship(source_id="x1", target_id="x2", rel_type="RELATED_TO")
        kg.add_relationship(rel)
        stats = kg.get_statistics()
        assert stats.num_nodes == 2
        assert stats.num_edges == 1


# ---------------------------------------------------------------------------
# add_paper — full pipeline
# ---------------------------------------------------------------------------


class TestAddPaper:
    def test_add_paper_creates_entities_and_relations(self, kg, sample_paper):
        kg.add_paper(sample_paper)
        stats = kg.get_statistics()
        # Should have: paper, 2 authors, venue, + abstract entities (genes, methods, diseases)
        assert stats.num_nodes >= 4
        assert stats.num_edges >= 3  # 2x WROTE + PUBLISHED_IN + CITES + co-occurrence

    def test_add_paper_includes_references(self, kg, sample_paper):
        kg.add_paper(sample_paper)
        stats = kg.get_statistics()
        assert "CITES" in stats.edge_types
        assert stats.edge_types["CITES"] == 2


# ---------------------------------------------------------------------------
# query_entity
# ---------------------------------------------------------------------------


class TestQueryEntity:
    def test_query_existing_entity(self, kg, sample_paper):
        kg.add_paper(sample_paper)
        result = kg.query_entity("Alice Smith")
        assert result != {}
        assert result["entity"]["name"] == "Alice Smith"
        assert result["entity"]["entity_type"] == "author"
        # Alice should have a WROTE edge outgoing
        assert len(result["outgoing"]) >= 1

    def test_query_nonexistent_entity(self, kg):
        result = kg.query_entity("Nonexistent")
        assert result == {}

    def test_query_case_insensitive(self, kg):
        e = Entity(id="e1", name="BRCA1", entity_type="gene")
        kg.add_entity(e)
        result = kg.query_entity("brca1")
        assert result != {}
        assert result["entity"]["name"] == "BRCA1"


# ---------------------------------------------------------------------------
# find_path
# ---------------------------------------------------------------------------


class TestFindPath:
    def test_find_path_between_connected_entities(self, kg, sample_paper):
        kg.add_paper(sample_paper)
        # Alice -> paper -> Nature Methods
        path = kg.find_path("Alice Smith", "Nature Methods")
        assert len(path) >= 2
        assert path[0] == "Alice Smith"
        assert path[-1] == "Nature Methods"

    def test_find_path_no_connection(self, kg):
        e1 = Entity(id="e1", name="A", entity_type="gene")
        e2 = Entity(id="e2", name="B", entity_type="gene")
        kg.add_entity(e1)
        kg.add_entity(e2)
        path = kg.find_path("A", "B")
        assert path == []

    def test_find_path_nonexistent_entity(self, kg):
        path = kg.find_path("X", "Y")
        assert path == []


# ---------------------------------------------------------------------------
# get_subgraph
# ---------------------------------------------------------------------------


class TestGetSubgraph:
    def test_subgraph_returns_neighborhood(self, kg, sample_paper):
        kg.add_paper(sample_paper)
        subgraph = kg.get_subgraph("p1", depth=1)
        assert len(subgraph["nodes"]) >= 1
        assert len(subgraph["edges"]) >= 1

    def test_subgraph_nonexistent_entity(self, kg):
        result = kg.get_subgraph("nonexistent")
        assert result == {"nodes": [], "edges": []}

    def test_subgraph_depth_limits_results(self, kg):
        # Build a chain: a -> b -> c -> d
        for eid, name in [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")]:
            kg.add_entity(Entity(id=eid, name=name, entity_type="gene"))
        for src, tgt in [("a", "b"), ("b", "c"), ("c", "d")]:
            kg.add_relationship(Relationship(source_id=src, target_id=tgt, rel_type="RELATED_TO"))
        sub_d1 = kg.get_subgraph("a", depth=1)
        sub_d2 = kg.get_subgraph("a", depth=2)
        # depth=1 should include a, b; depth=2 should include a, b, c
        assert len(sub_d1["nodes"]) == 2
        assert len(sub_d2["nodes"]) == 3


# ---------------------------------------------------------------------------
# get_statistics
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_empty_graph(self, kg):
        stats = kg.get_statistics()
        assert stats.num_nodes == 0
        assert stats.num_edges == 0
        assert stats.node_types == {}
        assert stats.edge_types == {}

    def test_stats_after_add_paper(self, kg, sample_paper):
        kg.add_paper(sample_paper)
        stats = kg.get_statistics()
        assert stats.num_nodes > 0
        assert stats.num_edges > 0
        assert "paper" in stats.node_types
        assert "author" in stats.node_types


# ---------------------------------------------------------------------------
# Serialization: to_json / from_json / save / load
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_json_roundtrip(self, kg, sample_paper):
        kg.add_paper(sample_paper)
        data = kg.to_json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) > 0

        # Roundtrip
        kg2 = KnowledgeGraph.from_json(data)
        stats1 = kg.get_statistics()
        stats2 = kg2.get_statistics()
        assert stats1.num_nodes == stats2.num_nodes
        assert stats1.num_edges == stats2.num_edges

    def test_save_and_load(self, kg, sample_paper):
        kg.add_paper(sample_paper)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            kg.save(path)
            assert path.exists()

            kg2 = KnowledgeGraph.load(path)
            assert kg2.get_statistics().num_nodes == kg.get_statistics().num_nodes

    def test_from_json_empty(self):
        kg = KnowledgeGraph.from_json({"nodes": [], "edges": []})
        assert kg.get_statistics().num_nodes == 0

    def test_json_is_valid(self, kg, sample_paper):
        kg.add_paper(sample_paper)
        data = kg.to_json()
        # Should be JSON-serializable
        json_str = json.dumps(data)
        assert json.loads(json_str) == data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
