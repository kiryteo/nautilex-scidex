"""Tests for GapDetector — knowledge-graph gap analysis."""

from __future__ import annotations

import pytest

from scidex.knowledge_graph.models import Entity, Relationship
from scidex.knowledge_graph.graph import KnowledgeGraph
from scidex.hypothesis.gap_detector import GapDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector():
    return GapDetector()


@pytest.fixture
def kg_with_gaps():
    """Build a graph with known gaps:

    Cluster 1: CRISPR --RELATED_TO--> BRCA1 --RELATED_TO--> breast cancer
    Cluster 2: RNA-seq --RELATED_TO--> TP53 --RELATED_TO--> lung cancer

    Clusters are disconnected — gap detector should find bridge opportunities.
    Also: BRCA1 has high degree (well-studied), TP53 is connected but lung cancer
    has low degree (understudied seed).
    """
    kg = KnowledgeGraph()

    entities = [
        Entity(id="m1", name="CRISPR", entity_type="method"),
        Entity(id="g1", name="BRCA1", entity_type="gene"),
        Entity(id="d1", name="breast cancer", entity_type="disease"),
        Entity(id="m2", name="RNA-seq", entity_type="method"),
        Entity(id="g2", name="TP53", entity_type="gene"),
        Entity(id="d2", name="lung cancer", entity_type="disease"),
        # Extra connections to make BRCA1 well-studied
        Entity(id="p1", name="Paper 1", entity_type="paper"),
        Entity(id="p2", name="Paper 2", entity_type="paper"),
    ]
    for e in entities:
        kg.add_entity(e)

    rels = [
        # Cluster 1
        Relationship(source_id="m1", target_id="g1", rel_type="RELATED_TO"),
        Relationship(source_id="g1", target_id="d1", rel_type="RELATED_TO"),
        Relationship(source_id="g1", target_id="p1", rel_type="RELATED_TO"),
        Relationship(source_id="g1", target_id="p2", rel_type="RELATED_TO"),
        # Cluster 2
        Relationship(source_id="m2", target_id="g2", rel_type="RELATED_TO"),
        Relationship(source_id="g2", target_id="d2", rel_type="RELATED_TO"),
    ]
    for r in rels:
        kg.add_relationship(r)

    return kg


@pytest.fixture
def kg_connected():
    """A single connected graph with shared-neighbour gaps.

    A -- B -- C
    A -- D -- C  (A and C share B and D but have no direct edge)
    """
    kg = KnowledgeGraph()
    for eid, name in [("a", "GeneA"), ("b", "GeneB"), ("c", "GeneC"), ("d", "GeneD")]:
        kg.add_entity(Entity(id=eid, name=name, entity_type="gene"))
    for src, tgt in [("a", "b"), ("b", "c"), ("a", "d"), ("d", "c")]:
        kg.add_relationship(Relationship(source_id=src, target_id=tgt, rel_type="RELATED_TO"))
    return kg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDetectGaps:
    def test_detects_understudied(self, detector, kg_with_gaps):
        gap = detector.detect_gaps(kg_with_gaps, "gene therapy")
        # lung cancer has degree 1 and is a disease → understudied
        assert any("lung cancer" in name for name in gap.understudied)

    def test_detects_well_studied(self, detector, kg_with_gaps):
        gap = detector.detect_gaps(kg_with_gaps, "gene therapy")
        # BRCA1 has degree >= 3 → well studied
        assert "BRCA1" in gap.well_studied

    def test_produces_suggested_directions(self, detector, kg_with_gaps):
        gap = detector.detect_gaps(kg_with_gaps, "gene therapy")
        assert len(gap.suggested_directions) > 0

    def test_topic_is_preserved(self, detector, kg_with_gaps):
        gap = detector.detect_gaps(kg_with_gaps, "gene therapy")
        assert gap.topic == "gene therapy"

    def test_empty_graph(self, detector):
        kg = KnowledgeGraph()
        gap = detector.detect_gaps(kg, "nothing")
        assert gap.well_studied == []
        assert gap.understudied == []
        assert gap.connections_missing == []


class TestFindMissingConnections:
    def test_shared_neighbour_heuristic(self, detector, kg_connected):
        gap = detector.detect_gaps(kg_connected, "genetics")
        # GeneA and GeneC share GeneB and GeneD → missing connection
        names = {tuple(sorted(pair)) for pair in gap.connections_missing}
        assert ("GeneA", "GeneC") in names or ("GeneC", "GeneA") in names


class TestBridgeOpportunities:
    def test_disconnected_clusters(self, detector, kg_with_gaps):
        bridges = detector.find_bridge_opportunities(kg_with_gaps)
        assert len(bridges) >= 1
        # Each bridge is (name, name, reasoning)
        for name1, name2, reason in bridges:
            assert isinstance(name1, str)
            assert isinstance(name2, str)
            assert len(reason) > 0

    def test_connected_graph_uses_communities(self, detector, kg_connected):
        bridges = detector.find_bridge_opportunities(kg_connected)
        # May or may not find bridges depending on community detection,
        # but should not crash
        assert isinstance(bridges, list)
