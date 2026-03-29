"""Tests for AnalogyEngine — cross-domain analogy discovery."""

from __future__ import annotations

import pytest

from scidex.knowledge_graph.models import Entity, Relationship
from scidex.knowledge_graph.graph import KnowledgeGraph
from scidex.hypothesis.analogy_engine import AnalogyEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return AnalogyEngine()


@pytest.fixture
def kg_analogy():
    """Graph with a known analogy opportunity.

    CRISPR (method) -- BRCA1 (gene) -- breast cancer (disease)
    BRCA1 (gene) -- TP53 (gene)
    TP53 (gene) -- lung cancer (disease)

    CRISPR is connected to BRCA1 (domain), BRCA1 connects to TP53,
    TP53 connects to lung cancer → CRISPR could be applied to lung cancer.
    """
    kg = KnowledgeGraph()
    entities = [
        Entity(id="m1", name="CRISPR", entity_type="method"),
        Entity(id="g1", name="BRCA1", entity_type="gene"),
        Entity(id="g2", name="TP53", entity_type="gene"),
        Entity(id="d1", name="breast cancer", entity_type="disease"),
        Entity(id="d2", name="lung cancer", entity_type="disease"),
    ]
    for e in entities:
        kg.add_entity(e)

    rels = [
        Relationship(source_id="m1", target_id="g1", rel_type="USES_METHOD"),
        Relationship(source_id="g1", target_id="d1", rel_type="RELATED_TO"),
        Relationship(source_id="g1", target_id="g2", rel_type="RELATED_TO"),
        Relationship(source_id="g2", target_id="d2", rel_type="RELATED_TO"),
    ]
    for r in rels:
        kg.add_relationship(r)
    return kg


@pytest.fixture
def kg_no_analogy():
    """Graph where method is connected to all domains — no analogy candidates."""
    kg = KnowledgeGraph()
    entities = [
        Entity(id="m1", name="PCR", entity_type="method"),
        Entity(id="g1", name="GeneA", entity_type="gene"),
        Entity(id="g2", name="GeneB", entity_type="gene"),
    ]
    for e in entities:
        kg.add_entity(e)
    rels = [
        Relationship(source_id="m1", target_id="g1", rel_type="RELATED_TO"),
        Relationship(source_id="m1", target_id="g2", rel_type="RELATED_TO"),
    ]
    for r in rels:
        kg.add_relationship(r)
    return kg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFindAnalogies:
    def test_finds_cross_domain_analogy(self, engine, kg_analogy):
        results = engine.find_analogies(kg_analogy, "CRISPR")
        assert len(results) >= 1
        statements = " ".join(h.statement for h in results)
        # Should suggest CRISPR for TP53 or lung cancer
        assert "CRISPR" in statements

    def test_hypothesis_type_is_analogy(self, engine, kg_analogy):
        results = engine.find_analogies(kg_analogy, "CRISPR")
        for h in results:
            assert h.hypothesis_type == "analogy_transfer"

    def test_evidence_populated(self, engine, kg_analogy):
        results = engine.find_analogies(kg_analogy, "CRISPR")
        for h in results:
            assert len(h.supporting_evidence) > 0

    def test_confidence_between_0_and_1(self, engine, kg_analogy):
        results = engine.find_analogies(kg_analogy, "CRISPR")
        for h in results:
            assert 0.0 <= h.confidence <= 1.0

    def test_no_analogy_when_all_connected(self, engine, kg_no_analogy):
        results = engine.find_analogies(kg_no_analogy, "PCR")
        assert results == []

    def test_unknown_method_returns_empty(self, engine, kg_analogy):
        results = engine.find_analogies(kg_analogy, "nonexistent_method")
        assert results == []

    def test_sorted_by_confidence(self, engine, kg_analogy):
        results = engine.find_analogies(kg_analogy, "CRISPR")
        if len(results) >= 2:
            assert results[0].confidence >= results[1].confidence
