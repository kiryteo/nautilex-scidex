"""Tests for SwansonLinker — ABC literature-based discovery."""

from __future__ import annotations

import pytest

from scidex.knowledge_graph.models import Entity, Relationship
from scidex.knowledge_graph.graph import KnowledgeGraph
from scidex.hypothesis.swanson_linker import SwansonLinker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linker():
    return SwansonLinker()


@pytest.fixture
def kg_abc():
    """Graph with a known Swanson chain: A -> B -> C, where A and C are not connected.

    fish oil (A) -- inflammation (B) -- Raynaud's disease (C)
    fish oil (A) -- blood viscosity (B2) -- Raynaud's disease (C)

    Classic Swanson example: fish oil may help Raynaud's through inflammation
    and blood viscosity.
    """
    kg = KnowledgeGraph()
    entities = [
        Entity(id="a", name="fish oil", entity_type="compound"),
        Entity(id="b1", name="inflammation", entity_type="disease"),
        Entity(id="b2", name="blood viscosity", entity_type="protein"),
        Entity(id="c", name="Raynaud's disease", entity_type="disease"),
        # A direct connection that should not appear in results
        Entity(id="d", name="omega-3", entity_type="compound"),
    ]
    for e in entities:
        kg.add_entity(e)

    rels = [
        Relationship(source_id="a", target_id="b1", rel_type="RELATED_TO"),
        Relationship(source_id="a", target_id="b2", rel_type="RELATED_TO"),
        Relationship(source_id="b1", target_id="c", rel_type="RELATED_TO"),
        Relationship(source_id="b2", target_id="c", rel_type="RELATED_TO"),
        # Direct connection a-d should be excluded from ABC results
        Relationship(source_id="a", target_id="d", rel_type="RELATED_TO"),
    ]
    for r in rels:
        kg.add_relationship(r)

    return kg


@pytest.fixture
def kg_with_authors():
    """Graph where the only intermediaries are authors/venues — should be filtered."""
    kg = KnowledgeGraph()
    entities = [
        Entity(id="a", name="CRISPR", entity_type="method"),
        Entity(id="auth", name="Dr. Smith", entity_type="author"),
        Entity(id="c", name="p53", entity_type="gene"),
    ]
    for e in entities:
        kg.add_entity(e)
    rels = [
        Relationship(source_id="a", target_id="auth", rel_type="RELATED_TO"),
        Relationship(source_id="auth", target_id="c", rel_type="RELATED_TO"),
    ]
    for r in rels:
        kg.add_relationship(r)
    return kg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiscoverLinks:
    def test_finds_abc_chain(self, linker, kg_abc):
        results = linker.discover_links(kg_abc, "fish oil")
        assert len(results) >= 1
        # Should find Raynaud's disease as C
        statements = " ".join(h.statement for h in results)
        assert "Raynaud's disease" in statements

    def test_confidence_scales_with_intermediaries(self, linker, kg_abc):
        results = linker.discover_links(kg_abc, "fish oil")
        # Raynaud's has 2 intermediaries → higher confidence than 1
        raynaud = [h for h in results if "Raynaud's disease" in h.statement]
        assert len(raynaud) == 1
        assert raynaud[0].confidence > 0.5  # 0.3 + 0.15*2 = 0.6

    def test_hypothesis_type(self, linker, kg_abc):
        results = linker.discover_links(kg_abc, "fish oil")
        for h in results:
            assert h.hypothesis_type == "combination"

    def test_supporting_evidence_populated(self, linker, kg_abc):
        results = linker.discover_links(kg_abc, "fish oil")
        for h in results:
            assert len(h.supporting_evidence) > 0

    def test_filters_author_intermediaries(self, linker, kg_with_authors):
        results = linker.discover_links(kg_with_authors, "CRISPR")
        # Author intermediary should be filtered → no results
        assert len(results) == 0

    def test_unknown_entity_returns_empty(self, linker, kg_abc):
        results = linker.discover_links(kg_abc, "nonexistent_entity")
        assert results == []

    def test_sorted_by_confidence(self, linker, kg_abc):
        results = linker.discover_links(kg_abc, "fish oil")
        if len(results) >= 2:
            assert results[0].confidence >= results[1].confidence

    def test_hypothesis_has_id(self, linker, kg_abc):
        results = linker.discover_links(kg_abc, "fish oil")
        for h in results:
            assert len(h.id) > 0

    def test_novelty_score_present(self, linker, kg_abc):
        results = linker.discover_links(kg_abc, "fish oil")
        for h in results:
            assert 0.0 <= h.novelty_score <= 1.0
