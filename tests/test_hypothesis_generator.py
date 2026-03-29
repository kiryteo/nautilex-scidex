"""Tests for HypothesisGenerator — full pipeline integration."""

from __future__ import annotations

import json

import pytest

from scidex.knowledge_graph.models import Entity, Relationship
from scidex.knowledge_graph.graph import KnowledgeGraph
from scidex.hypothesis.generator import HypothesisGenerator
from scidex.hypothesis.models import HypothesisReport


# ---------------------------------------------------------------------------
# Mock LLM for contradiction mining + refinement
# ---------------------------------------------------------------------------


def _mock_chat(messages, **kwargs) -> str:
    """Route mock responses based on prompt content."""
    content = messages[-1]["content"] if messages else ""

    # Contradiction mining prompt
    if "Identify contradictions" in content:
        return json.dumps(
            [
                {
                    "paper_a": "Paper Alpha",
                    "paper_b": "Paper Beta",
                    "claim_a": "Method A works",
                    "claim_b": "Method A does not work",
                    "resolution_direction": "Context-dependent efficacy of Method A",
                }
            ]
        )

    # Testability refinement prompt (JSON array of hypothesis summaries)
    try:
        data = json.loads(content)
        if isinstance(data, list) and data and "id" in data[0]:
            return json.dumps([{"id": item["id"], "testability_score": 0.9} for item in data])
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    return "[]"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kg_full():
    """A rich graph for the full pipeline.

    method: CRISPR, RNA-seq
    genes: BRCA1, TP53
    diseases: breast cancer, lung cancer
    compounds: fish oil

    Connections designed to trigger:
    - Gap detection (lung cancer is understudied)
    - Swanson links (fish oil → inflammation → lung cancer)
    - Analogy (CRISPR for BRCA1/breast cancer but not TP53/lung cancer)
    """
    kg = KnowledgeGraph()
    entities = [
        Entity(id="m1", name="CRISPR", entity_type="method"),
        Entity(id="m2", name="RNA-seq", entity_type="method"),
        Entity(id="g1", name="BRCA1", entity_type="gene"),
        Entity(id="g2", name="TP53", entity_type="gene"),
        Entity(id="d1", name="breast cancer", entity_type="disease"),
        Entity(id="d2", name="lung cancer", entity_type="disease"),
        Entity(id="c1", name="fish oil", entity_type="compound"),
        Entity(id="x1", name="inflammation", entity_type="disease"),
        # Extra nodes for connectivity
        Entity(id="p1", name="Paper 1", entity_type="paper"),
        Entity(id="p2", name="Paper 2", entity_type="paper"),
    ]
    for e in entities:
        kg.add_entity(e)

    rels = [
        # CRISPR cluster
        Relationship(source_id="m1", target_id="g1", rel_type="USES_METHOD"),
        Relationship(source_id="g1", target_id="d1", rel_type="RELATED_TO"),
        Relationship(source_id="g1", target_id="p1", rel_type="RELATED_TO"),
        Relationship(source_id="g1", target_id="p2", rel_type="RELATED_TO"),
        # TP53 cluster
        Relationship(source_id="g2", target_id="d2", rel_type="RELATED_TO"),
        Relationship(source_id="g1", target_id="g2", rel_type="RELATED_TO"),
        # Swanson chain: fish oil → inflammation → lung cancer
        Relationship(source_id="c1", target_id="x1", rel_type="RELATED_TO"),
        Relationship(source_id="x1", target_id="d2", rel_type="RELATED_TO"),
        # RNA-seq connected to BRCA1 only
        Relationship(source_id="m2", target_id="g1", rel_type="USES_METHOD"),
    ]
    for r in rels:
        kg.add_relationship(r)

    return kg


SAMPLE_PAPERS = [
    {"title": "Paper Alpha", "abstract": "Method A works on disease X."},
    {"title": "Paper Beta", "abstract": "Method A does not work on disease X."},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHypothesisGenerator:
    def test_generate_returns_report(self, kg_full):
        gen = HypothesisGenerator(chat_fn=_mock_chat)
        report = gen.generate(kg_full, "gene therapy")
        assert isinstance(report, HypothesisReport)
        assert report.topic == "gene therapy"

    def test_generates_hypotheses(self, kg_full):
        gen = HypothesisGenerator(chat_fn=_mock_chat)
        report = gen.generate(kg_full, "gene therapy")
        assert len(report.hypotheses) > 0

    def test_gap_filling_hypotheses_present(self, kg_full):
        gen = HypothesisGenerator(chat_fn=_mock_chat)
        report = gen.generate(kg_full, "gene therapy")
        types = {h.hypothesis_type for h in report.hypotheses}
        assert "gap_filling" in types

    def test_swanson_hypotheses_present(self, kg_full):
        gen = HypothesisGenerator(chat_fn=_mock_chat)
        report = gen.generate(kg_full, "fish oil", source_entity="fish oil")
        types = {h.hypothesis_type for h in report.hypotheses}
        assert "combination" in types

    def test_analogy_hypotheses_when_method_given(self, kg_full):
        gen = HypothesisGenerator(chat_fn=_mock_chat)
        report = gen.generate(kg_full, "gene therapy", source_method="CRISPR")
        types = {h.hypothesis_type for h in report.hypotheses}
        assert "analogy_transfer" in types

    def test_contradiction_hypotheses_when_papers_given(self, kg_full):
        gen = HypothesisGenerator(chat_fn=_mock_chat)
        report = gen.generate(kg_full, "gene therapy", papers=SAMPLE_PAPERS)
        types = {h.hypothesis_type for h in report.hypotheses}
        assert "contradiction_resolution" in types

    def test_knowledge_gaps_populated(self, kg_full):
        gen = HypothesisGenerator(chat_fn=_mock_chat)
        report = gen.generate(kg_full, "gene therapy")
        assert len(report.knowledge_gaps) > 0

    def test_summary_not_empty(self, kg_full):
        gen = HypothesisGenerator(chat_fn=_mock_chat)
        report = gen.generate(kg_full, "gene therapy")
        assert len(report.summary) > 0

    def test_hypotheses_sorted_by_confidence(self, kg_full):
        gen = HypothesisGenerator(chat_fn=_mock_chat)
        report = gen.generate(kg_full, "gene therapy")
        confidences = [h.confidence for h in report.hypotheses]
        assert confidences == sorted(confidences, reverse=True)

    def test_refine_with_llm(self, kg_full):
        gen = HypothesisGenerator(chat_fn=_mock_chat)
        report = gen.generate(kg_full, "gene therapy", refine_with_llm=True)
        # After refinement, some hypotheses should have testability_score=0.9
        scores = {h.testability_score for h in report.hypotheses}
        assert 0.9 in scores

    def test_no_chat_fn_skips_contradictions(self, kg_full):
        gen = HypothesisGenerator(chat_fn=None)
        # Should not crash even without LLM — just skips contradiction step
        report = gen.generate(kg_full, "gene therapy")
        assert isinstance(report, HypothesisReport)

    def test_empty_graph(self):
        gen = HypothesisGenerator(chat_fn=_mock_chat)
        kg = KnowledgeGraph()
        report = gen.generate(kg, "empty topic")
        assert isinstance(report, HypothesisReport)
        assert report.topic == "empty topic"
