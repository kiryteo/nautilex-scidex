"""Tests for the end-to-end SciDEX pipeline.

All external dependencies (S2Client, LLM) are mocked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scidex.pipeline import SciDEXPipeline
from scidex.knowledge.accumulator import KnowledgeAccumulator


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


@dataclass
class _MockPaper:
    paper_id: str = "p1"
    title: str = "Mock Paper"
    abstract: str = "Mock abstract about CRISPR and cancer"
    year: int = 2024
    citation_count: int = 10
    authors: list = field(default_factory=lambda: ["Author A"])
    url: str = "https://example.com"
    publication_types: list = field(default_factory=lambda: ["JournalArticle"])
    embedding: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "paperId": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "citation_count": self.citation_count,
            "citationCount": self.citation_count,
            "authors": [{"name": a} for a in self.authors],
        }


def _mock_chat(messages, **kwargs) -> str:
    """Mock LLM that routes based on prompt content."""
    content = messages[-1]["content"] if messages else ""
    # Critic response
    if "Evaluate" in content or "evaluate" in content:
        return json.dumps(
            {
                "novelty": 0.8,
                "feasibility": 0.7,
                "testability": 0.7,
                "significance": 0.8,
                "strengths": ["Good"],
                "weaknesses": ["Weak"],
                "suggested_improvements": ["Improve"],
                "overall_score": 0.75,
                "reasoning": "OK",
            }
        )
    # Evolver response
    if "evolution" in content.lower() or "improved" in content.lower():
        return json.dumps(
            [
                {
                    "statement": "Evolved hypothesis",
                    "supporting_evidence": ["evidence"],
                    "parent_ids": ["h1"],
                    "improvements_applied": ["fix"],
                }
            ]
        )
    # Experiment designer response
    if "Hypothesis:" in content:
        return json.dumps(
            {
                "title": "Mock Experiment",
                "objective": "Test hypothesis",
                "background": "Background",
                "variables": [
                    {
                        "name": "Treatment",
                        "variable_type": "independent",
                        "description": "CRISPR treatment",
                        "measurement_method": "qPCR",
                        "units": "",
                        "levels": ["treated", "control"],
                    }
                ],
                "controls": [
                    {
                        "name": "Negative control",
                        "description": "No treatment",
                        "control_type": "negative",
                        "rationale": "Baseline",
                    }
                ],
                "statistical_plan": {
                    "primary_test": "t-test",
                    "significance_level": 0.05,
                    "power": 0.8,
                    "sample_size_per_group": 10,
                    "sample_size_justification": "Standard",
                    "corrections": [],
                    "secondary_analyses": [],
                },
                "methodology": ["Step 1", "Step 2"],
                "expected_outcomes": ["Outcome 1"],
                "potential_pitfalls": ["Pitfall 1"],
                "timeline_weeks": 8,
                "equipment_needed": ["Microscope"],
                "reagents_needed": ["Buffer"],
                "ethical_considerations": ["IRB approval"],
            }
        )
    # Contradiction mining / testability
    return json.dumps([])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSciDEXPipeline:
    @patch("scidex.pipeline.SciDEXPipeline._search_literature")
    def test_run_returns_expected_keys(self, mock_search):
        mock_search.return_value = []
        pipeline = SciDEXPipeline()
        result = pipeline.run("test topic")
        assert "topic" in result
        assert "papers" in result
        assert "knowledge_graph" in result
        assert "hypotheses" in result
        assert "gde_result" in result
        assert "experiments" in result
        assert "errors" in result

    @patch("scidex.pipeline.SciDEXPipeline._search_literature")
    def test_empty_search_still_completes(self, mock_search):
        mock_search.return_value = []
        pipeline = SciDEXPipeline()
        result = pipeline.run("obscure topic xyz")
        # Should complete without crashing
        assert result["topic"] == "obscure topic xyz"
        assert result["hypotheses"] == []

    @patch("scidex.pipeline.SciDEXPipeline._search_literature")
    @patch("scidex.pipeline.SciDEXPipeline._build_knowledge_graph")
    def test_kg_failure_continues(self, mock_kg, mock_search):
        mock_search.return_value = [_MockPaper()]
        mock_kg.return_value = None
        pipeline = SciDEXPipeline()
        result = pipeline.run("test")
        # Hypothesis generation should skip gracefully
        assert result["hypotheses"] == []

    def test_progress_callback_invoked(self):
        messages: list[str] = []

        pipeline = SciDEXPipeline(on_progress=lambda m: messages.append(m))

        with patch.object(pipeline, "_search_literature", return_value=[]):
            pipeline.run("test")

        # At minimum, the literature search progress is called
        assert len(messages) >= 1

    def test_with_knowledge_accumulator(self, tmp_path: Path):
        acc = KnowledgeAccumulator(storage_path=tmp_path / "knowledge.json")
        pipeline = SciDEXPipeline(knowledge_accumulator=acc)

        # Mock the stages to inject controlled data
        with patch.object(pipeline, "_search_literature", return_value=[]) as mock_search:
            mock_search.side_effect = lambda topic, max_papers, result: (
                result.update({"papers": [{"paper_id": "p1", "title": "Test Paper"}]}),
                [],
            )[1]
            with patch.object(pipeline, "_build_knowledge_graph", return_value=None):
                pipeline.run("test")

        snap = acc.get_snapshot()
        assert snap.total_papers == 1

    def test_pipeline_with_chat_fn(self, tmp_path: Path):
        acc = KnowledgeAccumulator(storage_path=tmp_path / "knowledge.json")
        pipeline = SciDEXPipeline(
            knowledge_accumulator=acc,
            chat_fn=_mock_chat,
        )

        mock_papers = [_MockPaper(paper_id="p1"), _MockPaper(paper_id="p2")]

        with patch.object(pipeline, "_search_literature", return_value=mock_papers) as mock_search:
            mock_search.side_effect = lambda topic, max_papers, result: (
                result.update({"papers": [p.to_dict() for p in mock_papers]}),
                mock_papers,
            )[1]
            result = pipeline.run("CRISPR cancer", max_papers=5, gde_rounds=1)

        # Should have built a KG and generated hypotheses
        assert result["knowledge_graph"] is not None
        assert len(result["hypotheses"]) > 0

    def test_pipeline_accumulates_all_layers(self, tmp_path: Path):
        acc = KnowledgeAccumulator(storage_path=tmp_path / "knowledge.json")
        pipeline = SciDEXPipeline(knowledge_accumulator=acc)

        # Manually build a result dict and test accumulation
        result = {
            "topic": "test",
            "papers": [{"paper_id": "p1", "title": "Paper"}],
            "knowledge_graph": {
                "nodes": [{"id": "g1", "name": "BRCA1", "entity_type": "gene", "properties": {}}],
                "edges": [],
            },
            "hypotheses": [{"id": "h1", "statement": "Hypothesis", "hypothesis_type": "gap"}],
            "gde_result": None,
            "experiments": [{"title": "Experiment", "hypothesis_id": "h1"}],
            "errors": [],
        }
        pipeline._accumulate(result)

        snap = acc.get_snapshot()
        assert snap.total_papers == 1
        assert snap.total_entities == 1
        assert snap.total_hypotheses == 1
        assert snap.total_experiments == 1

    @patch("scidex.pipeline.SciDEXPipeline._search_literature")
    def test_literature_failure_recorded_in_errors(self, mock_search):
        mock_search.side_effect = lambda topic, max_papers, result: (
            result["errors"].append("Literature search failed: ConnectionError"),
            [],
        )[1]
        pipeline = SciDEXPipeline()
        result = pipeline.run("test")
        assert any("Literature search failed" in e for e in result["errors"])

    def test_default_init(self):
        pipeline = SciDEXPipeline()
        assert pipeline.knowledge is None
        assert pipeline.on_progress is None
        assert pipeline._chat_fn is None

    def test_init_with_all_params(self, tmp_path: Path):
        acc = KnowledgeAccumulator(storage_path=tmp_path / "k.json")
        progress_fn = lambda m: None
        chat_fn = lambda m, **kw: ""
        pipeline = SciDEXPipeline(
            knowledge_accumulator=acc,
            on_progress=progress_fn,
            chat_fn=chat_fn,
        )
        assert pipeline.knowledge is acc
        assert pipeline.on_progress is progress_fn
        assert pipeline._chat_fn is chat_fn
