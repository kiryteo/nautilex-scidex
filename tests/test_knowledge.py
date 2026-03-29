"""Tests for the knowledge accumulation system.

Covers: KnowledgeEntry, KnowledgeSnapshot, KnowledgeLayer models,
and KnowledgeAccumulator persistence, search, and session tracking.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scidex.knowledge.models import KnowledgeEntry, KnowledgeLayer, KnowledgeSnapshot
from scidex.knowledge.accumulator import KnowledgeAccumulator


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestKnowledgeModels:
    def test_knowledge_layer_values(self):
        assert KnowledgeLayer.RAW == "raw"
        assert KnowledgeLayer.DOMAIN == "domain"
        assert KnowledgeLayer.OUTCOMES == "outcomes"
        assert KnowledgeLayer.META == "meta"

    def test_knowledge_entry_creation(self):
        entry = KnowledgeEntry(
            id="e1",
            layer=KnowledgeLayer.RAW,
            content="Test paper title",
            source="paper_123",
        )
        assert entry.id == "e1"
        assert entry.layer == KnowledgeLayer.RAW
        assert entry.content == "Test paper title"
        assert entry.source == "paper_123"
        assert entry.tags == []
        assert entry.metadata == {}

    def test_knowledge_entry_with_all_fields(self):
        entry = KnowledgeEntry(
            id="e2",
            layer=KnowledgeLayer.DOMAIN,
            content="BRCA1",
            source="gene_id_123",
            tags=["entity", "gene"],
            created_at="2025-01-01T00:00:00Z",
            metadata={"entity_type": "gene"},
        )
        assert entry.tags == ["entity", "gene"]
        assert entry.metadata["entity_type"] == "gene"

    def test_knowledge_entry_serialization_round_trip(self):
        entry = KnowledgeEntry(
            id="e3",
            layer=KnowledgeLayer.OUTCOMES,
            content="Hypothesis statement",
            source="h_123",
            tags=["hypothesis"],
        )
        data = entry.model_dump()
        restored = KnowledgeEntry(**data)
        assert restored.id == entry.id
        assert restored.layer == entry.layer
        assert restored.content == entry.content

    def test_knowledge_snapshot_defaults(self):
        snap = KnowledgeSnapshot()
        assert snap.entries == []
        assert snap.total_papers == 0
        assert snap.total_entities == 0
        assert snap.total_hypotheses == 0
        assert snap.total_experiments == 0
        assert snap.sessions == []

    def test_knowledge_snapshot_with_data(self):
        entry = KnowledgeEntry(id="e1", layer=KnowledgeLayer.RAW, content="Paper", source="p1")
        snap = KnowledgeSnapshot(
            entries=[entry],
            total_papers=1,
            total_entities=5,
            total_hypotheses=3,
            total_experiments=1,
            sessions=[{"topic": "cancer"}],
        )
        assert len(snap.entries) == 1
        assert snap.total_papers == 1
        assert len(snap.sessions) == 1


# ---------------------------------------------------------------------------
# Accumulator tests
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path: Path) -> KnowledgeAccumulator:
    """Create a KnowledgeAccumulator with a temp storage path."""
    return KnowledgeAccumulator(storage_path=tmp_path / "knowledge.json")


class TestKnowledgeAccumulator:
    def test_init_creates_empty_store(self, tmp_store: KnowledgeAccumulator):
        snap = tmp_store.get_snapshot()
        assert snap.total_papers == 0
        assert snap.total_entities == 0

    def test_add_papers(self, tmp_store: KnowledgeAccumulator):
        papers = [
            {"paper_id": "p1", "title": "Paper One", "abstract": "Abstract one"},
            {"paper_id": "p2", "title": "Paper Two", "abstract": "Abstract two"},
        ]
        added = tmp_store.add_papers(papers)
        assert added == 2
        snap = tmp_store.get_snapshot()
        assert snap.total_papers == 2

    def test_add_papers_dedup(self, tmp_store: KnowledgeAccumulator):
        papers = [{"paper_id": "p1", "title": "Paper One"}]
        tmp_store.add_papers(papers)
        added = tmp_store.add_papers(papers)
        assert added == 0
        snap = tmp_store.get_snapshot()
        assert snap.total_papers == 1

    def test_add_papers_skips_missing_fields(self, tmp_store: KnowledgeAccumulator):
        papers = [
            {"paper_id": "", "title": "No ID"},
            {"paper_id": "p1", "title": ""},
            {"title": "No paper_id field"},
        ]
        added = tmp_store.add_papers(papers)
        assert added == 0

    def test_add_papers_with_s2_keys(self, tmp_store: KnowledgeAccumulator):
        papers = [{"paperId": "s2_123", "title": "S2 Paper", "citationCount": 42}]
        added = tmp_store.add_papers(papers)
        assert added == 1

    def test_add_entities(self, tmp_store: KnowledgeAccumulator):
        entities = [
            {"id": "g1", "name": "BRCA1", "entity_type": "gene"},
            {"id": "d1", "name": "breast cancer", "entity_type": "disease"},
        ]
        added = tmp_store.add_entities(entities)
        assert added == 2
        snap = tmp_store.get_snapshot()
        assert snap.total_entities == 2

    def test_add_entities_dedup(self, tmp_store: KnowledgeAccumulator):
        entities = [{"id": "g1", "name": "BRCA1", "entity_type": "gene"}]
        tmp_store.add_entities(entities)
        added = tmp_store.add_entities(entities)
        assert added == 0

    def test_add_hypothesis(self, tmp_store: KnowledgeAccumulator):
        hyp = {
            "id": "h1",
            "statement": "BRCA1 inhibition reduces tumor growth",
            "hypothesis_type": "gap_filling",
            "confidence": 0.7,
        }
        tmp_store.add_hypothesis(hyp)
        snap = tmp_store.get_snapshot()
        assert snap.total_hypotheses == 1

    def test_add_hypothesis_dedup(self, tmp_store: KnowledgeAccumulator):
        hyp = {"id": "h1", "statement": "Test hypothesis"}
        tmp_store.add_hypothesis(hyp)
        tmp_store.add_hypothesis(hyp)
        snap = tmp_store.get_snapshot()
        assert snap.total_hypotheses == 1

    def test_add_experiment(self, tmp_store: KnowledgeAccumulator):
        protocol = {
            "title": "CRISPR knockout experiment",
            "hypothesis_id": "h1",
            "objective": "Test BRCA1 role",
            "timeline_weeks": 12,
        }
        tmp_store.add_experiment(protocol)
        snap = tmp_store.get_snapshot()
        assert snap.total_experiments == 1

    def test_add_insight(self, tmp_store: KnowledgeAccumulator):
        tmp_store.add_insight("Gene X appears in multiple contexts across sessions")
        snap = tmp_store.get_snapshot()
        meta_entries = [e for e in snap.entries if e.layer == KnowledgeLayer.META]
        assert len(meta_entries) == 1
        assert "Gene X" in meta_entries[0].content

    def test_add_insight_dedup(self, tmp_store: KnowledgeAccumulator):
        tmp_store.add_insight("Pattern A")
        tmp_store.add_insight("Pattern A")
        snap = tmp_store.get_snapshot()
        meta_entries = [e for e in snap.entries if e.layer == KnowledgeLayer.META]
        assert len(meta_entries) == 1

    def test_add_insight_empty_ignored(self, tmp_store: KnowledgeAccumulator):
        tmp_store.add_insight("")
        snap = tmp_store.get_snapshot()
        assert len(snap.entries) == 0

    def test_search_by_keyword(self, tmp_store: KnowledgeAccumulator):
        tmp_store.add_papers([{"paper_id": "p1", "title": "CRISPR gene therapy"}])
        tmp_store.add_papers([{"paper_id": "p2", "title": "RNA sequencing methods"}])
        results = tmp_store.search("CRISPR")
        assert len(results) == 1
        assert "CRISPR" in results[0].content

    def test_search_case_insensitive(self, tmp_store: KnowledgeAccumulator):
        tmp_store.add_papers([{"paper_id": "p1", "title": "Breast Cancer Study"}])
        results = tmp_store.search("breast cancer")
        assert len(results) == 1

    def test_search_by_layer(self, tmp_store: KnowledgeAccumulator):
        tmp_store.add_papers([{"paper_id": "p1", "title": "Paper about cancer"}])
        tmp_store.add_insight("Cancer pattern observed")
        # Search with layer filter
        raw_results = tmp_store.search("cancer", layer=KnowledgeLayer.RAW)
        meta_results = tmp_store.search("cancer", layer=KnowledgeLayer.META)
        assert len(raw_results) == 1
        assert len(meta_results) == 1

    def test_search_no_results(self, tmp_store: KnowledgeAccumulator):
        tmp_store.add_papers([{"paper_id": "p1", "title": "Something else"}])
        results = tmp_store.search("nonexistent_term_xyz")
        assert results == []

    def test_persistence_save_and_load(self, tmp_path: Path):
        store_path = tmp_path / "persist_test.json"

        # Create and populate
        store1 = KnowledgeAccumulator(storage_path=store_path)
        store1.add_papers([{"paper_id": "p1", "title": "Persisted Paper"}])
        store1.add_insight("Persisted insight")

        # Load in a new instance
        store2 = KnowledgeAccumulator(storage_path=store_path)
        snap = store2.get_snapshot()
        assert snap.total_papers == 1
        assert len([e for e in snap.entries if e.layer == KnowledgeLayer.META]) == 1

    def test_persistence_file_created(self, tmp_path: Path):
        store_path = tmp_path / "sub" / "knowledge.json"
        store = KnowledgeAccumulator(storage_path=store_path)
        store.add_insight("trigger save")
        assert store_path.exists()
        data = json.loads(store_path.read_text())
        assert "entries" in data
        assert "sessions" in data

    def test_session_tracking(self, tmp_store: KnowledgeAccumulator):
        tmp_store.start_session("cancer research")
        sessions = tmp_store.get_session_history()
        assert len(sessions) == 1
        assert sessions[0]["topic"] == "cancer research"
        assert sessions[0]["ended_at"] is None

        tmp_store.end_session("Found 5 hypotheses")
        sessions = tmp_store.get_session_history()
        assert sessions[0]["ended_at"] is not None
        assert sessions[0]["summary"] == "Found 5 hypotheses"

    def test_session_persisted(self, tmp_path: Path):
        store_path = tmp_path / "session_test.json"
        store1 = KnowledgeAccumulator(storage_path=store_path)
        store1.start_session("topic A")
        store1.end_session("done")

        store2 = KnowledgeAccumulator(storage_path=store_path)
        sessions = store2.get_session_history()
        assert len(sessions) == 1
        assert sessions[0]["topic"] == "topic A"

    def test_multiple_sessions(self, tmp_store: KnowledgeAccumulator):
        tmp_store.start_session("topic A")
        tmp_store.end_session("done A")
        tmp_store.start_session("topic B")
        tmp_store.end_session("done B")
        sessions = tmp_store.get_session_history()
        assert len(sessions) == 2

    def test_end_session_without_start(self, tmp_store: KnowledgeAccumulator):
        # Should not crash
        tmp_store.end_session("orphan summary")
        sessions = tmp_store.get_session_history()
        assert len(sessions) == 0

    def test_load_corrupt_file(self, tmp_path: Path):
        store_path = tmp_path / "corrupt.json"
        store_path.write_text("not valid json {{{")
        store = KnowledgeAccumulator(storage_path=store_path)
        snap = store.get_snapshot()
        assert len(snap.entries) == 0

    def test_snapshot_includes_all_layers(self, tmp_store: KnowledgeAccumulator):
        tmp_store.add_papers([{"paper_id": "p1", "title": "Paper"}])
        tmp_store.add_entities([{"id": "g1", "name": "BRCA1", "entity_type": "gene"}])
        tmp_store.add_hypothesis({"id": "h1", "statement": "Hypothesis"})
        tmp_store.add_experiment({"title": "Experiment", "hypothesis_id": "h1"})
        tmp_store.add_insight("Meta insight")

        snap = tmp_store.get_snapshot()
        assert snap.total_papers == 1
        assert snap.total_entities == 1
        assert snap.total_hypotheses == 1
        assert snap.total_experiments == 1
        assert len(snap.entries) == 5
