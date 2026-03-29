"""Tests for the RelationshipBuilder."""

from __future__ import annotations

import pytest

from scidex.knowledge_graph.models import Entity, Relationship
from scidex.knowledge_graph.extractor import EntityExtractor, _make_id
from scidex.knowledge_graph.relations import RelationshipBuilder


@pytest.fixture
def builder():
    return RelationshipBuilder()


@pytest.fixture
def extractor():
    return EntityExtractor()


# ---------------------------------------------------------------------------
# build_from_paper — structural relationships
# ---------------------------------------------------------------------------


class TestBuildFromPaper:
    def test_author_wrote_paper(self, builder, extractor):
        meta = {
            "paperId": "p1",
            "title": "Test Paper",
            "authors": [{"name": "Alice", "authorId": "a1"}],
        }
        entities = extractor.extract_from_paper(meta)
        rels = builder.build_from_paper(meta, entities)
        wrote = [r for r in rels if r.rel_type == "WROTE"]
        assert len(wrote) == 1
        assert wrote[0].source_id == "a1"
        assert wrote[0].target_id == "p1"

    def test_paper_published_in_venue(self, builder, extractor):
        meta = {
            "paperId": "p1",
            "title": "Paper",
            "venue": "Nature",
            "authors": [],
        }
        entities = extractor.extract_from_paper(meta)
        rels = builder.build_from_paper(meta, entities)
        pub = [r for r in rels if r.rel_type == "PUBLISHED_IN"]
        assert len(pub) == 1
        assert pub[0].source_id == "p1"

    def test_paper_cites_references(self, builder, extractor):
        meta = {
            "paperId": "p1",
            "title": "Paper",
            "authors": [],
            "references": [
                {"paperId": "ref1"},
                {"paperId": "ref2"},
            ],
        }
        entities = extractor.extract_from_paper(meta)
        rels = builder.build_from_paper(meta, entities)
        cites = [r for r in rels if r.rel_type == "CITES"]
        assert len(cites) == 2
        cited_ids = {r.target_id for r in cites}
        assert cited_ids == {"ref1", "ref2"}

    def test_author_affiliated_with_institution(self, builder, extractor):
        meta = {
            "paperId": "p1",
            "title": "Paper",
            "authors": [
                {"name": "Alice", "authorId": "a1", "affiliations": ["MIT"]},
            ],
        }
        entities = extractor.extract_from_paper(meta)
        rels = builder.build_from_paper(meta, entities)
        aff = [r for r in rels if r.rel_type == "AFFILIATED_WITH"]
        assert len(aff) == 1
        assert aff[0].source_id == "a1"

    def test_no_paper_id_returns_empty(self, builder):
        rels = builder.build_from_paper({}, [])
        assert rels == []

    def test_references_as_strings(self, builder, extractor):
        meta = {
            "paperId": "p1",
            "title": "Paper",
            "authors": [],
            "references": ["ref1", "ref2"],
        }
        entities = extractor.extract_from_paper(meta)
        rels = builder.build_from_paper(meta, entities)
        cites = [r for r in rels if r.rel_type == "CITES"]
        assert len(cites) == 2


# ---------------------------------------------------------------------------
# build_from_cooccurrence — sentence co-occurrence
# ---------------------------------------------------------------------------


class TestBuildFromCooccurrence:
    def test_cooccurrence_same_sentence(self, builder):
        e1 = Entity(id="g1", name="BRCA1", entity_type="gene")
        e2 = Entity(id="d1", name="breast cancer", entity_type="disease")
        text = "BRCA1 is implicated in breast cancer."
        rels = builder.build_from_cooccurrence([e1, e2], text)
        assert len(rels) == 1
        assert rels[0].rel_type == "RELATED_TO"
        assert rels[0].confidence == 0.7

    def test_no_cooccurrence_different_sentences(self, builder):
        e1 = Entity(id="g1", name="BRCA1", entity_type="gene")
        e2 = Entity(id="m1", name="CRISPR", entity_type="method")
        text = "BRCA1 is a tumor suppressor gene. CRISPR is a gene editing tool."
        rels = builder.build_from_cooccurrence([e1, e2], text)
        assert len(rels) == 0

    def test_empty_text(self, builder):
        e1 = Entity(id="g1", name="BRCA1", entity_type="gene")
        rels = builder.build_from_cooccurrence([e1], "")
        assert rels == []

    def test_single_entity(self, builder):
        e1 = Entity(id="g1", name="BRCA1", entity_type="gene")
        rels = builder.build_from_cooccurrence([e1], "BRCA1 is important.")
        assert rels == []

    def test_no_duplicate_pairs(self, builder):
        e1 = Entity(id="g1", name="BRCA1", entity_type="gene")
        e2 = Entity(id="d1", name="cancer", entity_type="disease")
        text = "BRCA1 causes cancer. BRCA1 also drives cancer."
        rels = builder.build_from_cooccurrence([e1, e2], text)
        # Should only create one RELATED_TO edge, not two
        assert len(rels) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
