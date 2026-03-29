"""Tests for the EntityExtractor."""

from __future__ import annotations

import pytest

from scidex.knowledge_graph.extractor import EntityExtractor


@pytest.fixture
def extractor():
    return EntityExtractor()


# ---------------------------------------------------------------------------
# extract_from_paper — structured metadata extraction
# ---------------------------------------------------------------------------


class TestExtractFromPaper:
    def test_extracts_paper_entity(self, extractor):
        meta = {"paperId": "abc123", "title": "A Great Paper", "year": 2024}
        entities = extractor.extract_from_paper(meta)
        papers = [e for e in entities if e.entity_type == "paper"]
        assert len(papers) == 1
        assert papers[0].id == "abc123"
        assert papers[0].name == "A Great Paper"
        assert papers[0].properties["year"] == 2024

    def test_extracts_authors(self, extractor):
        meta = {
            "paperId": "p1",
            "title": "Paper",
            "authors": [
                {"name": "Alice Smith", "authorId": "a1"},
                {"name": "Bob Jones", "authorId": "a2"},
            ],
        }
        entities = extractor.extract_from_paper(meta)
        authors = [e for e in entities if e.entity_type == "author"]
        assert len(authors) == 2
        assert authors[0].name == "Alice Smith"
        assert authors[0].id == "a1"
        assert authors[1].name == "Bob Jones"

    def test_extracts_venue(self, extractor):
        meta = {"paperId": "p1", "title": "Paper", "venue": "Nature"}
        entities = extractor.extract_from_paper(meta)
        venues = [e for e in entities if e.entity_type == "venue"]
        assert len(venues) == 1
        assert venues[0].name == "Nature"

    def test_authors_as_strings(self, extractor):
        meta = {"paperId": "p1", "title": "Paper", "authors": ["Alice", "Bob"]}
        entities = extractor.extract_from_paper(meta)
        authors = [e for e in entities if e.entity_type == "author"]
        assert len(authors) == 2
        assert authors[0].name == "Alice"

    def test_empty_metadata(self, extractor):
        entities = extractor.extract_from_paper({})
        assert entities == []

    def test_uses_paper_id_key_variant(self, extractor):
        """Supports both 'paperId' and 'paper_id' keys."""
        meta = {"paper_id": "x1", "title": "Variant"}
        entities = extractor.extract_from_paper(meta)
        papers = [e for e in entities if e.entity_type == "paper"]
        assert len(papers) == 1
        assert papers[0].id == "x1"


# ---------------------------------------------------------------------------
# extract_from_abstract — regex-based entity extraction
# ---------------------------------------------------------------------------


class TestExtractFromAbstract:
    def test_extracts_gene_names(self, extractor):
        abstract = "We studied the role of BRCA1 and TP53 in tumor suppression."
        entities = extractor.extract_from_abstract(abstract)
        gene_names = {e.name for e in entities if e.entity_type == "gene"}
        assert "BRCA1" in gene_names
        assert "TP53" in gene_names

    def test_extracts_methods(self, extractor):
        abstract = "We used CRISPR to edit genes and validated with PCR."
        entities = extractor.extract_from_abstract(abstract)
        method_names = {e.name for e in entities if e.entity_type == "method"}
        assert "CRISPR" in method_names
        assert "PCR" in method_names

    def test_extracts_diseases(self, extractor):
        abstract = "This study examines breast cancer and diabetes risk factors."
        entities = extractor.extract_from_abstract(abstract)
        disease_names = {e.name for e in entities if e.entity_type == "disease"}
        assert "breast cancer" in disease_names
        assert "diabetes" in disease_names

    def test_extracts_proteins(self, extractor):
        abstract = "The kinase activity of Casein and Insulin were measured."
        entities = extractor.extract_from_abstract(abstract)
        protein_names = {e.name for e in entities if e.entity_type == "protein"}
        assert "Casein" in protein_names
        assert "Insulin" in protein_names

    def test_gene_excludes_common_words(self, extractor):
        abstract = "THE RESULTS AND DATA ARE NOT WHAT WE EXPECTED FROM THIS STUDY."
        entities = extractor.extract_from_abstract(abstract)
        gene_names = {e.name for e in entities if e.entity_type == "gene"}
        # None of these should be detected as genes
        assert "THE" not in gene_names
        assert "RESULTS" not in gene_names
        assert "DATA" not in gene_names

    def test_empty_abstract(self, extractor):
        assert extractor.extract_from_abstract("") == []
        assert extractor.extract_from_abstract(None) == []

    def test_mentions_have_context(self, extractor):
        abstract = "The mutation of EGFR drives resistance in lung cancer patients."
        entities = extractor.extract_from_abstract(abstract)
        egfr = [e for e in entities if e.name == "EGFR"]
        assert len(egfr) == 1
        assert len(egfr[0].mentions) == 1
        assert "EGFR" in egfr[0].mentions[0]

    def test_no_duplicate_entities(self, extractor):
        abstract = "CRISPR is used widely. We also applied CRISPR in our study."
        entities = extractor.extract_from_abstract(abstract)
        method_names = [e.name for e in entities if e.entity_type == "method"]
        assert method_names.count("CRISPR") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
