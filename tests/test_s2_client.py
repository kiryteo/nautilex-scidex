"""Tests for the Semantic Scholar API client."""

from __future__ import annotations

import json
import time
import threading
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scidex.literature.s2_client import Paper, S2Client, _RateLimitedSession


# ---------------------------------------------------------------------------
# Paper dataclass tests
# ---------------------------------------------------------------------------


class TestPaper:
    def test_from_s2_basic(self):
        data = {
            "paperId": "abc123",
            "title": "Test Paper",
            "abstract": "An abstract.",
            "year": 2024,
            "citationCount": 42,
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "url": "https://example.com",
            "publicationTypes": ["JournalArticle"],
        }
        paper = Paper.from_s2(data)
        assert paper.paper_id == "abc123"
        assert paper.title == "Test Paper"
        assert paper.abstract == "An abstract."
        assert paper.year == 2024
        assert paper.citation_count == 42
        assert paper.authors == ["Alice", "Bob"]
        assert paper.url == "https://example.com"
        assert paper.publication_types == ["JournalArticle"]
        assert paper.embedding is None

    def test_from_s2_with_embedding(self):
        data = {
            "paperId": "xyz",
            "title": "Embedded",
            "embedding": {"model": "specter_v2", "vector": [0.1, 0.2, 0.3]},
        }
        paper = Paper.from_s2(data)
        assert paper.embedding == [0.1, 0.2, 0.3]

    def test_from_s2_missing_fields(self):
        data = {"paperId": "min"}
        paper = Paper.from_s2(data)
        assert paper.paper_id == "min"
        assert paper.title == ""
        assert paper.abstract is None
        assert paper.authors == []

    def test_to_dict_roundtrip(self):
        paper = Paper(
            paper_id="test",
            title="Test",
            abstract="Abstract",
            year=2024,
            citation_count=10,
            authors=["Alice"],
        )
        d = paper.to_dict()
        assert d["paper_id"] == "test"
        assert d["title"] == "Test"


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------


class TestRateLimitedSession:
    def test_throttle_enforces_1_second_gap(self):
        session = _RateLimitedSession(api_key="test")
        # Mock the underlying session.get
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        session._session.get = MagicMock(return_value=mock_resp)

        t0 = time.monotonic()
        session.get("https://example.com/1")
        session.get("https://example.com/2")
        elapsed = time.monotonic() - t0

        # Second call should have waited ~1s
        assert elapsed >= 0.9, f"Expected >= 0.9s, got {elapsed:.2f}s"

    def test_throttle_no_wait_after_delay(self):
        session = _RateLimitedSession(api_key="test")
        mock_resp = MagicMock()
        session._session.get = MagicMock(return_value=mock_resp)

        session.get("https://example.com/1")
        time.sleep(1.1)  # Wait longer than throttle period

        t0 = time.monotonic()
        session.get("https://example.com/2")
        elapsed = time.monotonic() - t0

        # Should not have waited
        assert elapsed < 0.5, f"Expected < 0.5s, got {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# S2Client caching tests
# ---------------------------------------------------------------------------


class TestS2ClientCache:
    def test_write_and_read_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = S2Client(api_key="test", cache_dir=tmp)

            data = {
                "paperId": "cached1",
                "title": "Cached Paper",
                "abstract": "An abstract",
                "year": 2024,
                "citationCount": 5,
                "authors": [{"name": "Alice"}],
            }
            client._write_cache("cached1", data)

            cached = client._read_cache("cached1")
            assert cached is not None
            assert cached["paperId"] == "cached1"
            assert cached["title"] == "Cached Paper"

    def test_cache_miss_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = S2Client(api_key="test", cache_dir=tmp)
            assert client._read_cache("nonexistent") is None

    def test_get_paper_uses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = S2Client(api_key="test", cache_dir=tmp)

            data = {
                "paperId": "cached2",
                "title": "From Cache",
                "abstract": "Cached abstract",
                "year": 2023,
                "citationCount": 10,
                "authors": [{"name": "Bob"}],
            }
            client._write_cache("cached2", data)

            # Should not make an API call — reads from cache
            paper = client.get_paper("cached2", use_cache=True)
            assert paper is not None
            assert paper.title == "From Cache"
            assert paper.year == 2023


# ---------------------------------------------------------------------------
# S2Client API tests (mocked)
# ---------------------------------------------------------------------------


class TestS2ClientAPI:
    def _make_client(self, tmp_dir: str) -> S2Client:
        client = S2Client(api_key="test", cache_dir=tmp_dir)
        return client

    def test_search_papers(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "data": [
                    {
                        "paperId": "p1",
                        "title": "Paper 1",
                        "abstract": "Abstract 1",
                        "year": 2024,
                        "citationCount": 10,
                        "authors": [{"name": "Alice"}],
                        "url": "https://s2.org/p1",
                    },
                    {
                        "paperId": "p2",
                        "title": "Paper 2",
                        "abstract": "Abstract 2",
                        "year": 2023,
                        "citationCount": 20,
                        "authors": [{"name": "Bob"}],
                        "url": "https://s2.org/p2",
                    },
                ]
            }
            client._session.get = MagicMock(return_value=mock_resp)

            papers = client.search_papers("test query", limit=10)
            assert len(papers) == 2
            assert papers[0].paper_id == "p1"
            assert papers[1].paper_id == "p2"

            # Check cache was written
            assert client._read_cache("p1") is not None
            assert client._read_cache("p2") is not None

    def test_get_citations(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp)
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "data": [
                    {"citingPaper": {"paperId": "c1", "title": "Citing 1", "citationCount": 5}},
                    {"citingPaper": {"paperId": "c2", "title": "Citing 2", "citationCount": 3}},
                    {"citingPaper": {}},  # Missing paperId — should be skipped
                ]
            }
            client._session.get = MagicMock(return_value=mock_resp)

            citations = client.get_citations("p1", limit=10)
            assert len(citations) == 2
            assert citations[0].paper_id == "c1"

    def test_get_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp)
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "data": [
                    {"citedPaper": {"paperId": "r1", "title": "Ref 1", "citationCount": 50}},
                ]
            }
            client._session.get = MagicMock(return_value=mock_resp)

            refs = client.get_references("p1", limit=10)
            assert len(refs) == 1
            assert refs[0].paper_id == "r1"

    def test_get_paper_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._make_client(tmp)
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            client._session.get = MagicMock(return_value=mock_resp)

            paper = client.get_paper("nonexistent", use_cache=False)
            assert paper is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
