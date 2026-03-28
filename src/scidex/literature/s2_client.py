"""Semantic Scholar API client with rate limiting and local file caching.

Uses the shared _RateLimitedSession to enforce 1 req/sec across all S2 endpoints.
Caches individual paper responses as JSON in data/cache/ to avoid redundant fetches.
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Paper:
    """Represents a paper from Semantic Scholar."""

    paper_id: str
    title: str
    abstract: str | None = None
    year: int | None = None
    citation_count: int = 0
    authors: list[str] = field(default_factory=list)
    url: str | None = None
    publication_types: list[str] | None = None
    embedding: list[float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_s2(cls, data: dict) -> Paper:
        """Parse a paper from an S2 API response dict."""
        authors = []
        for a in data.get("authors") or []:
            if isinstance(a, dict):
                authors.append(a.get("name", ""))
            else:
                authors.append(str(a))

        embedding = None
        emb_data = data.get("embedding")
        if emb_data and isinstance(emb_data, dict):
            embedding = emb_data.get("vector")
        elif isinstance(emb_data, list):
            embedding = emb_data

        return cls(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract"),
            year=data.get("year"),
            citation_count=data.get("citationCount", 0) or 0,
            authors=authors,
            url=data.get("url"),
            publication_types=data.get("publicationTypes"),
            embedding=embedding,
        )


# ---------------------------------------------------------------------------
# Rate-limited session (mirrors shared/llm_config.py but standalone)
# ---------------------------------------------------------------------------


class _RateLimitedSession:
    """requests.Session wrapper enforcing 1 req/sec for Semantic Scholar."""

    def __init__(self, api_key: str = ""):
        self._session = requests.Session()
        if api_key:
            self._session.headers["x-api-key"] = api_key
        self._lock = threading.Lock()
        self._last_request = 0.0

    def _throttle(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
            self._last_request = time.monotonic()

    def get(self, url: str, **kwargs) -> requests.Response:
        self._throttle()
        return self._session.get(url, **kwargs)


# ---------------------------------------------------------------------------
# S2 Client
# ---------------------------------------------------------------------------

BASE_URL = "https://api.semanticscholar.org/graph/v1"

DEFAULT_FIELDS = [
    "paperId",
    "title",
    "abstract",
    "year",
    "citationCount",
    "authors",
    "url",
    "publicationTypes",
    "externalIds",
]

EMBEDDING_FIELD = "embedding.specter_v2"


class S2Client:
    """Semantic Scholar API client with rate limiting and file caching."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path = "data/cache",
    ):
        # Resolve API key
        if api_key is None:
            load_dotenv()
            api_key = os.environ.get("S2_API_KEY", "")
        self._session = _RateLimitedSession(api_key)
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- cache helpers ----------------------------------------------------

    def _cache_path(self, paper_id: str) -> Path:
        safe = paper_id.replace("/", "_").replace(":", "_")
        return self._cache_dir / f"{safe}.json"

    def _read_cache(self, paper_id: str) -> dict | None:
        p = self._cache_path(paper_id)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _write_cache(self, paper_id: str, data: dict) -> None:
        try:
            self._cache_path(paper_id).write_text(json.dumps(data))
        except OSError as e:
            logger.warning(f"Failed to write cache for {paper_id}: {e}")

    # ---- API methods ------------------------------------------------------

    def search_papers(
        self,
        query: str,
        limit: int = 20,
        fields: list[str] | None = None,
    ) -> list[Paper]:
        """Search for papers by keyword query.

        Args:
            query: Search string.
            limit: Max results (up to 100).
            fields: S2 fields to retrieve.

        Returns:
            List of Paper objects.
        """
        if fields is None:
            fields = DEFAULT_FIELDS + [EMBEDDING_FIELD]

        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": ",".join(fields),
        }
        resp = self._session.get(f"{BASE_URL}/paper/search", params=params)
        resp.raise_for_status()
        data = resp.json()

        papers = []
        for item in data.get("data", []):
            paper = Paper.from_s2(item)
            if paper.paper_id:
                self._write_cache(paper.paper_id, item)
                papers.append(paper)
        return papers

    def get_paper(
        self,
        paper_id: str,
        fields: list[str] | None = None,
        use_cache: bool = True,
    ) -> Paper | None:
        """Get a single paper by its S2 paper ID.

        Args:
            paper_id: Semantic Scholar paper ID (or DOI, ArXiv ID, etc.).
            fields: S2 fields to retrieve.
            use_cache: Whether to check local cache first.

        Returns:
            Paper object, or None if not found.
        """
        if use_cache:
            cached = self._read_cache(paper_id)
            if cached:
                return Paper.from_s2(cached)

        if fields is None:
            fields = DEFAULT_FIELDS + [EMBEDDING_FIELD]

        resp = self._session.get(
            f"{BASE_URL}/paper/{paper_id}",
            params={"fields": ",".join(fields)},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        self._write_cache(data.get("paperId", paper_id), data)
        return Paper.from_s2(data)

    def get_citations(
        self,
        paper_id: str,
        limit: int = 50,
        fields: list[str] | None = None,
    ) -> list[Paper]:
        """Get papers that cite the given paper.

        Args:
            paper_id: S2 paper ID.
            limit: Max citing papers to retrieve.
            fields: S2 fields for each citing paper.

        Returns:
            List of citing Paper objects.
        """
        if fields is None:
            fields = DEFAULT_FIELDS + [EMBEDDING_FIELD]

        params = {
            "limit": min(limit, 1000),
            "fields": ",".join(fields),
        }
        resp = self._session.get(
            f"{BASE_URL}/paper/{paper_id}/citations",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

        papers = []
        for item in data.get("data", []):
            citing = item.get("citingPaper", {})
            if not citing or not citing.get("paperId"):
                continue
            paper = Paper.from_s2(citing)
            self._write_cache(paper.paper_id, citing)
            papers.append(paper)
        return papers

    def get_references(
        self,
        paper_id: str,
        limit: int = 50,
        fields: list[str] | None = None,
    ) -> list[Paper]:
        """Get papers referenced by the given paper.

        Args:
            paper_id: S2 paper ID.
            limit: Max referenced papers to retrieve.
            fields: S2 fields for each referenced paper.

        Returns:
            List of referenced Paper objects.
        """
        if fields is None:
            fields = DEFAULT_FIELDS + [EMBEDDING_FIELD]

        params = {
            "limit": min(limit, 1000),
            "fields": ",".join(fields),
        }
        resp = self._session.get(
            f"{BASE_URL}/paper/{paper_id}/references",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

        papers = []
        for item in data.get("data", []):
            cited = item.get("citedPaper", {})
            if not cited or not cited.get("paperId"):
                continue
            paper = Paper.from_s2(cited)
            self._write_cache(paper.paper_id, cited)
            papers.append(paper)
        return papers

    def get_embedding(self, paper_id: str) -> list[float] | None:
        """Get SPECTER2 embedding for a paper.

        Checks cache first, then fetches from S2 API if needed.
        """
        cached = self._read_cache(paper_id)
        if cached:
            paper = Paper.from_s2(cached)
            if paper.embedding:
                return paper.embedding

        paper = self.get_paper(paper_id, fields=[EMBEDDING_FIELD], use_cache=False)
        if paper and paper.embedding:
            return paper.embedding
        return None
