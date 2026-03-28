"""ChromaDB-backed vector store for paper embeddings.

Stores papers with SPECTER2 embeddings (when available) for semantic search.
Falls back to ChromaDB's default embedding function when SPECTER2 is absent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

from scidex.literature.s2_client import Paper

logger = logging.getLogger(__name__)


class PaperStore:
    """ChromaDB collection for storing and searching papers by embedding."""

    def __init__(self, persist_dir: str | Path = "data/chromadb"):
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._chroma = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._chroma.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        """Number of papers in the store."""
        return self._collection.count()

    def _paper_metadata(self, paper: Paper) -> dict:
        """Build metadata dict for ChromaDB (must be flat primitives)."""
        return {
            "paper_id": paper.paper_id,
            "title": paper.title or "",
            "year": paper.year or 0,
            "citation_count": paper.citation_count,
            "authors": "; ".join(paper.authors) if paper.authors else "",
            "url": paper.url or "",
            "publication_types": ", ".join(paper.publication_types)
            if paper.publication_types
            else "",
        }

    def add_papers(self, papers: list[Paper]) -> int:
        """Add papers to the store.

        Uses SPECTER2 embeddings if available; otherwise relies on ChromaDB's
        default embedding function over the abstract text.

        Args:
            papers: Papers to add.

        Returns:
            Number of papers actually added (skips those without abstract).
        """
        if not papers:
            return 0

        ids = []
        documents = []
        metadatas = []
        embeddings = []
        has_embeddings = False

        for paper in papers:
            doc_text = paper.abstract or paper.title or ""
            if not doc_text.strip():
                continue

            ids.append(paper.paper_id)
            documents.append(doc_text)
            metadatas.append(self._paper_metadata(paper))
            if paper.embedding:
                embeddings.append(paper.embedding)
                has_embeddings = True
            else:
                embeddings.append(None)

        if not ids:
            return 0

        # If some papers have embeddings and some don't, we can't mix modes.
        # Use embeddings only if ALL papers in this batch have them.
        all_have_embeddings = has_embeddings and all(e is not None for e in embeddings)

        if all_have_embeddings:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        else:
            # Let ChromaDB compute embeddings from document text
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

        logger.info(f"Added {len(ids)} papers to store (embeddings: {all_have_embeddings})")
        return len(ids)

    def search_similar(self, query: str, n_results: int = 10) -> list[Paper]:
        """Semantic search by text query.

        Args:
            query: Natural language query.
            n_results: Number of results to return.

        Returns:
            List of matching Paper objects, ordered by similarity.
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, self.count) if self.count > 0 else n_results,
        )
        return self._results_to_papers(results)

    def search_similar_to_paper(self, paper_id: str, n_results: int = 10) -> list[Paper]:
        """Find papers similar to a given paper already in the store.

        Args:
            paper_id: ID of the paper to use as query.
            n_results: Number of results to return.

        Returns:
            List of similar Paper objects (excludes the query paper).
        """
        # Try to get the paper's embedding from the store
        try:
            stored = self._collection.get(
                ids=[paper_id],
                include=["embeddings", "documents"],
            )
        except Exception:
            return []

        if not stored["ids"]:
            return []

        embeddings = stored.get("embeddings")
        has_embedding = (
            embeddings is not None
            and len(embeddings) > 0
            and embeddings[0] is not None
            and len(embeddings[0]) > 0
        )

        if has_embedding:
            results = self._collection.query(
                query_embeddings=[stored["embeddings"][0]],
                n_results=min(n_results + 1, self.count) if self.count > 0 else n_results + 1,
            )
        elif stored.get("documents") and stored["documents"][0]:
            results = self._collection.query(
                query_texts=[stored["documents"][0]],
                n_results=min(n_results + 1, self.count) if self.count > 0 else n_results + 1,
            )
        else:
            return []

        papers = self._results_to_papers(results)
        # Exclude the query paper itself
        return [p for p in papers if p.paper_id != paper_id][:n_results]

    def get_paper(self, paper_id: str) -> Paper | None:
        """Retrieve a single paper from the store by ID."""
        try:
            stored = self._collection.get(ids=[paper_id], include=["metadatas", "documents"])
        except Exception:
            return None

        if not stored["ids"]:
            return None

        meta = stored["metadatas"][0]
        return Paper(
            paper_id=meta.get("paper_id", paper_id),
            title=meta.get("title", ""),
            abstract=stored["documents"][0] if stored["documents"] else None,
            year=meta.get("year") or None,
            citation_count=meta.get("citation_count", 0),
            authors=meta.get("authors", "").split("; ") if meta.get("authors") else [],
            url=meta.get("url") or None,
            publication_types=(
                meta.get("publication_types", "").split(", ")
                if meta.get("publication_types")
                else None
            ),
        )

    def _results_to_papers(self, results: dict) -> list[Paper]:
        """Convert ChromaDB query results to Paper objects."""
        papers = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return papers

        for i, pid in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            doc = results["documents"][0][i] if results.get("documents") else None

            papers.append(
                Paper(
                    paper_id=meta.get("paper_id", pid),
                    title=meta.get("title", ""),
                    abstract=doc,
                    year=meta.get("year") or None,
                    citation_count=meta.get("citation_count", 0),
                    authors=meta.get("authors", "").split("; ") if meta.get("authors") else [],
                    url=meta.get("url") or None,
                    publication_types=(
                        meta.get("publication_types", "").split(", ")
                        if meta.get("publication_types")
                        else None
                    ),
                )
            )
        return papers
