"""Paper ingestion pipeline — search S2, fetch details, and store in ChromaDB.

Provides generator-based progress tracking so callers (e.g. Streamlit) can
display incremental status updates.
"""

from __future__ import annotations

import logging
from typing import Generator

from scidex.literature.s2_client import Paper, S2Client
from scidex.literature.paper_store import PaperStore

logger = logging.getLogger(__name__)


def ingest_from_query(
    query: str,
    limit: int = 20,
    client: S2Client | None = None,
    store: PaperStore | None = None,
) -> Generator[dict, None, list[Paper]]:
    """Search S2 for papers and ingest them into the local store.

    Yields status dicts for progress tracking:
        {"stage": "searching", "message": "..."}
        {"stage": "storing", "message": "...", "count": N}
        {"stage": "done", "message": "...", "count": N}

    Returns the final list of ingested papers.
    """
    if client is None:
        client = S2Client()
    if store is None:
        store = PaperStore()

    yield {"stage": "searching", "message": f"Searching for '{query}' (limit={limit})..."}

    papers = client.search_papers(query, limit=limit)
    yield {
        "stage": "fetched",
        "message": f"Found {len(papers)} papers from Semantic Scholar",
        "count": len(papers),
    }

    if not papers:
        yield {"stage": "done", "message": "No papers found.", "count": 0}
        return []

    yield {"stage": "storing", "message": f"Storing {len(papers)} papers in vector store..."}
    added = store.add_papers(papers)

    yield {
        "stage": "done",
        "message": f"Ingested {added} papers (total in store: {store.count})",
        "count": added,
    }
    return papers


def ingest_citations(
    paper_id: str,
    limit: int = 50,
    client: S2Client | None = None,
    store: PaperStore | None = None,
) -> Generator[dict, None, list[Paper]]:
    """Fetch citations for a paper and ingest them into the store.

    Yields status dicts for progress tracking.
    Returns the list of citing papers.
    """
    if client is None:
        client = S2Client()
    if store is None:
        store = PaperStore()

    yield {"stage": "fetching", "message": f"Fetching citations for paper {paper_id}..."}

    citations = client.get_citations(paper_id, limit=limit)
    yield {
        "stage": "fetched",
        "message": f"Found {len(citations)} citing papers",
        "count": len(citations),
    }

    if not citations:
        yield {"stage": "done", "message": "No citations found.", "count": 0}
        return []

    yield {"stage": "storing", "message": f"Storing {len(citations)} citing papers..."}
    added = store.add_papers(citations)

    yield {
        "stage": "done",
        "message": f"Ingested {added} citing papers (total in store: {store.count})",
        "count": added,
    }
    return citations


def ingest_references(
    paper_id: str,
    limit: int = 50,
    client: S2Client | None = None,
    store: PaperStore | None = None,
) -> Generator[dict, None, list[Paper]]:
    """Fetch references for a paper and ingest them into the store.

    Yields status dicts for progress tracking.
    Returns the list of referenced papers.
    """
    if client is None:
        client = S2Client()
    if store is None:
        store = PaperStore()

    yield {"stage": "fetching", "message": f"Fetching references for paper {paper_id}..."}

    references = client.get_references(paper_id, limit=limit)
    yield {
        "stage": "fetched",
        "message": f"Found {len(references)} referenced papers",
        "count": len(references),
    }

    if not references:
        yield {"stage": "done", "message": "No references found.", "count": 0}
        return []

    yield {"stage": "storing", "message": f"Storing {len(references)} referenced papers..."}
    added = store.add_papers(references)

    yield {
        "stage": "done",
        "message": f"Ingested {added} referenced papers (total in store: {store.count})",
        "count": added,
    }
    return references
