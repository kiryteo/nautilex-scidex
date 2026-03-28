"""Literature agent — Semantic Scholar client, paper store, and ingestion pipeline."""

from scidex.literature.s2_client import Paper, S2Client
from scidex.literature.paper_store import PaperStore
from scidex.literature.ingestion import ingest_from_query, ingest_citations

__all__ = ["Paper", "S2Client", "PaperStore", "ingest_from_query", "ingest_citations"]
