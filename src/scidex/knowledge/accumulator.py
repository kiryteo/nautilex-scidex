"""Persist and retrieve accumulated knowledge across sessions.

Stores all knowledge entries in a single JSON file on disk. Provides
methods to add papers, entities, hypotheses, experiments, and meta-insights,
as well as keyword search and session tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from scidex.knowledge.models import KnowledgeEntry, KnowledgeLayer, KnowledgeSnapshot

logger = logging.getLogger(__name__)


def _make_id(prefix: str, *parts: str) -> str:
    """Deterministic ID from prefix + parts."""
    raw = ":".join([prefix, *parts])
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class KnowledgeAccumulator:
    """Persist and retrieve accumulated knowledge across sessions."""

    def __init__(self, storage_path: str | Path = ".scidex_knowledge.json") -> None:
        self.storage_path = Path(storage_path)
        self._entries: list[KnowledgeEntry] = []
        self._sessions: list[dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Add knowledge
    # ------------------------------------------------------------------

    def add_papers(self, papers: list[dict]) -> int:
        """Record ingested papers as raw knowledge.

        Args:
            papers: List of paper dicts with keys like paper_id/paperId, title, abstract.

        Returns:
            Number of papers added (deduped by ID).
        """
        added = 0
        existing_ids = {e.id for e in self._entries}
        for p in papers:
            paper_id = p.get("paper_id") or p.get("paperId") or ""
            title = p.get("title", "")
            if not paper_id or not title:
                continue
            entry_id = _make_id("paper", paper_id)
            if entry_id in existing_ids:
                continue
            entry = KnowledgeEntry(
                id=entry_id,
                layer=KnowledgeLayer.RAW,
                content=title,
                source=paper_id,
                tags=["paper"],
                created_at=_now_iso(),
                metadata={
                    "abstract": p.get("abstract", ""),
                    "year": p.get("year"),
                    "citation_count": p.get("citation_count", p.get("citationCount", 0)),
                },
            )
            self._entries.append(entry)
            existing_ids.add(entry_id)
            added += 1
        self._save()
        return added

    def add_entities(self, entities: list[dict]) -> int:
        """Record extracted entities as domain knowledge.

        Args:
            entities: List of entity dicts with keys like id, name, entity_type.

        Returns:
            Number of entities added (deduped by ID).
        """
        added = 0
        existing_ids = {e.id for e in self._entries}
        for ent in entities:
            ent_id = ent.get("id", "")
            name = ent.get("name", "")
            if not name:
                continue
            entry_id = _make_id("entity", ent_id or name)
            if entry_id in existing_ids:
                continue
            entry = KnowledgeEntry(
                id=entry_id,
                layer=KnowledgeLayer.DOMAIN,
                content=name,
                source=ent_id,
                tags=["entity", ent.get("entity_type", "unknown")],
                created_at=_now_iso(),
                metadata=ent.get("properties", {}),
            )
            self._entries.append(entry)
            existing_ids.add(entry_id)
            added += 1
        self._save()
        return added

    def add_hypothesis(self, hypothesis: dict) -> None:
        """Record a hypothesis as an outcome.

        Args:
            hypothesis: Dict with keys like id, statement, hypothesis_type.
        """
        hyp_id = hypothesis.get("id", "")
        statement = hypothesis.get("statement", "")
        if not statement:
            return
        entry_id = _make_id("hypothesis", hyp_id or statement)
        if any(e.id == entry_id for e in self._entries):
            return
        entry = KnowledgeEntry(
            id=entry_id,
            layer=KnowledgeLayer.OUTCOMES,
            content=statement,
            source=hyp_id,
            tags=["hypothesis", hypothesis.get("hypothesis_type", "")],
            created_at=_now_iso(),
            metadata={
                "confidence": hypothesis.get("confidence", 0),
                "novelty_score": hypothesis.get("novelty_score", 0),
                "testability_score": hypothesis.get("testability_score", 0),
            },
        )
        self._entries.append(entry)
        self._save()

    def add_experiment(self, protocol: dict) -> None:
        """Record a designed experiment as an outcome.

        Args:
            protocol: Dict with keys like title, hypothesis_id, objective.
        """
        title = protocol.get("title", "")
        hyp_id = protocol.get("hypothesis_id", "")
        if not title:
            return
        entry_id = _make_id("experiment", hyp_id, title)
        if any(e.id == entry_id for e in self._entries):
            return
        entry = KnowledgeEntry(
            id=entry_id,
            layer=KnowledgeLayer.OUTCOMES,
            content=title,
            source=hyp_id,
            tags=["experiment"],
            created_at=_now_iso(),
            metadata={
                "objective": protocol.get("objective", ""),
                "timeline_weeks": protocol.get("timeline_weeks", 0),
            },
        )
        self._entries.append(entry)
        self._save()

    def add_insight(self, insight: str, source: str = "session") -> None:
        """Record a meta-insight or pattern.

        Args:
            insight: The insight text.
            source: Where the insight came from.
        """
        if not insight:
            return
        entry_id = _make_id("insight", insight)
        if any(e.id == entry_id for e in self._entries):
            return
        entry = KnowledgeEntry(
            id=entry_id,
            layer=KnowledgeLayer.META,
            content=insight,
            source=source,
            tags=["insight"],
            created_at=_now_iso(),
        )
        self._entries.append(entry)
        self._save()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: str, layer: KnowledgeLayer | None = None) -> list[KnowledgeEntry]:
        """Search accumulated knowledge by keyword.

        Args:
            query: Keyword(s) to search for (case-insensitive).
            layer: Optional layer filter.

        Returns:
            List of matching KnowledgeEntry objects.
        """
        query_lower = query.lower()
        results: list[KnowledgeEntry] = []
        for entry in self._entries:
            if layer is not None and entry.layer != layer:
                continue
            # Search in content, source, and tags
            if (
                query_lower in entry.content.lower()
                or query_lower in entry.source.lower()
                or any(query_lower in tag.lower() for tag in entry.tags)
            ):
                results.append(entry)
        return results

    def get_snapshot(self) -> KnowledgeSnapshot:
        """Get current knowledge state summary."""
        total_papers = sum(1 for e in self._entries if "paper" in e.tags)
        total_entities = sum(1 for e in self._entries if "entity" in e.tags)
        total_hypotheses = sum(1 for e in self._entries if "hypothesis" in e.tags)
        total_experiments = sum(1 for e in self._entries if "experiment" in e.tags)

        return KnowledgeSnapshot(
            entries=list(self._entries),
            total_papers=total_papers,
            total_entities=total_entities,
            total_hypotheses=total_hypotheses,
            total_experiments=total_experiments,
            sessions=list(self._sessions),
        )

    # ------------------------------------------------------------------
    # Session tracking
    # ------------------------------------------------------------------

    def get_session_history(self) -> list[dict]:
        """Get history of research sessions."""
        return list(self._sessions)

    def start_session(self, topic: str) -> None:
        """Record the start of a research session.

        Args:
            topic: The research topic for this session.
        """
        session = {
            "topic": topic,
            "started_at": _now_iso(),
            "ended_at": None,
            "summary": "",
        }
        self._sessions.append(session)
        self._save()

    def end_session(self, summary: str) -> None:
        """Record the end of a research session.

        Args:
            summary: Summary of what was accomplished.
        """
        if self._sessions:
            self._sessions[-1]["ended_at"] = _now_iso()
            self._sessions[-1]["summary"] = summary
            self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Save all entries and sessions to disk."""
        data = {
            "entries": [e.model_dump() for e in self._entries],
            "sessions": self._sessions,
        }
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(json.dumps(data, indent=2))
        except OSError as exc:
            logger.warning("Failed to save knowledge store: %s", exc)

    def _load(self) -> None:
        """Load entries and sessions from disk."""
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text())
            self._entries = [KnowledgeEntry(**e) for e in data.get("entries", [])]
            self._sessions = data.get("sessions", [])
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning("Failed to load knowledge store: %s", exc)
            self._entries = []
            self._sessions = []
