"""JSON-backed persistence for named SciDEX workspaces."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from scidex.workspace.models import WorkspaceSnapshot

logger = logging.getLogger(__name__)


class WorkspaceStore:
    """Persist and retrieve named workspace snapshots."""

    def __init__(self, storage_path: str | Path = "data/workspaces.json") -> None:
        self.storage_path = Path(storage_path)
        self._workspaces: list[WorkspaceSnapshot] = []
        self._load()

    def save(self, snapshot: WorkspaceSnapshot) -> WorkspaceSnapshot:
        """Save a snapshot, replacing any existing workspace with the same name."""
        snapshot = snapshot.model_copy(
            update={"saved_at": snapshot.saved_at or WorkspaceSnapshot(name=snapshot.name).saved_at}
        )
        updated_workspaces = [
            workspace for workspace in self._workspaces if workspace.name != snapshot.name
        ]
        updated_workspaces.insert(0, snapshot)
        self._save(updated_workspaces)
        self._workspaces = updated_workspaces
        return snapshot

    def list_workspaces(self) -> list[WorkspaceSnapshot]:
        """List workspaces with most recently saved first."""
        return sorted(self._workspaces, key=lambda workspace: workspace.saved_at, reverse=True)

    def load(self, name: str) -> WorkspaceSnapshot | None:
        """Load a workspace by name."""
        return next((workspace for workspace in self._workspaces if workspace.name == name), None)

    def _save(self, workspaces: list[WorkspaceSnapshot]) -> None:
        data = {"workspaces": [workspace.model_dump() for workspace in workspaces]}
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to save workspace store: %s", exc)
            raise

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self._workspaces = [WorkspaceSnapshot(**item) for item in data.get("workspaces", [])]
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to load workspace store: %s", exc)
            self._workspaces = []
