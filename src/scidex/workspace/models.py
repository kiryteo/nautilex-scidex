"""Pydantic models for saved SciDEX workspaces."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkspaceSnapshot(BaseModel):
    """Saved hypothesis-session state for resuming work later."""

    name: str
    topic: str = ""
    hypothesis_report: dict[str, Any] = Field(default_factory=dict)
    graph_json: dict[str, Any] = Field(default_factory=dict)
    bookmarks: list[dict[str, Any]] = Field(default_factory=list)
    gde_result: dict[str, Any] = Field(default_factory=dict)
    current_protocol: dict[str, Any] = Field(default_factory=dict)
    designed_protocols: list[dict[str, Any]] = Field(default_factory=list)
    hypothesis_inputs: dict[str, str] = Field(default_factory=dict)
    saved_at: str = Field(default_factory=_now_iso)
