"""Tests for persisted SciDEX workspaces."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from scidex.workspace.models import WorkspaceSnapshot
from scidex.workspace.store import WorkspaceStore


def _make_snapshot(name: str, topic: str, statement: str) -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
        name=name,
        topic=topic,
        hypothesis_report={
            "topic": topic,
            "summary": "summary",
            "hypotheses": [
                {
                    "id": f"{name}-hyp-1",
                    "statement": statement,
                    "hypothesis_type": "gap_filling",
                    "supporting_evidence": ["Paper support"],
                    "confidence": 0.7,
                    "novelty_score": 0.8,
                    "testability_score": 0.9,
                    "source_papers": ["Paper A"],
                }
            ],
        },
        graph_json={
            "nodes": [{"id": "n1", "name": "TP53", "entity_type": "gene", "properties": {}}],
            "edges": [],
        },
        bookmarks=[
            {
                "id": f"{name}-hyp-1",
                "statement": statement,
                "hypothesis_type": "gap_filling",
            }
        ],
        gde_result={
            "total_rounds": 1,
            "initial_hypotheses": 1,
            "final_hypotheses": 1,
            "final_hypotheses_list": [{"id": f"{name}-hyp-1", "statement": statement}],
        },
        current_protocol={
            "title": f"{name} protocol",
            "hypothesis_id": f"{name}-hyp-1",
            "objective": "Test the idea",
        },
    )


class TestWorkspaceStore:
    def test_save_and_load_named_workspace(self, tmp_path):
        store = WorkspaceStore(tmp_path / "workspaces.json")
        snapshot = _make_snapshot("session-a", "Cancer immunology", "Hypothesis A")

        store.save(snapshot)
        loaded = store.load("session-a")

        assert loaded is not None
        assert loaded.name == "session-a"
        assert loaded.topic == "Cancer immunology"
        assert loaded.hypothesis_report["hypotheses"][0]["statement"] == "Hypothesis A"
        assert loaded.graph_json["nodes"][0]["name"] == "TP53"
        assert loaded.bookmarks[0]["id"] == "session-a-hyp-1"
        assert loaded.gde_result["final_hypotheses_list"][0]["statement"] == "Hypothesis A"
        assert loaded.current_protocol["title"] == "session-a protocol"

    def test_lists_workspaces_in_reverse_chronological_order(self, tmp_path):
        store = WorkspaceStore(tmp_path / "workspaces.json")

        store.save(_make_snapshot("older", "Topic 1", "Hypothesis 1"))
        store.save(_make_snapshot("newer", "Topic 2", "Hypothesis 2"))

        workspaces = store.list_workspaces()

        assert [workspace.name for workspace in workspaces] == ["newer", "older"]

    def test_load_returns_none_for_missing_workspace(self, tmp_path):
        store = WorkspaceStore(tmp_path / "workspaces.json")

        assert store.load("does-not-exist") is None

    def test_duplicate_name_replaces_existing_snapshot(self, tmp_path):
        store = WorkspaceStore(tmp_path / "workspaces.json")

        store.save(_make_snapshot("session-a", "Topic 1", "Hypothesis 1"))
        store.save(_make_snapshot("session-a", "Topic 2", "Hypothesis 2"))

        workspaces = store.list_workspaces()
        loaded = store.load("session-a")

        assert len(workspaces) == 1
        assert loaded is not None
        assert loaded.topic == "Topic 2"
        assert loaded.hypothesis_report["hypotheses"][0]["statement"] == "Hypothesis 2"

    def test_save_raises_when_persistence_fails(self, tmp_path, monkeypatch):
        store = WorkspaceStore(tmp_path / "workspaces.json")
        snapshot = _make_snapshot("session-a", "Topic 1", "Hypothesis 1")

        def _raise(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", _raise)

        with pytest.raises(OSError, match="disk full"):
            store.save(snapshot)

        assert store.list_workspaces() == []

    def test_importing_workspace_models_does_not_eagerly_import_session_helpers(self):
        command = [
            sys.executable,
            "-c",
            (
                "import importlib, sys; "
                "importlib.import_module('scidex.workspace.models'); "
                "print('scidex.workspace.session' in sys.modules)"
            ),
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=True)

        assert result.stdout.strip() == "False"
