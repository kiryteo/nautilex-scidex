"""Helpers for translating between Streamlit session state and saved workspaces."""

from __future__ import annotations

from typing import Any

from scidex.experiment.models import ExperimentProtocol
from scidex.hypothesis.gde import GDEResult
from scidex.hypothesis.models import Hypothesis, HypothesisReport
from scidex.knowledge_graph.graph import KnowledgeGraph
from scidex.workspace.models import WorkspaceSnapshot


def hydrate_hypothesis(payload: dict[str, Any] | Hypothesis | None) -> Hypothesis | None:
    """Hydrate a hypothesis from a saved payload when possible."""
    if payload is None:
        return payload
    if isinstance(payload, Hypothesis):
        return payload
    return Hypothesis(**payload)


def build_workspace_snapshot(name: str, session_state: dict[str, Any]) -> WorkspaceSnapshot:
    """Convert relevant page session state into a persistable workspace snapshot."""
    report = session_state.get("hypothesis_report")
    graph = session_state.get("kg")
    gde_result = session_state.get("gde_result")
    protocol = session_state.get("current_protocol")
    topic = str(session_state.get("hyp_topic", ""))
    if isinstance(report, HypothesisReport):
        topic = report.topic
        report_data = report.model_dump()
    else:
        report_data = dict(report or {})
        topic = report_data.get("topic", "")

    if isinstance(graph, KnowledgeGraph):
        graph_json = graph.to_json()
    else:
        graph_json = dict(graph or {})

    if isinstance(gde_result, GDEResult):
        gde_data = gde_result.model_dump()
    else:
        gde_data = dict(gde_result or {})

    if isinstance(protocol, ExperimentProtocol):
        protocol_data = protocol.model_dump()
    else:
        protocol_data = dict(protocol or {})

    designed_protocols = [dict(item) for item in session_state.get("designed_protocols", [])]

    return WorkspaceSnapshot(
        name=name,
        topic=topic,
        hypothesis_report=report_data,
        graph_json=graph_json,
        bookmarks=list(session_state.get("bookmarked", [])),
        gde_result=gde_data,
        current_protocol=protocol_data,
        designed_protocols=designed_protocols,
        hypothesis_inputs={
            "hyp_topic": str(session_state.get("hyp_topic", "")),
            "hyp_source_entity": str(session_state.get("hyp_source_entity", "")),
            "hyp_source_method": str(session_state.get("hyp_source_method", "")),
        },
    )


def restore_session_from_snapshot(snapshot: WorkspaceSnapshot) -> dict[str, Any]:
    """Restore Streamlit-friendly session state objects from a workspace snapshot."""
    restored: dict[str, Any] = {
        "active_workspace_name": snapshot.name,
        "bookmarked": list(snapshot.bookmarks),
    }

    restored["kg"] = (
        KnowledgeGraph.from_json(snapshot.graph_json) if snapshot.graph_json else KnowledgeGraph()
    )
    restored["hypothesis_report"] = (
        HypothesisReport(**snapshot.hypothesis_report) if snapshot.hypothesis_report else None
    )
    restored["gde_result"] = GDEResult(**snapshot.gde_result) if snapshot.gde_result else None
    restored["current_protocol"] = (
        ExperimentProtocol(**snapshot.current_protocol) if snapshot.current_protocol else None
    )
    restored["designed_protocols"] = [dict(item) for item in snapshot.designed_protocols]
    restored.update(snapshot.hypothesis_inputs)
    return restored


def build_comparison_payload(
    hypotheses: list[Hypothesis],
    selected_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build a UI-friendly comparison payload for selected hypotheses."""
    selected = set(selected_ids or [])
    ranked = sorted(hypotheses, key=lambda hypothesis: hypothesis.composite_score, reverse=True)
    if selected:
        ranked = [hypothesis for hypothesis in ranked if hypothesis.id in selected]

    comparison = []
    for hypothesis in ranked:
        comparison.append(
            {
                "id": hypothesis.id,
                "statement": hypothesis.statement,
                "hypothesis_type": hypothesis.hypothesis_type,
                "confidence": hypothesis.confidence,
                "novelty_score": hypothesis.novelty_score,
                "testability_score": hypothesis.testability_score,
                "composite_score": hypothesis.composite_score,
                "evidence_summary": dict(hypothesis.evidence_summary),
                "evidence_sections": {
                    "supporting_evidence": list(hypothesis.supporting_evidence),
                    "source_papers": list(hypothesis.source_papers),
                },
            }
        )
    return comparison
