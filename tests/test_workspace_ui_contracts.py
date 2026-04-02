"""Tests for workspace/session helpers used by the Streamlit pages."""

from __future__ import annotations

from scidex.experiment.models import (
    Control,
    ExperimentProtocol,
    StatisticalPlan,
    Variable,
    VariableType,
)
from scidex.hypothesis.gde import GDEResult
from scidex.hypothesis.models import Hypothesis, HypothesisReport
from scidex.knowledge_graph.graph import KnowledgeGraph
from scidex.knowledge_graph.models import Entity
from scidex.workspace.session import (
    build_comparison_payload,
    build_workspace_snapshot,
    restore_session_from_snapshot,
)


def _make_protocol() -> ExperimentProtocol:
    return ExperimentProtocol(
        title="Workspace Protocol",
        hypothesis_id="hyp-1",
        hypothesis_statement="Hypothesis one",
        objective="Test the hypothesis",
        background="Background",
        variables=[
            Variable(
                name="Dose",
                variable_type=VariableType.INDEPENDENT,
                description="Dose amount",
            )
        ],
        controls=[
            Control(
                name="Vehicle", description="vehicle", control_type="vehicle", rationale="baseline"
            )
        ],
        statistical_plan=StatisticalPlan(
            primary_test="t-test",
            sample_size_per_group=5,
            sample_size_justification="pilot",
        ),
        methodology=["Step 1"],
        expected_outcomes=["Outcome"],
        potential_pitfalls=["Pitfall"],
    )


def _make_hypothesis(hypothesis_id: str, statement: str, composite_score: float) -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        statement=statement,
        hypothesis_type="gap_filling",
        supporting_evidence=[f"Evidence for {statement}"],
        confidence=0.7,
        novelty_score=0.8,
        testability_score=0.9,
        source_papers=[f"Paper for {statement}"],
        evidence_summary={
            "support_count": 1,
            "source_paper_count": 1,
            "source_titles": [f"Paper for {statement}"],
            "top_support": [f"Evidence for {statement}"],
        },
        composite_score=composite_score,
    )


def test_workspace_snapshot_round_trips_page_session_payloads():
    kg = KnowledgeGraph()
    kg.add_entity(Entity(id="gene-1", name="TP53", entity_type="gene"))
    report = HypothesisReport(
        topic="TP53 signaling",
        hypotheses=[_make_hypothesis("hyp-1", "Hypothesis one", 0.91)],
        knowledge_gaps=["Gap 1"],
        summary="One strong idea",
    )
    protocol = _make_protocol()
    gde_result = GDEResult(
        total_rounds=1,
        initial_hypotheses=1,
        final_hypotheses=1,
        final_hypotheses_list=[report.hypotheses[0].model_dump()],
        best_hypothesis=report.hypotheses[0].model_dump(),
    )
    session_state = {
        "kg": kg,
        "hypothesis_report": report,
        "bookmarked": [report.hypotheses[0].model_dump()],
        "gde_result": gde_result,
        "current_protocol": protocol,
        "designed_protocols": [protocol.model_dump()],
        "active_workspace_name": "workspace-a",
        "hyp_topic": "TP53 signaling",
        "hyp_source_entity": "TP53",
        "hyp_source_method": "CRISPR",
    }

    snapshot = build_workspace_snapshot("workspace-a", session_state)
    restored = restore_session_from_snapshot(snapshot)

    assert restored["active_workspace_name"] == "workspace-a"
    assert restored["hypothesis_report"].topic == "TP53 signaling"
    assert restored["kg"].to_json()["nodes"][0]["name"] == "TP53"
    assert restored["bookmarked"][0]["id"] == "hyp-1"
    assert restored["gde_result"].best_hypothesis["id"] == "hyp-1"
    assert restored["current_protocol"].title == "Workspace Protocol"
    assert restored["designed_protocols"][0]["title"] == "Workspace Protocol"
    assert restored["hyp_topic"] == "TP53 signaling"
    assert restored["hyp_source_entity"] == "TP53"
    assert restored["hyp_source_method"] == "CRISPR"


def test_comparison_payload_uses_top_ranked_hypotheses_and_evidence_sections():
    hypotheses = [
        _make_hypothesis("hyp-1", "Lower ranked", 0.61),
        _make_hypothesis("hyp-2", "Highest ranked", 0.95),
        _make_hypothesis("hyp-3", "Middle ranked", 0.72),
    ]

    comparison = build_comparison_payload(hypotheses, selected_ids=["hyp-2", "hyp-3"])

    assert [item["id"] for item in comparison] == ["hyp-2", "hyp-3"]
    assert comparison[0]["composite_score"] == 0.95
    assert comparison[0]["evidence_sections"]["supporting_evidence"] == [
        "Evidence for Highest ranked"
    ]
    assert comparison[0]["evidence_sections"]["source_papers"] == ["Paper for Highest ranked"]
