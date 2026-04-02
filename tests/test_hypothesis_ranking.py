"""Tests for hypothesis evidence enrichment and ranking."""

from __future__ import annotations

from scidex.hypothesis.models import Hypothesis
from scidex.hypothesis.ranking import enrich_hypothesis, rank_hypotheses


def _make_hypothesis(
    hypothesis_id: str,
    *,
    confidence: float,
    novelty: float,
    testability: float,
    evidence_count: int,
    paper_count: int,
) -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        statement=f"Hypothesis {hypothesis_id}",
        hypothesis_type="gap_filling",
        supporting_evidence=[f"Evidence {i}" for i in range(evidence_count)],
        confidence=confidence,
        novelty_score=novelty,
        testability_score=testability,
        source_papers=[f"Paper {i}" for i in range(paper_count)],
    )


def test_enrich_hypothesis_adds_summary_without_mutating_core_fields():
    hypothesis = _make_hypothesis(
        "h1",
        confidence=0.7,
        novelty=0.8,
        testability=0.6,
        evidence_count=2,
        paper_count=1,
    )

    enriched = enrich_hypothesis(hypothesis)

    assert enriched.statement == hypothesis.statement
    assert enriched.supporting_evidence == hypothesis.supporting_evidence
    assert enriched.evidence_summary["support_count"] == 2
    assert enriched.evidence_summary["source_paper_count"] == 1


def test_rank_hypotheses_sorts_by_deterministic_composite_score():
    lower = _make_hypothesis(
        "lower",
        confidence=0.6,
        novelty=0.4,
        testability=0.5,
        evidence_count=1,
        paper_count=1,
    )
    higher = _make_hypothesis(
        "higher",
        confidence=0.8,
        novelty=0.9,
        testability=0.7,
        evidence_count=3,
        paper_count=2,
    )

    ranked = rank_hypotheses([lower, higher])

    assert [hypothesis.id for hypothesis in ranked] == ["higher", "lower"]
    assert ranked[0].composite_score > ranked[1].composite_score
