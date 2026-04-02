"""Evidence enrichment and deterministic ranking helpers for hypotheses."""

from __future__ import annotations

from scidex.hypothesis.models import Hypothesis


def enrich_hypothesis(hypothesis: Hypothesis) -> Hypothesis:
    """Attach a lightweight structured evidence summary to a hypothesis."""
    source_titles = list(dict.fromkeys(hypothesis.source_papers))
    evidence_summary = {
        "support_count": len(hypothesis.supporting_evidence),
        "source_paper_count": len(source_titles),
        "source_titles": source_titles,
        "top_support": hypothesis.supporting_evidence[:3],
    }
    return hypothesis.model_copy(update={"evidence_summary": evidence_summary})


def _composite_score(hypothesis: Hypothesis) -> float:
    evidence_summary = hypothesis.evidence_summary or {}
    evidence_count = evidence_summary.get("support_count", len(hypothesis.supporting_evidence))
    paper_count = evidence_summary.get("source_paper_count", len(hypothesis.source_papers))
    return round(
        (hypothesis.confidence * 0.35)
        + (hypothesis.novelty_score * 0.25)
        + (hypothesis.testability_score * 0.2)
        + (min(evidence_count, 5) / 5 * 0.1)
        + (min(paper_count, 5) / 5 * 0.1),
        6,
    )


def rank_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    """Enrich, score, and sort hypotheses by composite score."""
    ranked: list[Hypothesis] = []
    for hypothesis in hypotheses:
        enriched = enrich_hypothesis(hypothesis)
        ranked.append(enriched.model_copy(update={"composite_score": _composite_score(enriched)}))
    return sorted(
        ranked,
        key=lambda hypothesis: (hypothesis.composite_score, hypothesis.confidence),
        reverse=True,
    )
