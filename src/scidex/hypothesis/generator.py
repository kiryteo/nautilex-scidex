"""Orchestrate all hypothesis-generation strategies into a single pipeline.

1. GapDetector  → knowledge gaps
2. SwansonLinker → novel A-B-C connections
3. AnalogyEngine → cross-domain transfer
4. ContradictionMiner → contradictions to resolve
5. (Optional) LLM synthesis → refine & rank
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Callable

from scidex.knowledge_graph.graph import KnowledgeGraph
from scidex.hypothesis.models import Hypothesis, HypothesisReport, GapAnalysis
from scidex.hypothesis.gap_detector import GapDetector
from scidex.hypothesis.swanson_linker import SwansonLinker
from scidex.hypothesis.analogy_engine import AnalogyEngine
from scidex.hypothesis.contradictions import ContradictionMiner

logger = logging.getLogger(__name__)


def _hypothesis_id(prefix: str, *parts: str) -> str:
    raw = ":".join([prefix, *parts])
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


_RANK_SYSTEM_PROMPT = """\
You are a scientific hypothesis evaluator. Given a list of hypotheses with \
their supporting evidence, assign a testability_score (0-1) to each. \
Consider whether the hypothesis can be verified with existing experimental \
techniques and available data.

Return a JSON array of objects: [{"id": "...", "testability_score": 0.X}, ...]
"""


class HypothesisGenerator:
    """End-to-end hypothesis generation pipeline.

    Accepts an optional ``chat_fn`` for LLM-based refinement (synthesis step).
    When *None* is supplied, the pipeline still works but skips LLM ranking.
    """

    def __init__(self, chat_fn: Callable[..., str] | None = None) -> None:
        self._gap_detector = GapDetector()
        self._swanson = SwansonLinker()
        self._analogy = AnalogyEngine()
        self._contradiction = ContradictionMiner(chat_fn=chat_fn)
        self._chat_fn = chat_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        kg: KnowledgeGraph,
        topic: str,
        source_entity: str | None = None,
        source_method: str | None = None,
        papers: list[dict] | None = None,
        refine_with_llm: bool = False,
    ) -> HypothesisReport:
        """Run the full hypothesis-generation pipeline.

        Args:
            kg: Knowledge graph built from ingested papers.
            topic: Research topic / query.
            source_entity: Starting entity for Swanson linking (defaults to topic).
            source_method: Method name for analogy engine (optional).
            papers: Papers for contradiction mining (optional).
            refine_with_llm: If *True* and a chat_fn is available, use
                the LLM to re-score testability.

        Returns:
            A ``HypothesisReport`` aggregating results from all strategies.
        """
        all_hypotheses: list[Hypothesis] = []

        # 1 — Gap analysis
        gap_analysis = self._gap_detector.detect_gaps(kg, topic)

        # Generate gap-filling hypotheses from suggested directions
        for direction in gap_analysis.suggested_directions:
            h = Hypothesis(
                id=_hypothesis_id("gap", direction),
                statement=direction,
                hypothesis_type="gap_filling",
                supporting_evidence=[
                    f"Understudied: {', '.join(gap_analysis.understudied[:3])}",
                ],
                confidence=0.4,
                novelty_score=0.8,
                testability_score=0.5,
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
            all_hypotheses.append(h)

        # 2 — Swanson ABC linking
        entity = source_entity or topic
        swanson_results = self._swanson.discover_links(kg, entity)
        all_hypotheses.extend(swanson_results)

        # 3 — Analogy engine
        if source_method:
            analogy_results = self._analogy.find_analogies(kg, source_method)
            all_hypotheses.extend(analogy_results)

        # 4 — Contradiction mining
        if papers and len(papers) >= 2:
            contradiction_results = self._contradiction.find_contradictions(papers, topic)
            all_hypotheses.extend(contradiction_results)

        # 5 — LLM refinement (optional)
        if refine_with_llm and self._chat_fn and all_hypotheses:
            all_hypotheses = self._refine_scores(all_hypotheses)

        # Sort by confidence descending
        all_hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        # Build knowledge-gap list from the analysis
        knowledge_gaps = gap_analysis.suggested_directions + [
            f"Understudied: {name}" for name in gap_analysis.understudied[:5]
        ]

        summary = (
            f"Generated {len(all_hypotheses)} hypotheses for '{topic}'. "
            f"Found {len(gap_analysis.understudied)} understudied entities "
            f"and {len(gap_analysis.connections_missing)} missing connections."
        )

        return HypothesisReport(
            topic=topic,
            hypotheses=all_hypotheses,
            knowledge_gaps=knowledge_gaps,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _refine_scores(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Use LLM to update testability scores."""
        if not self._chat_fn:
            return hypotheses

        summaries = [
            {"id": h.id, "statement": h.statement, "evidence": h.supporting_evidence[:2]}
            for h in hypotheses[:20]  # cap to avoid token overflow
        ]

        try:
            response = self._chat_fn(
                messages=[
                    {"role": "system", "content": _RANK_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(summaries)},
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            scores = self._parse_scores(response)
            score_map = {s["id"]: s["testability_score"] for s in scores}
            for h in hypotheses:
                if h.id in score_map:
                    h.testability_score = score_map[h.id]
        except Exception:
            logger.warning("LLM refinement failed; keeping original scores.")

        return hypotheses

    @staticmethod
    def _parse_scores(response: str) -> list[dict]:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(line for line in lines if not line.strip().startswith("```"))
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        return []
