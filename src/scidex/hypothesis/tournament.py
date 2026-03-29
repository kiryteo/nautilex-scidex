"""Tournament-style selection for hypothesis ranking and survival.

Aggregates critic scores across multiple perspectives and selects the
top-N hypotheses to survive into the next round.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from scidex.hypothesis.models import Hypothesis
from scidex.hypothesis.critic import CriticScore

logger = logging.getLogger(__name__)

# Default weights for aggregating across score dimensions.
_DIMENSION_WEIGHTS: dict[str, float] = {
    "novelty": 0.25,
    "feasibility": 0.20,
    "testability": 0.20,
    "significance": 0.25,
    "overall_score": 0.10,
}


@dataclass
class TournamentResult:
    """Result of tournament selection."""

    survivors: list[Hypothesis] = field(default_factory=list)
    eliminated: list[Hypothesis] = field(default_factory=list)
    rankings: list[tuple[str, float]] = field(default_factory=list)


class TournamentSelector:
    """Select top hypotheses through tournament-style evaluation."""

    def __init__(
        self,
        survival_count: int = 2,
        dimension_weights: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            survival_count: Number of hypotheses to keep each round.
            dimension_weights: Optional custom weights for score dimensions.
        """
        self.survival_count = survival_count
        self._weights = dimension_weights or _DIMENSION_WEIGHTS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        hypotheses: list[Hypothesis],
        critic_scores: dict[str, list[CriticScore]],
    ) -> TournamentResult:
        """Select top hypotheses based on aggregated critic scores.

        Args:
            hypotheses: List of Hypothesis objects.
            critic_scores: Map of hypothesis_id -> list of CriticScore from
                different critics.

        Returns:
            TournamentResult with survivors, eliminated, and rankings.
        """
        if not hypotheses:
            return TournamentResult()

        # Compute aggregate score per hypothesis
        scored: list[tuple[Hypothesis, float]] = []
        for h in hypotheses:
            scores = critic_scores.get(h.id, [])
            agg = self._aggregate_scores(scores)
            scored.append((h, agg))

        # Sort by aggregate score descending
        scored.sort(key=lambda pair: pair[1], reverse=True)

        rankings = [(h.id, score) for h, score in scored]

        n = min(self.survival_count, len(scored))
        survivors = [h for h, _ in scored[:n]]
        eliminated = [h for h, _ in scored[n:]]

        return TournamentResult(
            survivors=survivors,
            eliminated=eliminated,
            rankings=rankings,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _aggregate_scores(self, scores: list[CriticScore]) -> float:
        """Compute a weighted aggregate score across all critics and dimensions.

        First averages each dimension across critics, then applies dimension
        weights to produce a single scalar.
        """
        if not scores:
            return 0.0

        # Average each dimension across critics
        dim_avgs: dict[str, float] = {}
        for dim in self._weights:
            values = [getattr(s, dim, 0.0) for s in scores]
            dim_avgs[dim] = sum(values) / len(values)

        # Weighted combination
        total_weight = sum(self._weights.values())
        if total_weight == 0:
            return 0.0

        aggregate = sum(dim_avgs.get(dim, 0.0) * w for dim, w in self._weights.items())
        return aggregate / total_weight
