"""Generate-Debate-Evolve (GDE) orchestrator.

Runs the full adversarial refinement loop:
    1. GENERATE  — produce initial hypotheses via HypothesisGenerator
    2. DEBATE    — three critics evaluate each hypothesis
    3. SELECT    — tournament selection keeps top N
    4. EVOLVE    — combine survivors + critique feedback → improved hypotheses
    5. Repeat 2-4 for the configured number of rounds
"""

from __future__ import annotations

import logging
from typing import Callable

from pydantic import BaseModel, Field

from scidex.knowledge_graph.graph import KnowledgeGraph
from scidex.hypothesis.models import Hypothesis
from scidex.hypothesis.generator import HypothesisGenerator
from scidex.hypothesis.critic import CriticScore, HypothesisCritic, PERSPECTIVES
from scidex.hypothesis.tournament import TournamentSelector, TournamentResult
from scidex.hypothesis.evolver import HypothesisEvolver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class GDERoundResult(BaseModel):
    """Result of one GDE round."""

    round_number: int
    hypotheses_in: int
    hypotheses_out: int
    survivors: list[dict] = Field(default_factory=list)
    eliminated: list[dict] = Field(default_factory=list)
    critic_scores: dict[str, list[dict]] = Field(default_factory=dict)
    evolved_hypotheses: list[dict] = Field(default_factory=list)


class GDEResult(BaseModel):
    """Complete result of Generate-Debate-Evolve."""

    total_rounds: int
    initial_hypotheses: int
    final_hypotheses: int
    rounds: list[GDERoundResult] = Field(default_factory=list)
    final_hypotheses_list: list[dict] = Field(default_factory=list)
    best_hypothesis: dict | None = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class GenerateDebateEvolve:
    """Orchestrate the full GDE loop.

    Algorithm:
        1. GENERATE: Use HypothesisGenerator to produce initial hypotheses
        2. DEBATE: 3 critics (methodologist, domain_expert, innovator) evaluate each
        3. SELECT: Tournament selection keeps top N
        4. EVOLVE: Combine survivors + critic feedback to produce improved hypotheses
        5. Repeat steps 2-4 for configured number of rounds
    """

    def __init__(
        self,
        max_rounds: int = 3,
        survivors_per_round: int = 2,
        chat_fn: Callable[..., str] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """
        Args:
            max_rounds: Number of debate-select-evolve rounds.
            survivors_per_round: How many hypotheses survive each round.
            chat_fn: LLM chat callable. If *None*, the real client is imported.
            on_progress: Callback for progress updates (e.g. Streamlit status).
        """
        self.max_rounds = max(1, max_rounds)
        self.survivors_per_round = max(1, survivors_per_round)
        self._chat_fn = chat_fn
        self.on_progress = on_progress

        # Build components
        self._critics = [HypothesisCritic(perspective=p, chat_fn=chat_fn) for p in PERSPECTIVES]
        self._selector = TournamentSelector(survival_count=survivors_per_round)
        self._evolver = HypothesisEvolver(chat_fn=chat_fn)
        self._generator = HypothesisGenerator(chat_fn=chat_fn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        knowledge_graph: KnowledgeGraph | None = None,
        topic: str = "",
        initial_hypotheses: list[Hypothesis] | None = None,
    ) -> GDEResult:
        """Run the full GDE loop.

        Args:
            knowledge_graph: KnowledgeGraph to generate hypotheses from.
            topic: Optional focus topic for generation.
            initial_hypotheses: Pre-generated hypotheses (skips generation).

        Returns:
            GDEResult with all round details and final winning hypotheses.
        """
        # --- Step 0: Generate initial hypotheses ---
        if initial_hypotheses is not None:
            current = list(initial_hypotheses)
        elif knowledge_graph is not None:
            self._progress(f"Round 0: Generating hypotheses for '{topic}'...")
            report = self._generator.generate(knowledge_graph, topic)
            current = report.hypotheses
        else:
            return GDEResult(
                total_rounds=0,
                initial_hypotheses=0,
                final_hypotheses=0,
            )

        if not current:
            return GDEResult(
                total_rounds=0,
                initial_hypotheses=0,
                final_hypotheses=0,
            )

        initial_count = len(current)
        all_rounds: list[GDERoundResult] = []

        # --- Rounds 1..N: Debate → Select → Evolve ---
        for round_num in range(1, self.max_rounds + 1):
            self._progress(f"Round {round_num}: Critics evaluating {len(current)} hypotheses...")

            # DEBATE
            all_critic_scores: dict[str, list[CriticScore]] = {}
            for h in current:
                scores: list[CriticScore] = []
                for critic in self._critics:
                    score = critic.critique(
                        hypothesis_text=h.statement,
                        hypothesis_id=h.id,
                        evidence=h.supporting_evidence,
                    )
                    scores.append(score)
                all_critic_scores[h.id] = scores

            # SELECT
            self._progress(f"Round {round_num}: Tournament selection...")
            result: TournamentResult = self._selector.select(current, all_critic_scores)

            # EVOLVE
            self._progress(f"Round {round_num}: Evolution...")
            evolved = self._evolver.evolve(result.survivors, all_critic_scores, round_num)

            # Record round
            round_result = GDERoundResult(
                round_number=round_num,
                hypotheses_in=len(current),
                hypotheses_out=len(result.survivors),
                survivors=[h.model_dump() for h in result.survivors],
                eliminated=[h.model_dump() for h in result.eliminated],
                critic_scores={
                    hid: [s.model_dump() for s in scores]
                    for hid, scores in all_critic_scores.items()
                },
                evolved_hypotheses=[h.model_dump() for h in evolved],
            )
            all_rounds.append(round_result)

            # Next round uses evolved hypotheses
            current = evolved

        # --- Build final result ---
        # Find best hypothesis by aggregating last round's critic scores
        best = self._find_best(current, all_rounds[-1] if all_rounds else None)

        return GDEResult(
            total_rounds=len(all_rounds),
            initial_hypotheses=initial_count,
            final_hypotheses=len(current),
            rounds=all_rounds,
            final_hypotheses_list=[h.model_dump() for h in current],
            best_hypothesis=best,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _progress(self, message: str) -> None:
        """Send progress update if callback is registered."""
        logger.info(message)
        if self.on_progress:
            self.on_progress(message)

    @staticmethod
    def _find_best(
        hypotheses: list[Hypothesis],
        last_round: GDERoundResult | None,
    ) -> dict | None:
        """Pick the best hypothesis from the final generation."""
        if not hypotheses:
            return None

        if last_round and last_round.critic_scores:
            # Use last round's critic scores to pick the best
            best_id = None
            best_score = -1.0
            for hid, score_dicts in last_round.critic_scores.items():
                avg = sum(s.get("overall_score", 0) for s in score_dicts) / max(len(score_dicts), 1)
                if avg > best_score:
                    best_score = avg
                    best_id = hid

            if best_id:
                for h in hypotheses:
                    if h.id == best_id:
                        return h.model_dump()

        # Fallback: return first hypothesis (highest confidence)
        return hypotheses[0].model_dump()
