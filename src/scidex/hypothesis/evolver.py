"""Evolve hypotheses by combining survivors and incorporating critic feedback.

Uses LLM to synthesize improved hypotheses from the surviving hypotheses
and the critique feedback (weaknesses to address, improvements to apply).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Callable

from scidex.hypothesis.models import Hypothesis
from scidex.hypothesis.critic import CriticScore

logger = logging.getLogger(__name__)


def _hypothesis_id(prefix: str, *parts: str) -> str:
    raw = ":".join([prefix, *parts])
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


_EVOLVE_SYSTEM_PROMPT = """\
You are a scientific hypothesis refinement engine. Given surviving hypotheses \
and their critic feedback, produce improved hypotheses that:
1. Combine the strongest elements of the survivors ("crossover")
2. Address weaknesses identified by critics ("mutation")
3. Incorporate suggested improvements

Return a JSON array of objects with keys:
  "statement": the improved hypothesis text,
  "supporting_evidence": [list of evidence strings],
  "parent_ids": [list of parent hypothesis IDs that were combined],
  "improvements_applied": [list of improvements incorporated]
"""


class HypothesisEvolver:
    """Evolve hypotheses by combining and refining survivors using critic feedback."""

    def __init__(self, chat_fn: Callable[..., str] | None = None) -> None:
        """
        Args:
            chat_fn: Chat callable matching ``scidex.llm.client.chat`` signature.
                     If *None*, the real client is imported at call time.
        """
        self._chat_fn = chat_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evolve(
        self,
        survivors: list[Hypothesis],
        critic_scores: dict[str, list[CriticScore]],
        round_number: int,
    ) -> list[Hypothesis]:
        """Produce next generation of hypotheses.

        Combines elements of survivors and addresses weaknesses identified by
        critics. Each evolved hypothesis references its parent hypotheses.

        Args:
            survivors: Hypotheses that survived tournament selection.
            critic_scores: Map of hypothesis_id -> list of CriticScore.
            round_number: Current GDE round (used in ID generation).

        Returns:
            New Hypothesis objects with updated text and evidence.
        """
        if not survivors:
            return []

        chat = self._get_chat_fn()

        # Build context for the LLM
        survivor_block = self._format_survivors(survivors, critic_scores)

        user_msg = (
            f"Round {round_number} evolution.\n\n"
            f"Surviving hypotheses and their critiques:\n{survivor_block}\n\n"
            f"Produce {len(survivors)} improved hypotheses that combine and "
            f"refine the survivors. Return ONLY a JSON array."
        )

        response = chat(
            messages=[
                {"role": "system", "content": _EVOLVE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.5,
            max_tokens=4096,
        )

        return self._parse_evolved(response, survivors, round_number)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_chat_fn(self) -> Callable[..., str]:
        if self._chat_fn is not None:
            return self._chat_fn
        from scidex.llm.client import chat  # pragma: no cover

        return chat  # pragma: no cover

    @staticmethod
    def _format_survivors(
        survivors: list[Hypothesis],
        critic_scores: dict[str, list[CriticScore]],
    ) -> str:
        """Format survivors and their critique feedback for the LLM prompt."""
        parts: list[str] = []
        for h in survivors:
            scores = critic_scores.get(h.id, [])
            weaknesses = []
            improvements = []
            for s in scores:
                weaknesses.extend(s.weaknesses)
                improvements.extend(s.suggested_improvements)

            # De-duplicate
            weaknesses = list(dict.fromkeys(weaknesses))
            improvements = list(dict.fromkeys(improvements))

            block = (
                f"ID: {h.id}\n"
                f"Statement: {h.statement}\n"
                f"Evidence: {'; '.join(h.supporting_evidence[:3])}\n"
                f"Weaknesses: {'; '.join(weaknesses[:5]) or 'None'}\n"
                f"Suggested improvements: {'; '.join(improvements[:5]) or 'None'}\n"
            )
            parts.append(block)

        return "\n---\n".join(parts)

    def _parse_evolved(
        self,
        response: str,
        survivors: list[Hypothesis],
        round_number: int,
    ) -> list[Hypothesis]:
        """Parse LLM response into new Hypothesis objects."""
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(line for line in lines if not line.strip().startswith("```"))

        try:
            items = json.loads(text)
            if not isinstance(items, list):
                items = []
        except json.JSONDecodeError:
            logger.warning("Could not parse evolution response as JSON: %s", text[:200])
            items = []

        # Fallback: if LLM returned nothing, return survivors unchanged
        if not items:
            return list(survivors)

        hypotheses: list[Hypothesis] = []
        parent_ids = [h.id for h in survivors]
        now = datetime.now(timezone.utc).isoformat()

        for i, item in enumerate(items):
            statement = item.get("statement", "")
            if not statement:
                continue

            evidence = item.get("supporting_evidence", [])
            if not isinstance(evidence, list):
                evidence = []

            hid = _hypothesis_id("evolved", str(round_number), str(i), statement[:50])

            h = Hypothesis(
                id=hid,
                statement=statement,
                hypothesis_type="extension",
                supporting_evidence=evidence,
                confidence=0.6 + 0.05 * round_number,  # slight boost per round
                novelty_score=0.7,
                testability_score=0.6,
                source_papers=[],
                generated_at=now,
            )
            hypotheses.append(h)

        return hypotheses if hypotheses else list(survivors)
