"""LLM-powered hypothesis critic with perspective-based evaluation.

Each critic evaluates hypotheses from a distinct perspective (methodologist,
domain expert, or innovator), producing structured scores and feedback.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Callable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CriticScore(BaseModel):
    """Detailed critique of a hypothesis."""

    hypothesis_id: str
    critic_id: str
    novelty: float = Field(ge=0.0, le=1.0, description="How novel is this hypothesis?")
    feasibility: float = Field(ge=0.0, le=1.0, description="Can this be tested with existing tech?")
    testability: float = Field(ge=0.0, le=1.0, description="Is there a clear experimental path?")
    significance: float = Field(ge=0.0, le=1.0, description="Would confirming this matter?")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    suggested_improvements: list[str] = Field(default_factory=list)
    overall_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Perspective system prompts
# ---------------------------------------------------------------------------

PERSPECTIVES: dict[str, str] = {
    "methodologist": (
        "You are a rigorous methodologist. Focus on experimental design, "
        "statistical power, confounds, and whether the hypothesis is actually "
        "testable. Be skeptical of vague claims."
    ),
    "domain_expert": (
        "You are a domain expert. Evaluate whether this hypothesis is consistent "
        "with established knowledge, whether the proposed mechanisms are plausible, "
        "and whether similar ideas have been tried before."
    ),
    "innovator": (
        "You are an innovation scout. Evaluate whether this hypothesis could lead "
        "to a breakthrough, whether it opens new research directions, and whether "
        "it challenges existing paradigms in productive ways."
    ),
}

_CRITIQUE_USER_TEMPLATE = """\
Evaluate the following hypothesis and return a JSON object with these keys:
  "novelty": float 0-1,
  "feasibility": float 0-1,
  "testability": float 0-1,
  "significance": float 0-1,
  "strengths": [list of strings],
  "weaknesses": [list of strings],
  "suggested_improvements": [list of strings],
  "overall_score": float 0-1,
  "reasoning": "one-paragraph explanation"

Hypothesis: {hypothesis_text}

Supporting evidence:
{evidence_block}

{domain_context_block}

Return ONLY the JSON object, no markdown fences.
"""


class HypothesisCritic:
    """LLM-powered critic that evaluates hypotheses from a specific perspective."""

    def __init__(
        self,
        perspective: str,
        critic_id: str | None = None,
        chat_fn: Callable[..., str] | None = None,
    ) -> None:
        """
        Args:
            perspective: One of 'methodologist', 'domain_expert', 'innovator'.
            critic_id: Optional unique ID for this critic instance.
            chat_fn: Chat callable matching ``scidex.llm.client.chat`` signature.
                     If *None*, the real client is imported at call time.
        """
        if perspective not in PERSPECTIVES:
            raise ValueError(
                f"Unknown perspective '{perspective}'. Choose from: {', '.join(PERSPECTIVES)}"
            )
        self.perspective = perspective
        self.critic_id = critic_id or f"{perspective}_{uuid.uuid4().hex[:8]}"
        self._chat_fn = chat_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def critique(
        self,
        hypothesis_text: str,
        hypothesis_id: str = "",
        evidence: list[str] | None = None,
        domain_context: str = "",
    ) -> CriticScore:
        """Evaluate a hypothesis and return a detailed critique.

        Args:
            hypothesis_text: The hypothesis statement to evaluate.
            hypothesis_id: ID of the hypothesis being critiqued.
            evidence: Supporting evidence strings.
            domain_context: Additional context about the research domain.

        Returns:
            A ``CriticScore`` with structured feedback.
        """
        chat = self._get_chat_fn()
        evidence = evidence or []

        evidence_block = "\n".join(f"- {e}" for e in evidence) if evidence else "None provided."
        domain_context_block = f"Domain context: {domain_context}" if domain_context else ""

        user_msg = _CRITIQUE_USER_TEMPLATE.format(
            hypothesis_text=hypothesis_text,
            evidence_block=evidence_block,
            domain_context_block=domain_context_block,
        )

        response = chat(
            messages=[
                {"role": "system", "content": PERSPECTIVES[self.perspective]},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=2048,
        )

        return self._parse_response(response, hypothesis_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_chat_fn(self) -> Callable[..., str]:
        if self._chat_fn is not None:
            return self._chat_fn
        from scidex.llm.client import chat  # pragma: no cover

        return chat  # pragma: no cover

    def _parse_response(self, response: str, hypothesis_id: str) -> CriticScore:
        """Parse LLM JSON response into a CriticScore."""
        text = response.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(line for line in lines if not line.strip().startswith("```"))

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse critic response as JSON: %s", text[:200])
            data = {}

        def _clamp(val: float | None, default: float = 0.5) -> float:
            if val is None:
                return default
            return max(0.0, min(1.0, float(val)))

        return CriticScore(
            hypothesis_id=hypothesis_id,
            critic_id=self.critic_id,
            novelty=_clamp(data.get("novelty")),
            feasibility=_clamp(data.get("feasibility")),
            testability=_clamp(data.get("testability")),
            significance=_clamp(data.get("significance")),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            suggested_improvements=data.get("suggested_improvements", []),
            overall_score=_clamp(data.get("overall_score")),
            reasoning=data.get("reasoning", ""),
        )
