"""Mine contradictions across papers using LLM-based claim comparison.

Compares conclusions, methods, and findings across papers that address the
same topic, flagging contradictions as hypothesis seeds for resolution.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Callable

from scidex.hypothesis.models import Hypothesis

logger = logging.getLogger(__name__)


def _hypothesis_id(prefix: str, *parts: str) -> str:
    raw = ":".join([prefix, *parts])
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


_SYSTEM_PROMPT = """\
You are a scientific literature analyst. Given a set of paper abstracts on the \
same topic, identify contradictions — conflicting results, opposing conclusions \
from similar methods, or disputed mechanisms.

Return a JSON array of objects with keys:
  "paper_a": title of first paper,
  "paper_b": title of second paper,
  "claim_a": the claim from paper A,
  "claim_b": the conflicting claim from paper B,
  "resolution_direction": a one-sentence hypothesis for resolving the discrepancy
"""


class ContradictionMiner:
    """Use LLM to compare claims across papers and find contradictions.

    Accepts a ``chat_fn`` callable so tests can inject a mock without
    hitting the real LLM endpoint.
    """

    def __init__(self, chat_fn: Callable[..., str] | None = None) -> None:
        """
        Args:
            chat_fn: A callable matching the signature of
                ``scidex.llm.client.chat`` (messages, **kwargs) → str.
                If *None*, the real ``chat`` function is imported at call time.
        """
        self._chat_fn = chat_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_contradictions(
        self,
        papers: list[dict],
        topic: str,
    ) -> list[Hypothesis]:
        """Compare *papers* and return contradiction-based hypotheses.

        Args:
            papers: List of dicts with at least ``title`` and ``abstract``.
            topic: The shared topic string.

        Returns:
            List of ``Hypothesis`` (type ``contradiction_resolution``).
        """
        if len(papers) < 2:
            return []

        chat = self._get_chat_fn()

        # Build the prompt
        paper_block = "\n\n".join(
            f"### {p.get('title', 'Untitled')}\n{p.get('abstract', '')}" for p in papers
        )
        user_msg = (
            f"Topic: {topic}\n\n"
            f"Papers:\n{paper_block}\n\n"
            "Identify contradictions between these papers. "
            "Return ONLY a JSON array."
        )

        response = chat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=2048,
        )

        contradictions = self._parse_response(response)

        hypotheses: list[Hypothesis] = []
        for c in contradictions:
            paper_a = c.get("paper_a", "Paper A")
            paper_b = c.get("paper_b", "Paper B")
            claim_a = c.get("claim_a", "")
            claim_b = c.get("claim_b", "")
            resolution = c.get("resolution_direction", "")

            h = Hypothesis(
                id=_hypothesis_id("contradiction", paper_a, paper_b),
                statement=(
                    f"Resolving the discrepancy between {paper_a} and {paper_b}: {resolution}"
                ),
                hypothesis_type="contradiction_resolution",
                supporting_evidence=[
                    f"{paper_a}: {claim_a}",
                    f"{paper_b}: {claim_b}",
                ],
                confidence=0.6,
                novelty_score=0.7,
                testability_score=0.5,
                source_papers=[paper_a, paper_b],
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
            hypotheses.append(h)

        return hypotheses

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_chat_fn(self) -> Callable[..., str]:
        """Resolve the chat function (lazy import to avoid requiring tokens in tests)."""
        if self._chat_fn is not None:
            return self._chat_fn
        from scidex.llm.client import chat  # pragma: no cover

        return chat  # pragma: no cover

    @staticmethod
    def _parse_response(response: str) -> list[dict]:
        """Extract a JSON array from the LLM response."""
        text = response.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(line for line in lines if not line.strip().startswith("```"))
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM response as JSON: %s", text[:200])
        return []
