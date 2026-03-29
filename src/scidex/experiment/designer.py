"""Generate structured experiment protocols from hypotheses using LLM."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from scidex.experiment.models import (
    Control,
    ExperimentProtocol,
    StatisticalPlan,
    Variable,
    VariableType,
)
from scidex.hypothesis.models import Hypothesis

logger = logging.getLogger(__name__)


_DESIGN_SYSTEM_PROMPT = """\
You are an expert experiment designer. Given a scientific hypothesis, design a \
rigorous experiment to test it. Return a JSON object with exactly these keys:

{
  "title": "short experiment title",
  "objective": "what this experiment aims to determine",
  "background": "brief scientific background and rationale",
  "variables": [
    {
      "name": "variable name",
      "variable_type": "independent|dependent|controlled|confounding",
      "description": "what this variable represents",
      "measurement_method": "how to measure it",
      "units": "measurement units",
      "levels": ["level1", "level2"]
    }
  ],
  "controls": [
    {
      "name": "control name",
      "description": "what this control does",
      "control_type": "positive|negative|vehicle|sham",
      "rationale": "why this control is needed"
    }
  ],
  "statistical_plan": {
    "primary_test": "e.g., two-way ANOVA",
    "significance_level": 0.05,
    "power": 0.8,
    "sample_size_per_group": 10,
    "sample_size_justification": "why this sample size",
    "corrections": ["e.g., Bonferroni"],
    "secondary_analyses": ["e.g., post-hoc Tukey"]
  },
  "methodology": ["step 1", "step 2", "..."],
  "expected_outcomes": ["outcome 1", "outcome 2"],
  "potential_pitfalls": ["pitfall 1", "pitfall 2"],
  "timeline_weeks": 12,
  "equipment_needed": ["equipment 1"],
  "reagents_needed": ["reagent 1"],
  "ethical_considerations": ["consideration 1"]
}

Return ONLY valid JSON — no markdown fences, no commentary.
"""

_REFINE_SYSTEM_PROMPT = """\
You are an expert experiment designer. You are given an existing experiment \
protocol (as JSON) and user feedback. Refine the protocol to address the \
feedback while maintaining scientific rigor. Return the complete updated \
protocol as a JSON object with the same structure as the input.

Return ONLY valid JSON — no markdown fences, no commentary.
"""


class ExperimentDesigner:
    """Generate structured experiment protocols from hypotheses using LLM.

    Accepts a ``chat_fn`` callable so tests can inject a mock without
    hitting the real LLM endpoint.
    """

    def __init__(self, chat_fn: Callable[..., str] | None = None) -> None:
        self._chat_fn = chat_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def design(
        self,
        hypothesis: Hypothesis,
        domain_context: str = "",
        constraints: dict[str, Any] | None = None,
    ) -> ExperimentProtocol:
        """Design an experiment to test a hypothesis.

        Args:
            hypothesis: The hypothesis to test.
            domain_context: Optional domain background to improve design.
            constraints: Optional dict with budget, timeline, equipment limitations.

        Returns:
            A fully structured ``ExperimentProtocol``.
        """
        chat = self._get_chat_fn()

        user_parts = [f"Hypothesis: {hypothesis.statement}"]
        if hypothesis.supporting_evidence:
            user_parts.append(
                "Supporting evidence:\n"
                + "\n".join(f"- {e}" for e in hypothesis.supporting_evidence)
            )
        if domain_context:
            user_parts.append(f"Domain context: {domain_context}")
        if constraints:
            user_parts.append(f"Constraints: {json.dumps(constraints)}")

        user_msg = "\n\n".join(user_parts)

        response = chat(
            messages=[
                {"role": "system", "content": _DESIGN_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=4096,
        )

        protocol = self._parse_protocol(
            response,
            hypothesis_id=hypothesis.id,
            hypothesis_statement=hypothesis.statement,
        )
        return protocol

    def refine(
        self,
        protocol: ExperimentProtocol,
        feedback: str,
    ) -> ExperimentProtocol:
        """Refine a protocol based on user feedback.

        Args:
            protocol: The existing protocol to refine.
            feedback: User feedback describing desired changes.

        Returns:
            An updated ``ExperimentProtocol``.
        """
        chat = self._get_chat_fn()

        protocol_json = protocol.model_dump()
        user_msg = (
            f"Current protocol:\n{json.dumps(protocol_json, indent=2)}\n\n"
            f"Feedback: {feedback}\n\n"
            "Please refine the protocol to address this feedback."
        )

        response = chat(
            messages=[
                {"role": "system", "content": _REFINE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=4096,
        )

        refined = self._parse_protocol(
            response,
            hypothesis_id=protocol.hypothesis_id,
            hypothesis_statement=protocol.hypothesis_statement,
        )
        return refined

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_chat_fn(self) -> Callable[..., str]:
        """Resolve the chat function (lazy import to avoid requiring tokens in tests)."""
        if self._chat_fn is not None:
            return self._chat_fn
        from scidex.llm.client import chat  # pragma: no cover

        return chat  # pragma: no cover

    def _parse_protocol(
        self,
        response: str,
        hypothesis_id: str,
        hypothesis_statement: str,
    ) -> ExperimentProtocol:
        """Parse LLM JSON response into an ExperimentProtocol."""
        data = self._extract_json(response)

        # Parse variables
        variables = []
        for v in data.get("variables", []):
            try:
                variables.append(
                    Variable(
                        name=v.get("name", ""),
                        variable_type=VariableType(v.get("variable_type", "independent")),
                        description=v.get("description", ""),
                        measurement_method=v.get("measurement_method", ""),
                        units=v.get("units", ""),
                        levels=v.get("levels", []),
                    )
                )
            except (ValueError, KeyError):
                logger.warning("Skipping malformed variable: %s", v)

        # Parse controls
        controls = []
        for c in data.get("controls", []):
            controls.append(
                Control(
                    name=c.get("name", ""),
                    description=c.get("description", ""),
                    control_type=c.get("control_type", "negative"),
                    rationale=c.get("rationale", ""),
                )
            )

        # Parse statistical plan
        sp_data = data.get("statistical_plan", {})
        statistical_plan = StatisticalPlan(
            primary_test=sp_data.get("primary_test", "t-test"),
            significance_level=sp_data.get("significance_level", 0.05),
            power=sp_data.get("power", 0.8),
            sample_size_per_group=sp_data.get("sample_size_per_group", 10),
            sample_size_justification=sp_data.get("sample_size_justification", ""),
            corrections=sp_data.get("corrections", []),
            secondary_analyses=sp_data.get("secondary_analyses", []),
        )

        return ExperimentProtocol(
            title=data.get("title", "Untitled Experiment"),
            hypothesis_id=hypothesis_id,
            hypothesis_statement=hypothesis_statement,
            objective=data.get("objective", ""),
            background=data.get("background", ""),
            variables=variables,
            controls=controls,
            statistical_plan=statistical_plan,
            methodology=data.get("methodology", []),
            expected_outcomes=data.get("expected_outcomes", []),
            potential_pitfalls=data.get("potential_pitfalls", []),
            timeline_weeks=data.get("timeline_weeks", 12),
            equipment_needed=data.get("equipment_needed", []),
            reagents_needed=data.get("reagents_needed", []),
            ethical_considerations=data.get("ethical_considerations", []),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _extract_json(response: str) -> dict:
        """Extract a JSON object from the LLM response, stripping fences."""
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(line for line in lines if not line.strip().startswith("```"))
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM response as JSON: %s", text[:200])
        return {}
