"""Export experiment protocols to various formats."""

from __future__ import annotations

from scidex.experiment.models import ExperimentProtocol, VariableType


class ProtocolExporter:
    """Export experiment protocols to various formats."""

    def to_markdown(self, protocol: ExperimentProtocol) -> str:
        """Export protocol as formatted Markdown.

        Args:
            protocol: The experiment protocol to export.

        Returns:
            A Markdown string with structured sections.
        """
        lines: list[str] = []

        lines.append(f"# {protocol.title}")
        lines.append("")
        lines.append(f"**Generated:** {protocol.generated_at}")
        lines.append("")

        # Hypothesis
        lines.append("## Hypothesis")
        lines.append("")
        lines.append(f"> {protocol.hypothesis_statement}")
        lines.append("")
        lines.append(f"*Hypothesis ID: {protocol.hypothesis_id}*")
        lines.append("")

        # Objective & Background
        lines.append("## Objective")
        lines.append("")
        lines.append(protocol.objective)
        lines.append("")

        lines.append("## Background")
        lines.append("")
        lines.append(protocol.background)
        lines.append("")

        # Variables
        lines.append("## Variables")
        lines.append("")
        lines.append("| Name | Type | Description | Measurement | Units |")
        lines.append("|------|------|-------------|-------------|-------|")
        for v in protocol.variables:
            lines.append(
                f"| {v.name} | {v.variable_type.value} | {v.description} "
                f"| {v.measurement_method} | {v.units} |"
            )
        lines.append("")

        # Show levels for variables that have them
        vars_with_levels = [v for v in protocol.variables if v.levels]
        if vars_with_levels:
            lines.append("### Variable Levels")
            lines.append("")
            for v in vars_with_levels:
                lines.append(f"- **{v.name}**: {', '.join(v.levels)}")
            lines.append("")

        # Controls
        lines.append("## Controls")
        lines.append("")
        lines.append("| Name | Type | Description | Rationale |")
        lines.append("|------|------|-------------|-----------|")
        for c in protocol.controls:
            lines.append(f"| {c.name} | {c.control_type} | {c.description} | {c.rationale} |")
        lines.append("")

        # Statistical Plan
        sp = protocol.statistical_plan
        lines.append("## Statistical Plan")
        lines.append("")
        lines.append(f"- **Primary test:** {sp.primary_test}")
        lines.append(f"- **Significance level:** {sp.significance_level}")
        lines.append(f"- **Power:** {sp.power}")
        lines.append(f"- **Sample size per group:** {sp.sample_size_per_group}")
        lines.append(f"- **Justification:** {sp.sample_size_justification}")
        if sp.corrections:
            lines.append(f"- **Corrections:** {', '.join(sp.corrections)}")
        if sp.secondary_analyses:
            lines.append(f"- **Secondary analyses:** {', '.join(sp.secondary_analyses)}")
        lines.append("")

        # Methodology
        lines.append("## Methodology")
        lines.append("")
        for i, step in enumerate(protocol.methodology, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

        # Expected Outcomes
        lines.append("## Expected Outcomes")
        lines.append("")
        for outcome in protocol.expected_outcomes:
            lines.append(f"- {outcome}")
        lines.append("")

        # Potential Pitfalls
        lines.append("## Potential Pitfalls")
        lines.append("")
        for pitfall in protocol.potential_pitfalls:
            lines.append(f"- {pitfall}")
        lines.append("")

        # Timeline
        lines.append("## Timeline")
        lines.append("")
        lines.append(f"Estimated duration: **{protocol.timeline_weeks} weeks**")
        lines.append("")

        # Equipment & Reagents
        if protocol.equipment_needed:
            lines.append("## Equipment Needed")
            lines.append("")
            for eq in protocol.equipment_needed:
                lines.append(f"- {eq}")
            lines.append("")

        if protocol.reagents_needed:
            lines.append("## Reagents Needed")
            lines.append("")
            for rg in protocol.reagents_needed:
                lines.append(f"- {rg}")
            lines.append("")

        # Ethical Considerations
        if protocol.ethical_considerations:
            lines.append("## Ethical Considerations")
            lines.append("")
            for ec in protocol.ethical_considerations:
                lines.append(f"- {ec}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self, protocol: ExperimentProtocol) -> dict:
        """Export as JSON-serializable dict.

        Args:
            protocol: The experiment protocol to export.

        Returns:
            A dict representation of the protocol.
        """
        return protocol.model_dump()
