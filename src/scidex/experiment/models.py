"""Pydantic models for experiment design — protocols, variables, controls, and statistics."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class VariableType(str, Enum):
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"
    CONTROLLED = "controlled"
    CONFOUNDING = "confounding"


class Variable(BaseModel):
    """A single experimental variable."""

    name: str
    variable_type: VariableType
    description: str
    measurement_method: str = ""
    units: str = ""
    levels: list[str] = Field(default_factory=list)


class Control(BaseModel):
    """An experimental control."""

    name: str
    description: str
    control_type: str  # "positive", "negative", "vehicle", "sham"
    rationale: str


class StatisticalPlan(BaseModel):
    """Statistical analysis plan for an experiment."""

    primary_test: str  # e.g., "two-way ANOVA"
    significance_level: float = 0.05
    power: float = 0.8
    sample_size_per_group: int
    sample_size_justification: str
    corrections: list[str] = Field(default_factory=list)
    secondary_analyses: list[str] = Field(default_factory=list)


class ExperimentProtocol(BaseModel):
    """A complete experiment protocol generated from a hypothesis."""

    title: str
    hypothesis_id: str
    hypothesis_statement: str
    objective: str
    background: str

    variables: list[Variable]
    controls: list[Control]
    statistical_plan: StatisticalPlan

    methodology: list[str]  # Step-by-step procedure
    expected_outcomes: list[str]
    potential_pitfalls: list[str]
    timeline_weeks: int = 12

    equipment_needed: list[str] = Field(default_factory=list)
    reagents_needed: list[str] = Field(default_factory=list)
    ethical_considerations: list[str] = Field(default_factory=list)

    generated_at: str = ""
