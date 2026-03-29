"""Hypothesis generation — gap detection, Swanson linking, analogies, contradictions, and GDE."""

from scidex.hypothesis.models import Hypothesis, HypothesisReport, GapAnalysis
from scidex.hypothesis.gap_detector import GapDetector
from scidex.hypothesis.swanson_linker import SwansonLinker
from scidex.hypothesis.analogy_engine import AnalogyEngine
from scidex.hypothesis.contradictions import ContradictionMiner
from scidex.hypothesis.generator import HypothesisGenerator
from scidex.hypothesis.critic import CriticScore, HypothesisCritic
from scidex.hypothesis.tournament import TournamentSelector, TournamentResult
from scidex.hypothesis.evolver import HypothesisEvolver
from scidex.hypothesis.gde import GenerateDebateEvolve, GDEResult, GDERoundResult

__all__ = [
    "Hypothesis",
    "HypothesisReport",
    "GapAnalysis",
    "GapDetector",
    "SwansonLinker",
    "AnalogyEngine",
    "ContradictionMiner",
    "HypothesisGenerator",
    "CriticScore",
    "HypothesisCritic",
    "TournamentSelector",
    "TournamentResult",
    "HypothesisEvolver",
    "GenerateDebateEvolve",
    "GDEResult",
    "GDERoundResult",
]
