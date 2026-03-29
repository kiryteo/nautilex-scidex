"""Shared exception hierarchy for the SciDEX platform."""


class SciDEXError(Exception):
    """Base exception for all SciDEX errors."""


class LiteratureError(SciDEXError):
    """Error during literature search or ingestion."""


class KnowledgeGraphError(SciDEXError):
    """Error during knowledge graph construction or querying."""


class HypothesisError(SciDEXError):
    """Error during hypothesis generation or refinement."""


class ExperimentError(SciDEXError):
    """Error during experiment design or export."""
