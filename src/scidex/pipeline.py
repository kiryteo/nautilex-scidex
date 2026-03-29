"""End-to-end SciDEX research pipeline.

Orchestrates:  Literature Search -> Knowledge Graph -> Hypothesis Generation
               -> GDE Refinement -> Experiment Design
"""

from __future__ import annotations

import logging
from typing import Callable

from scidex.errors import (
    ExperimentError,
    HypothesisError,
    KnowledgeGraphError,
    LiteratureError,
)
from scidex.knowledge.accumulator import KnowledgeAccumulator

logger = logging.getLogger(__name__)


class SciDEXPipeline:
    """Orchestrate the full research workflow.

    Literature Search -> Knowledge Graph -> Hypothesis Generation
    -> GDE -> Experiment Design
    """

    def __init__(
        self,
        knowledge_accumulator: KnowledgeAccumulator | None = None,
        on_progress: Callable[[str], None] | None = None,
        chat_fn: Callable[..., str] | None = None,
    ) -> None:
        """
        Args:
            knowledge_accumulator: Optional accumulator for persisting findings.
            on_progress: Callback for progress messages.
            chat_fn: LLM chat callable. If *None*, the real client is imported
                     only when needed (allows tests to run without tokens).
        """
        self.knowledge = knowledge_accumulator
        self.on_progress = on_progress
        self._chat_fn = chat_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        topic: str,
        max_papers: int = 20,
        gde_rounds: int = 3,
    ) -> dict:
        """Run the full pipeline for a research topic.

        Args:
            topic: Research topic / query string.
            max_papers: Maximum papers to fetch from Semantic Scholar.
            gde_rounds: Number of GDE debate rounds.

        Returns:
            Dict with keys: papers, knowledge_graph, hypotheses,
            gde_result, experiments, errors.
        """
        result: dict = {
            "topic": topic,
            "papers": [],
            "knowledge_graph": None,
            "hypotheses": [],
            "gde_result": None,
            "experiments": [],
            "errors": [],
        }

        # ------ 1. Search literature ------
        papers = self._search_literature(topic, max_papers, result)

        # ------ 2. Build knowledge graph ------
        kg = self._build_knowledge_graph(papers, result)

        # ------ 3. Generate hypotheses ------
        hypotheses = self._generate_hypotheses(kg, topic, result)

        # ------ 4. Run GDE ------
        gde_result = self._run_gde(kg, topic, hypotheses, gde_rounds, result)

        # ------ 5. Design experiments for top hypotheses ------
        self._design_experiments(gde_result, hypotheses, result)

        # ------ 6. Accumulate knowledge ------
        self._accumulate(result)

        return result

    # ------------------------------------------------------------------
    # Pipeline stages (each catches its own errors)
    # ------------------------------------------------------------------

    def _search_literature(self, topic: str, max_papers: int, result: dict) -> list:
        self._progress(f"Searching literature for '{topic}'...")
        try:
            from scidex.literature.s2_client import S2Client

            client = S2Client()
            papers = client.search_papers(topic, limit=max_papers)
            result["papers"] = [p.to_dict() for p in papers]
            self._progress(f"Found {len(papers)} papers.")
            return papers
        except Exception as exc:
            msg = f"Literature search failed: {exc}"
            logger.warning(msg)
            result["errors"].append(msg)
            return []

    def _build_knowledge_graph(self, papers: list, result: dict):
        self._progress("Building knowledge graph...")
        try:
            from scidex.knowledge_graph.graph import KnowledgeGraph

            kg = KnowledgeGraph()
            for paper in papers:
                meta = paper.to_dict() if hasattr(paper, "to_dict") else paper
                # Normalise key names for KnowledgeGraph.add_paper
                if "paper_id" in meta and "paperId" not in meta:
                    meta["paperId"] = meta["paper_id"]
                kg.add_paper(meta)
            stats = kg.get_statistics()
            result["knowledge_graph"] = kg.to_json()
            self._progress(f"Knowledge graph: {stats.num_nodes} nodes, {stats.num_edges} edges.")
            return kg
        except Exception as exc:
            msg = f"Knowledge graph construction failed: {exc}"
            logger.warning(msg)
            result["errors"].append(msg)
            return None

    def _generate_hypotheses(self, kg, topic: str, result: dict) -> list:
        if kg is None:
            return []
        self._progress("Generating hypotheses...")
        try:
            from scidex.hypothesis.generator import HypothesisGenerator

            gen = HypothesisGenerator(chat_fn=self._chat_fn)
            report = gen.generate(kg, topic)
            result["hypotheses"] = [h.model_dump() for h in report.hypotheses]
            self._progress(f"Generated {len(report.hypotheses)} hypotheses.")
            return report.hypotheses
        except Exception as exc:
            msg = f"Hypothesis generation failed: {exc}"
            logger.warning(msg)
            result["errors"].append(msg)
            return []

    def _run_gde(self, kg, topic: str, hypotheses: list, gde_rounds: int, result: dict):
        if not hypotheses:
            return None
        self._progress("Running Generate-Debate-Evolve...")
        try:
            from scidex.hypothesis.gde import GenerateDebateEvolve

            gde = GenerateDebateEvolve(
                max_rounds=gde_rounds,
                survivors_per_round=2,
                chat_fn=self._chat_fn,
                on_progress=self.on_progress,
            )
            gde_result = gde.run(
                knowledge_graph=kg,
                topic=topic,
                initial_hypotheses=hypotheses,
            )
            result["gde_result"] = gde_result.model_dump()
            self._progress(
                f"GDE complete: {gde_result.total_rounds} rounds, "
                f"{gde_result.final_hypotheses} final hypotheses."
            )
            return gde_result
        except Exception as exc:
            msg = f"GDE failed: {exc}"
            logger.warning(msg)
            result["errors"].append(msg)
            return None

    def _design_experiments(self, gde_result, hypotheses: list, result: dict) -> None:
        if self._chat_fn is None:
            return
        # Pick top hypotheses: from GDE final list, or fallback to first few generated
        top_hypotheses = []
        if gde_result is not None and gde_result.final_hypotheses_list:
            from scidex.hypothesis.models import Hypothesis

            for fh in gde_result.final_hypotheses_list[:2]:
                try:
                    top_hypotheses.append(Hypothesis(**fh))
                except Exception:
                    pass
        if not top_hypotheses and hypotheses:
            top_hypotheses = hypotheses[:2]

        if not top_hypotheses:
            return

        self._progress("Designing experiments for top hypotheses...")
        try:
            from scidex.experiment.designer import ExperimentDesigner

            designer = ExperimentDesigner(chat_fn=self._chat_fn)
            for hyp in top_hypotheses:
                protocol = designer.design(hyp)
                result["experiments"].append(protocol.model_dump())
            self._progress(f"Designed {len(result['experiments'])} experiment protocols.")
        except Exception as exc:
            msg = f"Experiment design failed: {exc}"
            logger.warning(msg)
            result["errors"].append(msg)

    # ------------------------------------------------------------------
    # Knowledge accumulation
    # ------------------------------------------------------------------

    def _accumulate(self, result: dict) -> None:
        """Persist all findings to the knowledge accumulator."""
        if self.knowledge is None:
            return
        try:
            if result["papers"]:
                self.knowledge.add_papers(result["papers"])
            if result["knowledge_graph"]:
                nodes = result["knowledge_graph"].get("nodes", [])
                self.knowledge.add_entities(nodes)
            for h in result["hypotheses"]:
                self.knowledge.add_hypothesis(h)
            for e in result["experiments"]:
                self.knowledge.add_experiment(e)
        except Exception as exc:
            logger.warning("Knowledge accumulation failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _progress(self, message: str) -> None:
        logger.info(message)
        if self.on_progress:
            self.on_progress(message)
