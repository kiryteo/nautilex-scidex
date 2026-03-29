"""Tests for the Generate-Debate-Evolve (GDE) system.

Covers: CriticScore model, HypothesisCritic, TournamentSelector,
HypothesisEvolver, and the GenerateDebateEvolve orchestrator.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from scidex.hypothesis.models import Hypothesis
from scidex.hypothesis.critic import CriticScore, HypothesisCritic, PERSPECTIVES
from scidex.hypothesis.tournament import TournamentSelector, TournamentResult
from scidex.hypothesis.evolver import HypothesisEvolver
from scidex.hypothesis.gde import GenerateDebateEvolve, GDEResult, GDERoundResult
from scidex.knowledge_graph.graph import KnowledgeGraph
from scidex.knowledge_graph.models import Entity, Relationship


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(timezone.utc).isoformat()


def _make_hypothesis(
    hid: str = "h1",
    statement: str = "Test hypothesis",
    **overrides,
) -> Hypothesis:
    defaults = dict(
        id=hid,
        statement=statement,
        hypothesis_type="gap_filling",
        supporting_evidence=["evidence 1", "evidence 2"],
        confidence=0.5,
        novelty_score=0.6,
        testability_score=0.5,
        source_papers=[],
        generated_at=_NOW,
    )
    defaults.update(overrides)
    return Hypothesis(**defaults)


def _make_critic_score(
    hypothesis_id: str = "h1",
    critic_id: str = "test_critic",
    overall_score: float = 0.7,
    **overrides,
) -> CriticScore:
    defaults = dict(
        hypothesis_id=hypothesis_id,
        critic_id=critic_id,
        novelty=0.7,
        feasibility=0.6,
        testability=0.7,
        significance=0.8,
        strengths=["Good evidence"],
        weaknesses=["Vague mechanism"],
        suggested_improvements=["Specify the target gene"],
        overall_score=overall_score,
        reasoning="Reasonable hypothesis with clear improvements needed.",
    )
    defaults.update(overrides)
    return CriticScore(**defaults)


def _mock_critic_chat(messages, **kwargs) -> str:
    """Return a valid CriticScore JSON from any critic prompt."""
    return json.dumps(
        {
            "novelty": 0.8,
            "feasibility": 0.7,
            "testability": 0.75,
            "significance": 0.9,
            "strengths": ["Novel approach", "Strong evidence base"],
            "weaknesses": ["Sample size concerns"],
            "suggested_improvements": ["Add control group"],
            "overall_score": 0.8,
            "reasoning": "A promising hypothesis that needs methodological refinement.",
        }
    )


def _mock_evolve_chat(messages, **kwargs) -> str:
    """Return evolved hypotheses JSON."""
    return json.dumps(
        [
            {
                "statement": "Evolved hypothesis combining parent insights",
                "supporting_evidence": ["combined evidence 1", "combined evidence 2"],
                "parent_ids": ["h1", "h2"],
                "improvements_applied": ["Added control group"],
            },
            {
                "statement": "Second evolved hypothesis addressing weaknesses",
                "supporting_evidence": ["refined evidence"],
                "parent_ids": ["h1"],
                "improvements_applied": ["Specified target gene"],
            },
        ]
    )


def _mock_combined_chat(messages, **kwargs) -> str:
    """Route to critic or evolver based on prompt content."""
    content = messages[-1]["content"] if messages else ""
    if "Evaluate the following hypothesis" in content:
        return _mock_critic_chat(messages, **kwargs)
    if "evolution" in content.lower() or "improved hypotheses" in content.lower():
        return _mock_evolve_chat(messages, **kwargs)
    # Contradiction mining
    if "Identify contradictions" in content:
        return json.dumps([])
    # Testability refinement
    return json.dumps([])


# ---------------------------------------------------------------------------
# CriticScore model tests
# ---------------------------------------------------------------------------


class TestCriticScore:
    def test_valid_score(self):
        score = CriticScore(
            hypothesis_id="h1",
            critic_id="c1",
            novelty=0.5,
            feasibility=0.5,
            testability=0.5,
            significance=0.5,
            overall_score=0.5,
        )
        assert score.novelty == 0.5

    def test_score_bounds_min(self):
        with pytest.raises(Exception):
            CriticScore(
                hypothesis_id="h1",
                critic_id="c1",
                novelty=-0.1,
                feasibility=0.5,
                testability=0.5,
                significance=0.5,
                overall_score=0.5,
            )

    def test_score_bounds_max(self):
        with pytest.raises(Exception):
            CriticScore(
                hypothesis_id="h1",
                critic_id="c1",
                novelty=1.1,
                feasibility=0.5,
                testability=0.5,
                significance=0.5,
                overall_score=0.5,
            )

    def test_default_lists(self):
        score = CriticScore(
            hypothesis_id="h1",
            critic_id="c1",
            novelty=0.5,
            feasibility=0.5,
            testability=0.5,
            significance=0.5,
            overall_score=0.5,
        )
        assert score.strengths == []
        assert score.weaknesses == []
        assert score.suggested_improvements == []

    def test_edge_values_zero_and_one(self):
        score = CriticScore(
            hypothesis_id="h1",
            critic_id="c1",
            novelty=0.0,
            feasibility=1.0,
            testability=0.0,
            significance=1.0,
            overall_score=0.0,
        )
        assert score.novelty == 0.0
        assert score.feasibility == 1.0

    def test_serialization_round_trip(self):
        score = _make_critic_score()
        data = score.model_dump()
        restored = CriticScore(**data)
        assert restored.overall_score == score.overall_score
        assert restored.strengths == score.strengths


# ---------------------------------------------------------------------------
# HypothesisCritic tests
# ---------------------------------------------------------------------------


class TestHypothesisCritic:
    def test_init_valid_perspectives(self):
        for p in PERSPECTIVES:
            critic = HypothesisCritic(perspective=p, chat_fn=_mock_critic_chat)
            assert critic.perspective == p

    def test_init_invalid_perspective_raises(self):
        with pytest.raises(ValueError, match="Unknown perspective"):
            HypothesisCritic(perspective="philosopher", chat_fn=_mock_critic_chat)

    def test_critic_id_auto_generated(self):
        critic = HypothesisCritic(perspective="methodologist", chat_fn=_mock_critic_chat)
        assert critic.critic_id.startswith("methodologist_")

    def test_critic_id_custom(self):
        critic = HypothesisCritic(
            perspective="innovator", critic_id="my_critic", chat_fn=_mock_critic_chat
        )
        assert critic.critic_id == "my_critic"

    def test_critique_returns_critic_score(self):
        critic = HypothesisCritic(perspective="methodologist", chat_fn=_mock_critic_chat)
        score = critic.critique("A novel gene therapy approach", hypothesis_id="h1")
        assert isinstance(score, CriticScore)
        assert score.hypothesis_id == "h1"

    def test_critique_parses_scores(self):
        critic = HypothesisCritic(perspective="domain_expert", chat_fn=_mock_critic_chat)
        score = critic.critique("Test hypothesis", hypothesis_id="h1")
        assert score.novelty == 0.8
        assert score.overall_score == 0.8
        assert "Novel approach" in score.strengths

    def test_critique_with_evidence(self):
        critic = HypothesisCritic(perspective="innovator", chat_fn=_mock_critic_chat)
        score = critic.critique(
            "Test hypothesis",
            hypothesis_id="h2",
            evidence=["Paper shows X", "Study confirms Y"],
        )
        assert isinstance(score, CriticScore)

    def test_critique_with_domain_context(self):
        critic = HypothesisCritic(perspective="methodologist", chat_fn=_mock_critic_chat)
        score = critic.critique(
            "Test hypothesis",
            hypothesis_id="h3",
            domain_context="Oncology research",
        )
        assert isinstance(score, CriticScore)

    def test_critique_handles_bad_json(self):
        def bad_chat(messages, **kwargs):
            return "This is not JSON"

        critic = HypothesisCritic(perspective="methodologist", chat_fn=bad_chat)
        score = critic.critique("Test", hypothesis_id="h1")
        # Should return defaults (0.5) rather than crashing
        assert isinstance(score, CriticScore)
        assert score.novelty == 0.5

    def test_critique_handles_markdown_fenced_json(self):
        def fenced_chat(messages, **kwargs):
            return (
                "```json\n"
                + json.dumps(
                    {
                        "novelty": 0.9,
                        "feasibility": 0.8,
                        "testability": 0.7,
                        "significance": 0.6,
                        "strengths": [],
                        "weaknesses": [],
                        "suggested_improvements": [],
                        "overall_score": 0.75,
                        "reasoning": "good",
                    }
                )
                + "\n```"
            )

        critic = HypothesisCritic(perspective="domain_expert", chat_fn=fenced_chat)
        score = critic.critique("Test", hypothesis_id="h1")
        assert score.novelty == 0.9

    def test_critique_clamps_out_of_range_values(self):
        def extreme_chat(messages, **kwargs):
            return json.dumps(
                {
                    "novelty": 5.0,
                    "feasibility": -1.0,
                    "testability": 0.5,
                    "significance": 0.5,
                    "overall_score": 0.5,
                }
            )

        critic = HypothesisCritic(perspective="innovator", chat_fn=extreme_chat)
        score = critic.critique("Test", hypothesis_id="h1")
        assert score.novelty == 1.0
        assert score.feasibility == 0.0


# ---------------------------------------------------------------------------
# TournamentSelector tests
# ---------------------------------------------------------------------------


class TestTournamentSelector:
    def test_select_returns_tournament_result(self):
        selector = TournamentSelector(survival_count=1)
        h1 = _make_hypothesis("h1", "Hypo 1")
        h2 = _make_hypothesis("h2", "Hypo 2")
        scores = {
            "h1": [_make_critic_score("h1", overall_score=0.9)],
            "h2": [_make_critic_score("h2", overall_score=0.3)],
        }
        result = selector.select([h1, h2], scores)
        assert isinstance(result, TournamentResult)

    def test_correct_survivor_count(self):
        selector = TournamentSelector(survival_count=1)
        h1 = _make_hypothesis("h1")
        h2 = _make_hypothesis("h2")
        h3 = _make_hypothesis("h3")
        scores = {
            "h1": [_make_critic_score("h1", overall_score=0.9)],
            "h2": [_make_critic_score("h2", overall_score=0.5)],
            "h3": [_make_critic_score("h3", overall_score=0.3)],
        }
        result = selector.select([h1, h2, h3], scores)
        assert len(result.survivors) == 1
        assert len(result.eliminated) == 2

    def test_highest_score_survives(self):
        selector = TournamentSelector(survival_count=1)
        h1 = _make_hypothesis("h1", "Low scorer")
        h2 = _make_hypothesis("h2", "High scorer")
        scores = {
            "h1": [_make_critic_score("h1", overall_score=0.2, novelty=0.2, significance=0.2)],
            "h2": [_make_critic_score("h2", overall_score=0.95, novelty=0.95, significance=0.95)],
        }
        result = selector.select([h1, h2], scores)
        assert result.survivors[0].id == "h2"

    def test_rankings_sorted_descending(self):
        selector = TournamentSelector(survival_count=2)
        hypotheses = [_make_hypothesis(f"h{i}") for i in range(4)]
        scores = {
            "h0": [_make_critic_score("h0", overall_score=0.3)],
            "h1": [_make_critic_score("h1", overall_score=0.9)],
            "h2": [_make_critic_score("h2", overall_score=0.5)],
            "h3": [_make_critic_score("h3", overall_score=0.7)],
        }
        result = selector.select(hypotheses, scores)
        ranking_scores = [s for _, s in result.rankings]
        assert ranking_scores == sorted(ranking_scores, reverse=True)

    def test_empty_hypotheses(self):
        selector = TournamentSelector(survival_count=2)
        result = selector.select([], {})
        assert result.survivors == []
        assert result.eliminated == []
        assert result.rankings == []

    def test_survival_count_exceeds_hypotheses(self):
        selector = TournamentSelector(survival_count=10)
        h1 = _make_hypothesis("h1")
        scores = {"h1": [_make_critic_score("h1")]}
        result = selector.select([h1], scores)
        assert len(result.survivors) == 1
        assert len(result.eliminated) == 0

    def test_multiple_critics_averaged(self):
        selector = TournamentSelector(survival_count=1)
        h1 = _make_hypothesis("h1")
        h2 = _make_hypothesis("h2")
        # h1 gets two scores averaging higher than h2
        scores = {
            "h1": [
                _make_critic_score(
                    "h1", critic_id="c1", overall_score=0.9, novelty=0.9, significance=0.9
                ),
                _make_critic_score(
                    "h1", critic_id="c2", overall_score=0.8, novelty=0.8, significance=0.8
                ),
            ],
            "h2": [
                _make_critic_score(
                    "h2", critic_id="c1", overall_score=0.3, novelty=0.3, significance=0.3
                ),
            ],
        }
        result = selector.select([h1, h2], scores)
        assert result.survivors[0].id == "h1"

    def test_no_scores_for_hypothesis_gets_zero(self):
        selector = TournamentSelector(survival_count=1)
        h1 = _make_hypothesis("h1")
        h2 = _make_hypothesis("h2")
        scores = {
            "h1": [_make_critic_score("h1", overall_score=0.5)],
            # h2 has no scores
        }
        result = selector.select([h1, h2], scores)
        assert result.survivors[0].id == "h1"

    def test_tied_scores(self):
        """When scores are exactly tied, all should still be ranked."""
        selector = TournamentSelector(survival_count=1)
        h1 = _make_hypothesis("h1")
        h2 = _make_hypothesis("h2")
        scores = {
            "h1": [_make_critic_score("h1", overall_score=0.5)],
            "h2": [_make_critic_score("h2", overall_score=0.5)],
        }
        result = selector.select([h1, h2], scores)
        assert len(result.survivors) == 1
        assert len(result.eliminated) == 1
        assert len(result.rankings) == 2


# ---------------------------------------------------------------------------
# HypothesisEvolver tests
# ---------------------------------------------------------------------------


class TestHypothesisEvolver:
    def test_evolve_returns_hypotheses(self):
        evolver = HypothesisEvolver(chat_fn=_mock_evolve_chat)
        survivors = [_make_hypothesis("h1"), _make_hypothesis("h2")]
        scores = {
            "h1": [_make_critic_score("h1")],
            "h2": [_make_critic_score("h2")],
        }
        result = evolver.evolve(survivors, scores, round_number=1)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(h, Hypothesis) for h in result)

    def test_evolved_hypotheses_are_extension_type(self):
        evolver = HypothesisEvolver(chat_fn=_mock_evolve_chat)
        survivors = [_make_hypothesis("h1")]
        scores = {"h1": [_make_critic_score("h1")]}
        result = evolver.evolve(survivors, scores, round_number=1)
        assert all(h.hypothesis_type == "extension" for h in result)

    def test_evolve_empty_survivors(self):
        evolver = HypothesisEvolver(chat_fn=_mock_evolve_chat)
        result = evolver.evolve([], {}, round_number=1)
        assert result == []

    def test_evolve_fallback_on_bad_json(self):
        def bad_chat(messages, **kwargs):
            return "not json"

        evolver = HypothesisEvolver(chat_fn=bad_chat)
        survivors = [_make_hypothesis("h1")]
        scores = {"h1": [_make_critic_score("h1")]}
        # Should fallback to returning survivors
        result = evolver.evolve(survivors, scores, round_number=1)
        assert len(result) == 1
        assert result[0].id == "h1"

    def test_evolve_confidence_increases_with_rounds(self):
        evolver = HypothesisEvolver(chat_fn=_mock_evolve_chat)
        survivors = [_make_hypothesis("h1")]
        scores = {"h1": [_make_critic_score("h1")]}
        r1 = evolver.evolve(survivors, scores, round_number=1)
        r3 = evolver.evolve(survivors, scores, round_number=3)
        assert r3[0].confidence > r1[0].confidence

    def test_evolved_have_unique_ids(self):
        evolver = HypothesisEvolver(chat_fn=_mock_evolve_chat)
        survivors = [_make_hypothesis("h1"), _make_hypothesis("h2")]
        scores = {
            "h1": [_make_critic_score("h1")],
            "h2": [_make_critic_score("h2")],
        }
        result = evolver.evolve(survivors, scores, round_number=1)
        ids = [h.id for h in result]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# GDE Orchestrator tests
# ---------------------------------------------------------------------------


@pytest.fixture
def kg_for_gde():
    """A small knowledge graph for GDE testing."""
    kg = KnowledgeGraph()
    entities = [
        Entity(id="g1", name="BRCA1", entity_type="gene"),
        Entity(id="g2", name="TP53", entity_type="gene"),
        Entity(id="d1", name="breast cancer", entity_type="disease"),
        Entity(id="m1", name="CRISPR", entity_type="method"),
    ]
    for e in entities:
        kg.add_entity(e)
    rels = [
        Relationship(source_id="m1", target_id="g1", rel_type="USES_METHOD"),
        Relationship(source_id="g1", target_id="d1", rel_type="RELATED_TO"),
        Relationship(source_id="g2", target_id="d1", rel_type="RELATED_TO"),
    ]
    for r in rels:
        kg.add_relationship(r)
    return kg


class TestGenerateDebateEvolve:
    def test_run_with_initial_hypotheses_single_round(self):
        gde = GenerateDebateEvolve(
            max_rounds=1,
            survivors_per_round=1,
            chat_fn=_mock_combined_chat,
        )
        hypotheses = [_make_hypothesis("h1", "Hypo 1"), _make_hypothesis("h2", "Hypo 2")]
        result = gde.run(initial_hypotheses=hypotheses)
        assert isinstance(result, GDEResult)
        assert result.total_rounds == 1
        assert result.initial_hypotheses == 2

    def test_run_multi_round(self):
        gde = GenerateDebateEvolve(
            max_rounds=3,
            survivors_per_round=2,
            chat_fn=_mock_combined_chat,
        )
        hypotheses = [
            _make_hypothesis("h1", "Hypo 1"),
            _make_hypothesis("h2", "Hypo 2"),
            _make_hypothesis("h3", "Hypo 3"),
        ]
        result = gde.run(initial_hypotheses=hypotheses)
        assert result.total_rounds == 3
        assert len(result.rounds) == 3

    def test_round_results_have_correct_numbers(self):
        gde = GenerateDebateEvolve(
            max_rounds=1,
            survivors_per_round=1,
            chat_fn=_mock_combined_chat,
        )
        hypotheses = [_make_hypothesis("h1"), _make_hypothesis("h2")]
        result = gde.run(initial_hypotheses=hypotheses)
        rnd = result.rounds[0]
        assert rnd.round_number == 1
        assert rnd.hypotheses_in == 2
        assert rnd.hypotheses_out == 1
        assert len(rnd.survivors) == 1
        assert len(rnd.eliminated) == 1

    def test_final_hypotheses_populated(self):
        gde = GenerateDebateEvolve(
            max_rounds=1,
            survivors_per_round=2,
            chat_fn=_mock_combined_chat,
        )
        hypotheses = [_make_hypothesis("h1"), _make_hypothesis("h2")]
        result = gde.run(initial_hypotheses=hypotheses)
        assert len(result.final_hypotheses_list) > 0

    def test_best_hypothesis_selected(self):
        gde = GenerateDebateEvolve(
            max_rounds=1,
            survivors_per_round=1,
            chat_fn=_mock_combined_chat,
        )
        hypotheses = [_make_hypothesis("h1"), _make_hypothesis("h2")]
        result = gde.run(initial_hypotheses=hypotheses)
        assert result.best_hypothesis is not None
        assert "statement" in result.best_hypothesis

    def test_empty_initial_hypotheses(self):
        gde = GenerateDebateEvolve(
            max_rounds=2,
            survivors_per_round=1,
            chat_fn=_mock_combined_chat,
        )
        result = gde.run(initial_hypotheses=[])
        assert result.total_rounds == 0
        assert result.final_hypotheses == 0

    def test_no_knowledge_graph_no_hypotheses(self):
        gde = GenerateDebateEvolve(
            max_rounds=1,
            survivors_per_round=1,
            chat_fn=_mock_combined_chat,
        )
        result = gde.run()
        assert result.total_rounds == 0

    def test_run_from_knowledge_graph(self, kg_for_gde):
        gde = GenerateDebateEvolve(
            max_rounds=1,
            survivors_per_round=1,
            chat_fn=_mock_combined_chat,
        )
        result = gde.run(knowledge_graph=kg_for_gde, topic="gene therapy")
        assert isinstance(result, GDEResult)
        assert result.initial_hypotheses > 0

    def test_progress_callback_called(self):
        messages: list[str] = []

        def on_progress(msg: str) -> None:
            messages.append(msg)

        gde = GenerateDebateEvolve(
            max_rounds=1,
            survivors_per_round=1,
            chat_fn=_mock_combined_chat,
            on_progress=on_progress,
        )
        hypotheses = [_make_hypothesis("h1"), _make_hypothesis("h2")]
        gde.run(initial_hypotheses=hypotheses)
        assert len(messages) > 0
        assert any("Critics evaluating" in m for m in messages)
        assert any("Tournament selection" in m for m in messages)
        assert any("Evolution" in m for m in messages)

    def test_progress_callback_per_round(self):
        messages: list[str] = []

        gde = GenerateDebateEvolve(
            max_rounds=2,
            survivors_per_round=1,
            chat_fn=_mock_combined_chat,
            on_progress=lambda m: messages.append(m),
        )
        hypotheses = [_make_hypothesis("h1"), _make_hypothesis("h2")]
        gde.run(initial_hypotheses=hypotheses)
        round_1_msgs = [m for m in messages if "Round 1" in m]
        round_2_msgs = [m for m in messages if "Round 2" in m]
        assert len(round_1_msgs) >= 3
        assert len(round_2_msgs) >= 3

    def test_single_hypothesis_input(self):
        gde = GenerateDebateEvolve(
            max_rounds=1,
            survivors_per_round=1,
            chat_fn=_mock_combined_chat,
        )
        result = gde.run(initial_hypotheses=[_make_hypothesis("h1")])
        assert result.total_rounds == 1
        assert result.final_hypotheses >= 1

    def test_critic_scores_in_round_result(self):
        gde = GenerateDebateEvolve(
            max_rounds=1,
            survivors_per_round=2,
            chat_fn=_mock_combined_chat,
        )
        hypotheses = [_make_hypothesis("h1"), _make_hypothesis("h2")]
        result = gde.run(initial_hypotheses=hypotheses)
        rnd = result.rounds[0]
        assert len(rnd.critic_scores) > 0
        for hid, scores in rnd.critic_scores.items():
            assert len(scores) == 3  # 3 critics

    def test_evolved_hypotheses_in_round_result(self):
        gde = GenerateDebateEvolve(
            max_rounds=1,
            survivors_per_round=2,
            chat_fn=_mock_combined_chat,
        )
        hypotheses = [_make_hypothesis("h1"), _make_hypothesis("h2")]
        result = gde.run(initial_hypotheses=hypotheses)
        rnd = result.rounds[0]
        assert len(rnd.evolved_hypotheses) > 0

    def test_max_rounds_clamped_to_one_minimum(self):
        gde = GenerateDebateEvolve(
            max_rounds=0,
            survivors_per_round=1,
            chat_fn=_mock_combined_chat,
        )
        assert gde.max_rounds == 1

    def test_survivors_per_round_clamped_to_one_minimum(self):
        gde = GenerateDebateEvolve(
            max_rounds=1,
            survivors_per_round=0,
            chat_fn=_mock_combined_chat,
        )
        assert gde.survivors_per_round == 1


# ---------------------------------------------------------------------------
# GDEResult / GDERoundResult model tests
# ---------------------------------------------------------------------------


class TestGDEModels:
    def test_gde_round_result_defaults(self):
        rnd = GDERoundResult(round_number=1, hypotheses_in=5, hypotheses_out=2)
        assert rnd.survivors == []
        assert rnd.eliminated == []
        assert rnd.critic_scores == {}
        assert rnd.evolved_hypotheses == []

    def test_gde_result_defaults(self):
        result = GDEResult(total_rounds=0, initial_hypotheses=0, final_hypotheses=0)
        assert result.rounds == []
        assert result.final_hypotheses_list == []
        assert result.best_hypothesis is None

    def test_gde_result_serialization(self):
        result = GDEResult(
            total_rounds=1,
            initial_hypotheses=3,
            final_hypotheses=2,
            final_hypotheses_list=[{"statement": "test"}],
            best_hypothesis={"statement": "best"},
        )
        data = result.model_dump()
        restored = GDEResult(**data)
        assert restored.total_rounds == 1
        assert restored.best_hypothesis["statement"] == "best"
