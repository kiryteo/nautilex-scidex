"""Tests for ContradictionMiner — LLM-based contradiction detection (mocked)."""

from __future__ import annotations

import json

import pytest

from scidex.hypothesis.contradictions import ContradictionMiner


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


def _mock_chat_success(messages, **kwargs) -> str:
    """Return a well-formed JSON response with one contradiction."""
    return json.dumps(
        [
            {
                "paper_a": "Study of X in cancer",
                "paper_b": "Re-evaluation of X",
                "claim_a": "X promotes tumour growth",
                "claim_b": "X inhibits tumour growth",
                "resolution_direction": "Dosage-dependent effect of X on tumour growth",
            }
        ]
    )


def _mock_chat_two_contradictions(messages, **kwargs) -> str:
    return json.dumps(
        [
            {
                "paper_a": "Paper A",
                "paper_b": "Paper B",
                "claim_a": "Claim A1",
                "claim_b": "Claim B1",
                "resolution_direction": "Resolution 1",
            },
            {
                "paper_a": "Paper C",
                "paper_b": "Paper D",
                "claim_a": "Claim C1",
                "claim_b": "Claim D1",
                "resolution_direction": "Resolution 2",
            },
        ]
    )


def _mock_chat_bad_json(messages, **kwargs) -> str:
    """Return unparseable response."""
    return "This is not JSON at all."


def _mock_chat_markdown_fence(messages, **kwargs) -> str:
    """Return JSON wrapped in markdown code fences."""
    return (
        "```json\n"
        + json.dumps(
            [
                {
                    "paper_a": "Paper X",
                    "paper_b": "Paper Y",
                    "claim_a": "Foo",
                    "claim_b": "Bar",
                    "resolution_direction": "Baz",
                }
            ]
        )
        + "\n```"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PAPERS = [
    {
        "title": "Study of X in cancer",
        "abstract": "We found that X promotes tumour growth via pathway P.",
    },
    {
        "title": "Re-evaluation of X",
        "abstract": "Our results show X inhibits tumour growth in a dose-dependent manner.",
    },
]


@pytest.fixture
def miner():
    return ContradictionMiner(chat_fn=_mock_chat_success)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFindContradictions:
    def test_returns_hypotheses(self, miner):
        results = miner.find_contradictions(SAMPLE_PAPERS, "cancer treatment")
        assert len(results) == 1

    def test_hypothesis_type(self, miner):
        results = miner.find_contradictions(SAMPLE_PAPERS, "cancer treatment")
        assert results[0].hypothesis_type == "contradiction_resolution"

    def test_evidence_contains_claims(self, miner):
        results = miner.find_contradictions(SAMPLE_PAPERS, "cancer treatment")
        evidence = " ".join(results[0].supporting_evidence)
        assert "promotes" in evidence or "inhibits" in evidence

    def test_source_papers_populated(self, miner):
        results = miner.find_contradictions(SAMPLE_PAPERS, "cancer treatment")
        assert len(results[0].source_papers) == 2

    def test_multiple_contradictions(self):
        miner = ContradictionMiner(chat_fn=_mock_chat_two_contradictions)
        results = miner.find_contradictions(SAMPLE_PAPERS, "topic")
        assert len(results) == 2

    def test_bad_json_returns_empty(self):
        miner = ContradictionMiner(chat_fn=_mock_chat_bad_json)
        results = miner.find_contradictions(SAMPLE_PAPERS, "topic")
        assert results == []

    def test_markdown_fence_stripped(self):
        miner = ContradictionMiner(chat_fn=_mock_chat_markdown_fence)
        results = miner.find_contradictions(SAMPLE_PAPERS, "topic")
        assert len(results) == 1
        assert "Paper X" in results[0].statement

    def test_fewer_than_two_papers(self, miner):
        results = miner.find_contradictions([SAMPLE_PAPERS[0]], "topic")
        assert results == []

    def test_empty_papers(self, miner):
        results = miner.find_contradictions([], "topic")
        assert results == []
