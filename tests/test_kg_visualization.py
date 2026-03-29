"""Tests for knowledge graph visualization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from scidex.knowledge_graph.models import Entity, Relationship
from scidex.knowledge_graph.graph import KnowledgeGraph
from scidex.knowledge_graph.visualization import visualize_graph


@pytest.fixture
def populated_kg():
    kg = KnowledgeGraph()
    kg.add_paper(
        {
            "paperId": "p1",
            "title": "CRISPR for breast cancer",
            "abstract": "We used CRISPR to edit BRCA1 in breast cancer models.",
            "year": 2024,
            "authors": [{"name": "Alice", "authorId": "a1"}],
            "venue": "Nature",
        }
    )
    return kg


class TestVisualizeGraph:
    def test_generates_html_file(self, populated_kg):
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "graph.html")
            result = visualize_graph(populated_kg, out)
            assert Path(result).exists()
            content = Path(result).read_text()
            assert "<html>" in content.lower() or "<!doctype" in content.lower()

    def test_html_contains_node_labels(self, populated_kg):
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "graph.html")
            visualize_graph(populated_kg, out)
            content = Path(out).read_text()
            # pyvis embeds node data in the HTML
            assert "Alice" in content

    def test_center_entity_limits_scope(self, populated_kg):
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "sub.html")
            result = visualize_graph(populated_kg, out, center_entity="p1", depth=1)
            assert Path(result).exists()
            content = Path(result).read_text()
            assert len(content) > 0

    def test_empty_graph(self):
        kg = KnowledgeGraph()
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "empty.html")
            result = visualize_graph(kg, out)
            assert Path(result).exists()
            content = Path(result).read_text()
            assert "No nodes to display" in content

    def test_returns_absolute_path(self, populated_kg):
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "graph.html")
            result = visualize_graph(populated_kg, out)
            assert Path(result).is_absolute()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
