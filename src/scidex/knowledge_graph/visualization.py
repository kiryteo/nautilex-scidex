"""Interactive knowledge graph visualization using pyvis.

Generates standalone HTML files with color-coded nodes and labeled edges.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyvis.network import Network

from scidex.knowledge_graph.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Color scheme by entity type
_NODE_COLORS: dict[str, str] = {
    "paper": "#4A90D9",  # blue
    "author": "#27AE60",  # green
    "method": "#E67E22",  # orange
    "gene": "#E74C3C",  # red
    "protein": "#E74C3C",  # red
    "disease": "#9B59B6",  # purple
    "compound": "#F39C12",  # amber
    "venue": "#95A5A6",  # gray
    "institution": "#1ABC9C",  # teal
    "dataset": "#3498DB",  # light blue
    "unknown": "#BDC3C7",  # light gray
}

_NODE_SHAPES: dict[str, str] = {
    "paper": "dot",
    "author": "diamond",
    "method": "triangle",
    "gene": "star",
    "protein": "star",
    "disease": "square",
    "venue": "box",
    "institution": "box",
}


def visualize_graph(
    kg: KnowledgeGraph,
    output_path: str,
    center_entity: str | None = None,
    depth: int = 2,
) -> str:
    """Render an interactive knowledge graph visualization as HTML.

    Args:
        kg: The knowledge graph to visualize.
        output_path: Path to write the HTML file.
        center_entity: If given, only show the local neighborhood of this
            entity (by ID). If None, render the full graph.
        depth: Neighborhood depth when center_entity is set.

    Returns:
        Absolute path to the generated HTML file.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Get the data to visualize
    if center_entity is not None:
        data = kg.get_subgraph(center_entity, depth=depth)
    else:
        data = kg.to_json()

    if not data.get("nodes"):
        # Write a minimal HTML page
        output.write_text("<html><body><p>No nodes to display.</p></body></html>")
        return str(output.resolve())

    # Build pyvis network
    net = Network(
        height="600px",
        width="100%",
        directed=True,
        bgcolor="#FFFFFF",
        font_color="#333333",
    )
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)

    # Add nodes
    for node in data["nodes"]:
        entity_type = node.get("entity_type", "unknown")
        color = _NODE_COLORS.get(entity_type, _NODE_COLORS["unknown"])
        shape = _NODE_SHAPES.get(entity_type, "dot")

        # Build hover title
        props = node.get("properties", {})
        title_parts = [
            f"<b>{node['name']}</b>",
            f"Type: {entity_type}",
        ]
        for k, v in props.items():
            if v is not None:
                title_parts.append(f"{k}: {v}")
        title = "<br>".join(title_parts)

        # Truncate long names for display
        label = node["name"]
        if len(label) > 40:
            label = label[:37] + "..."

        net.add_node(
            node["id"],
            label=label,
            color=color,
            shape=shape,
            title=title,
            size=15 if entity_type == "paper" else 10,
        )

    # Add edges
    for edge in data["edges"]:
        net.add_edge(
            edge["source_id"],
            edge["target_id"],
            label=edge.get("rel_type", ""),
            title=edge.get("evidence", ""),
            color="#888888",
            arrows="to",
        )

    net.write_html(str(output))
    logger.info(f"Graph visualization written to {output}")
    return str(output.resolve())
