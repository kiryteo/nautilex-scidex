"""Knowledge Graph Viewer — Explore entities and relationships from ingested papers."""

import streamlit as st
import streamlit.components.v1 as components
import tempfile
from pathlib import Path

from scidex.errors import KnowledgeGraphError
from scidex.knowledge_graph.graph import KnowledgeGraph
from scidex.knowledge_graph.visualization import visualize_graph
from scidex.literature.paper_store import PaperStore

st.header("Knowledge Graph Viewer")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "kg" not in st.session_state:
    # Try loading a saved graph
    kg_path = Path("data/knowledge_graph.json")
    if kg_path.exists():
        st.session_state.kg = KnowledgeGraph.load(kg_path)
    else:
        st.session_state.kg = KnowledgeGraph()

if "paper_store" not in st.session_state:
    st.session_state.paper_store = PaperStore()

kg: KnowledgeGraph = st.session_state.kg
store: PaperStore = st.session_state.paper_store

# ---------------------------------------------------------------------------
# Sidebar — Statistics
# ---------------------------------------------------------------------------

stats = kg.get_statistics()
st.sidebar.subheader("Graph Statistics")

col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Nodes", stats.num_nodes)
col_s2.metric("Edges", stats.num_edges)

if stats.node_types:
    st.sidebar.markdown("**Node Types**")
    for ntype, count in sorted(stats.node_types.items(), key=lambda x: -x[1]):
        st.sidebar.text(f"  {ntype}: {count}")

if stats.edge_types:
    st.sidebar.markdown("**Edge Types**")
    for etype, count in sorted(stats.edge_types.items(), key=lambda x: -x[1]):
        st.sidebar.text(f"  {etype}: {count}")

st.sidebar.metric("Papers in Store", store.count)

# ---------------------------------------------------------------------------
# Top-level metrics
# ---------------------------------------------------------------------------

if stats.num_nodes > 0:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Entities", stats.num_nodes)
    m2.metric("Total Relationships", stats.num_edges)
    m3.metric("Entity Types", len(stats.node_types))
    m4.metric("Relation Types", len(stats.edge_types))

# ---------------------------------------------------------------------------
# Add papers to knowledge graph
# ---------------------------------------------------------------------------

st.subheader("Add Papers to Knowledge Graph")

add_tab, manual_tab = st.tabs(["From Literature Store", "Manual Entry"])

with add_tab:
    if store.count == 0:
        st.info(
            "No papers in the literature store. Go to Literature Explorer to ingest papers first."
        )
    else:
        st.caption(f"{store.count} papers available in the literature store.")
        query = st.text_input(
            "Search store for papers to add",
            placeholder="e.g., gene therapy CRISPR",
            key="kg_store_search",
        )
        n_results = st.slider("Number of papers", 1, min(20, store.count), 5, key="kg_n_results")

        if st.button("Search & Add to Graph", disabled=not query):
            with st.spinner("Searching store and building graph..."):
                try:
                    papers = store.search_similar(query, n_results=n_results)
                    added_count = 0
                    for paper in papers:
                        paper_meta = {
                            "paperId": paper.paper_id,
                            "title": paper.title,
                            "abstract": paper.abstract,
                            "year": paper.year,
                            "citationCount": paper.citation_count,
                            "authors": [{"name": a} for a in paper.authors],
                        }
                        kg.add_paper(paper_meta)
                        added_count += 1

                    # Save after adding
                    save_path = Path("data/knowledge_graph.json")
                    kg.save(save_path)

                    st.success(f"Added {added_count} papers to the knowledge graph.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to add papers: {exc}")

with manual_tab:
    paper_id = st.text_input("Paper ID (Semantic Scholar)", key="manual_pid")
    paper_title = st.text_input("Title", key="manual_title")
    paper_abstract = st.text_area("Abstract", key="manual_abstract")

    if st.button("Add to Graph", disabled=not (paper_id and paper_title)):
        try:
            meta = {
                "paperId": paper_id,
                "title": paper_title,
                "abstract": paper_abstract,
            }
            kg.add_paper(meta)
            kg.save(Path("data/knowledge_graph.json"))
            st.success(f"Added '{paper_title}' to the knowledge graph.")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to add paper: {exc}")

# ---------------------------------------------------------------------------
# Entity search with type filter
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Entity Search")

if stats.num_nodes == 0:
    st.info("The knowledge graph is empty. Add some papers above to get started.")
else:
    # Filter by entity type
    filter_col, search_col = st.columns([1, 3])
    with filter_col:
        all_types = sorted(stats.node_types.keys())
        type_filter = st.selectbox(
            "Entity type",
            options=["All"] + all_types,
            key="entity_type_filter",
        )
    with search_col:
        entity_query = st.text_input(
            "Search for an entity by name",
            placeholder="e.g., BRCA1, CRISPR, breast cancer",
            key="entity_search",
        )

    if entity_query:
        try:
            result = kg.query_entity(entity_query)
            if not result:
                st.warning(f"No entity found matching '{entity_query}'.")
            else:
                entity = result["entity"]
                if type_filter != "All" and entity.get("entity_type") != type_filter:
                    st.warning(
                        f"Found '{entity['name']}' but it's type '{entity['entity_type']}', "
                        f"not '{type_filter}'."
                    )
                else:
                    st.markdown(f"### {entity['name']}")
                    st.caption(f"Type: **{entity['entity_type']}** | ID: `{entity['id']}`")

                    if entity.get("properties"):
                        st.json(entity["properties"])

                    col_in, col_out = st.columns(2)
                    with col_in:
                        st.markdown("**Incoming connections**")
                        if result["incoming"]:
                            for edge in result["incoming"]:
                                st.text(
                                    f"  {edge['source_name']} --[{edge['rel_type']}]--> "
                                    f"{entity['name']}"
                                )
                        else:
                            st.caption("None")

                    with col_out:
                        st.markdown("**Outgoing connections**")
                        if result["outgoing"]:
                            for edge in result["outgoing"]:
                                st.text(
                                    f"  {entity['name']} --[{edge['rel_type']}]--> "
                                    f"{edge['target_name']}"
                                )
                        else:
                            st.caption("None")
        except Exception as exc:
            st.error(f"Entity search failed: {exc}")

    # Path finding
    st.divider()
    st.subheader("Path Finder")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        path_from = st.text_input("From entity", key="path_from")
    with pcol2:
        path_to = st.text_input("To entity", key="path_to")

    if st.button("Find Path", disabled=not (path_from and path_to)):
        try:
            path = kg.find_path(path_from, path_to)
            if path:
                st.success(" → ".join(path))
            else:
                st.warning("No path found between these entities.")
        except Exception as exc:
            st.error(f"Path finding failed: {exc}")

# ---------------------------------------------------------------------------
# Interactive graph visualization
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Interactive Graph")

if stats.num_nodes == 0:
    st.info("Add papers to the knowledge graph to see the visualization.")
else:
    center = st.text_input(
        "Center on entity ID (leave blank for full graph)",
        key="viz_center",
    )
    depth = st.slider("Neighborhood depth", 1, 4, 2, key="viz_depth")

    if st.button("Render Graph", type="primary"):
        with st.spinner("Generating visualization..."):
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    out_path = str(Path(tmp) / "graph.html")
                    visualize_graph(
                        kg,
                        out_path,
                        center_entity=center if center else None,
                        depth=depth,
                    )
                    html_content = Path(out_path).read_text()

                components.html(html_content, height=650, scrolling=True)
            except Exception as exc:
                st.error(f"Visualization failed: {exc}")
