"""Literature Explorer — Search, ingest, and explore scientific papers."""

import streamlit as st

from scidex.errors import LiteratureError
from scidex.literature.s2_client import S2Client, Paper
from scidex.literature.paper_store import PaperStore
from scidex.literature.ingestion import ingest_from_query, ingest_citations

st.header("Literature Explorer")


@st.cache_resource
def _get_s2_client():
    return S2Client()


@st.cache_resource
def _get_paper_store():
    return PaperStore()


# Initialize shared state
if "s2_client" not in st.session_state:
    st.session_state.s2_client = _get_s2_client()
if "paper_store" not in st.session_state:
    st.session_state.paper_store = _get_paper_store()
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

client: S2Client = st.session_state.s2_client
store: PaperStore = st.session_state.paper_store

# Sidebar stats
st.sidebar.metric("Papers in Store", store.count)

# -------------------------------------------------------------------
# Paper display helper
# -------------------------------------------------------------------


def render_paper_card(paper: Paper, show_actions: bool = True, key_prefix: str = ""):
    """Render a paper as a styled card with detail expander."""
    with st.container(border=True):
        # Title row
        st.markdown(f"**{paper.title}**")

        # Metadata row
        meta_parts = []
        if paper.authors:
            author_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                author_str += f" +{len(paper.authors) - 3}"
            meta_parts.append(author_str)
        if paper.year:
            meta_parts.append(str(paper.year))
        meta_parts.append(f"Citations: {paper.citation_count}")
        if paper.publication_types:
            meta_parts.append(", ".join(paper.publication_types))

        st.caption(" · ".join(meta_parts))

        # Abstract in expander
        if paper.abstract:
            with st.expander("Abstract", expanded=False):
                st.markdown(paper.abstract)

        # Action buttons
        if show_actions:
            act_cols = st.columns(4)
            with act_cols[0]:
                if st.button("Ingest", key=f"{key_prefix}_ingest_{paper.paper_id}"):
                    try:
                        added = store.add_papers([paper])
                        st.success(f"Added {added} paper to store.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Ingest failed: {exc}")
            with act_cols[1]:
                if st.button("Citations", key=f"{key_prefix}_cite_{paper.paper_id}"):
                    with st.spinner("Fetching citations..."):
                        try:
                            gen = ingest_citations(
                                paper.paper_id, limit=20, client=client, store=store
                            )
                            for status in gen:
                                st.info(status["message"])
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Citation fetch failed: {exc}")
            with act_cols[2]:
                if st.button("Add to KG", key=f"{key_prefix}_kg_{paper.paper_id}"):
                    try:
                        from scidex.knowledge_graph.graph import KnowledgeGraph
                        from pathlib import Path

                        kg_path = Path("data/knowledge_graph.json")
                        if kg_path.exists():
                            kg = KnowledgeGraph.load(kg_path)
                        else:
                            kg = KnowledgeGraph()
                        paper_meta = {
                            "paperId": paper.paper_id,
                            "title": paper.title,
                            "abstract": paper.abstract,
                            "year": paper.year,
                            "citationCount": paper.citation_count,
                            "authors": [{"name": a} for a in paper.authors],
                        }
                        kg.add_paper(paper_meta)
                        kg.save(kg_path)
                        st.session_state.pop("kg", None)  # Force reload
                        st.success("Added to Knowledge Graph.")
                    except Exception as exc:
                        st.error(f"Failed: {exc}")
            with act_cols[3]:
                if paper.url:
                    st.markdown(f"[Open on S2]({paper.url})")


# -------------------------------------------------------------------
# Search section
# -------------------------------------------------------------------

st.subheader("Search Semantic Scholar")

search_col, limit_col = st.columns([4, 1])
with search_col:
    query = st.text_input(
        "Research topic or keywords",
        placeholder="e.g., CAR-T cell therapy resistance mechanisms",
    )
with limit_col:
    limit = st.number_input("Max results", min_value=5, max_value=100, value=20, step=5)

if st.button("Search", type="primary", disabled=not query):
    with st.spinner(f"Searching for '{query}'..."):
        try:
            results = client.search_papers(query, limit=limit)
            st.session_state.search_results = results
            st.session_state.last_query = query
        except Exception as exc:
            st.error(f"Search failed: {exc}")

if st.session_state.search_results:
    st.markdown(f"### Results for: *{st.session_state.last_query}*")
    st.caption(f"Found {len(st.session_state.search_results)} papers")

    # Batch ingest button
    if st.button("Ingest All Results"):
        with st.spinner("Ingesting all papers..."):
            try:
                added = store.add_papers(st.session_state.search_results)
                st.success(f"Ingested {added} papers into store (total: {store.count})")
                st.rerun()
            except Exception as exc:
                st.error(f"Batch ingest failed: {exc}")

    for paper in st.session_state.search_results:
        render_paper_card(paper, key_prefix="search")

# -------------------------------------------------------------------
# Semantic search in local store
# -------------------------------------------------------------------

st.divider()
st.subheader("Semantic Search (Local Store)")

if store.count == 0:
    st.info("No papers in the local store yet. Use the search above to ingest some papers first.")
else:
    local_query = st.text_input(
        "Search your ingested papers",
        placeholder="e.g., tumor microenvironment immune evasion",
        key="local_search",
    )

    n_results = st.slider("Number of results", 1, min(50, store.count), 10)

    if st.button("Search Local Store", disabled=not local_query):
        with st.spinner("Searching local store..."):
            try:
                results = store.search_similar(local_query, n_results=n_results)
                if results:
                    st.markdown(f"### Similar papers in your store ({len(results)} results)")
                    for paper in results:
                        render_paper_card(paper, show_actions=False, key_prefix="local")
                else:
                    st.warning("No similar papers found.")
            except Exception as exc:
                st.error(f"Local search failed: {exc}")
