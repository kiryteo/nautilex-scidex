"""Literature Explorer — Search, ingest, and explore scientific papers."""

import streamlit as st
from scidex.literature.s2_client import S2Client, Paper
from scidex.literature.paper_store import PaperStore
from scidex.literature.ingestion import ingest_from_query, ingest_citations

st.set_page_config(page_title="Literature Explorer — SciDEX", layout="wide")
st.header("Literature Explorer")

# Initialize shared state
if "s2_client" not in st.session_state:
    st.session_state.s2_client = S2Client()
if "paper_store" not in st.session_state:
    st.session_state.paper_store = PaperStore()
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
    """Render a paper as a styled card."""
    with st.container(border=True):
        col_main, col_meta = st.columns([3, 1])

        with col_main:
            st.markdown(f"**{paper.title}**")
            if paper.abstract:
                snippet = paper.abstract[:300] + ("..." if len(paper.abstract) > 300 else "")
                st.caption(snippet)

        with col_meta:
            if paper.year:
                st.markdown(f"**Year:** {paper.year}")
            st.markdown(f"**Citations:** {paper.citation_count}")
            if paper.authors:
                author_str = ", ".join(paper.authors[:3])
                if len(paper.authors) > 3:
                    author_str += f" + {len(paper.authors) - 3} more"
                st.markdown(f"**Authors:** {author_str}")
            if paper.url:
                st.markdown(f"[Open on S2]({paper.url})")

        if show_actions:
            act_cols = st.columns(3)
            with act_cols[0]:
                if st.button("Ingest", key=f"{key_prefix}_ingest_{paper.paper_id}"):
                    added = store.add_papers([paper])
                    st.success(f"Added {added} paper to store.")
                    st.rerun()
            with act_cols[1]:
                if st.button("Explore Citations", key=f"{key_prefix}_cite_{paper.paper_id}"):
                    with st.spinner("Fetching citations..."):
                        gen = ingest_citations(paper.paper_id, limit=20, client=client, store=store)
                        for status in gen:
                            st.info(status["message"])
                    st.rerun()
            with act_cols[2]:
                if paper.publication_types:
                    st.caption(", ".join(paper.publication_types))


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
        results = client.search_papers(query, limit=limit)
        st.session_state.search_results = results
        st.session_state.last_query = query

if st.session_state.search_results:
    st.markdown(f"### Results for: *{st.session_state.last_query}*")
    st.caption(f"Found {len(st.session_state.search_results)} papers")

    # Batch ingest button
    if st.button("Ingest All Results"):
        with st.spinner("Ingesting all papers..."):
            added = store.add_papers(st.session_state.search_results)
        st.success(f"Ingested {added} papers into store (total: {store.count})")
        st.rerun()

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
            results = store.search_similar(local_query, n_results=n_results)

        if results:
            st.markdown(f"### Similar papers in your store ({len(results)} results)")
            for paper in results:
                render_paper_card(paper, show_actions=False, key_prefix="local")
        else:
            st.warning("No similar papers found.")
