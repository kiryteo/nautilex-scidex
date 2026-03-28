"""SciDEX — The Scientific Discovery Exchange

Multi-page Streamlit application.
"""

import streamlit as st

st.set_page_config(
    page_title="SciDEX — Scientific Discovery Exchange",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("SciDEX — The Scientific Discovery Exchange")
st.markdown("""
An agentic platform that reads scientific literature, builds a knowledge graph,
generates novel hypotheses through structured debate, and designs experiments to test them.

---

### Navigation

Use the sidebar to navigate between modules:

1. **Literature Explorer** — Search and ingest papers from Semantic Scholar
2. **Knowledge Graph** — Visualize entity relationships *(coming Day 2)*
3. **Hypothesis Workshop** — Generate and refine hypotheses *(coming Day 3-4)*
4. **Experiment Designer** — Design protocols to test hypotheses *(coming Day 5)*

---

### Quick Start

1. Go to **Literature Explorer** in the sidebar
2. Enter a research topic (e.g., "CAR-T cell therapy resistance")
3. Click **Search** to find papers from Semantic Scholar
4. Click **Ingest** to add papers to the local vector store
5. Use **Semantic Search** to find similar papers in your collection
""")

# Show store stats in sidebar
try:
    from scidex.literature.paper_store import PaperStore

    store = PaperStore()
    st.sidebar.metric("Papers in Store", store.count)
except Exception:
    st.sidebar.info("Paper store not yet initialized.")
