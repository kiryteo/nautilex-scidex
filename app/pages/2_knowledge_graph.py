"""Knowledge Graph Viewer — Placeholder for Day 2."""

import streamlit as st

st.set_page_config(page_title="Knowledge Graph — SciDEX", layout="wide")
st.header("Knowledge Graph Viewer")
st.info("Coming Day 2: Neo4j-backed knowledge graph with entity extraction and visualization.")
st.markdown("""
### Planned Features
- **Entity extraction**: Genes, proteins, diseases, compounds, methods from paper abstracts
- **Relationship extraction**: ASSOCIATED_WITH, INHIBITS, ACTIVATES, BINDS, etc.
- **Interactive graph visualization** with pyvis
- **Ontology alignment** with OBI, IAO, PROV-O
""")
