"""SciDEX — The Scientific Discovery Exchange

Multi-page Streamlit application.
"""

import streamlit as st
from pathlib import Path

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
""")

# ---------------------------------------------------------------------------
# Knowledge accumulator stats
# ---------------------------------------------------------------------------

try:
    from scidex.knowledge.accumulator import KnowledgeAccumulator

    @st.cache_resource
    def _get_knowledge():
        return KnowledgeAccumulator()

    ka = _get_knowledge()
    snap = ka.get_snapshot()

    st.divider()
    st.subheader("Knowledge Base")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Papers", snap.total_papers)
    m2.metric("Entities", snap.total_entities)
    m3.metric("Hypotheses", snap.total_hypotheses)
    m4.metric("Experiments", snap.total_experiments)

except Exception:
    snap = None
    ka = None

# ---------------------------------------------------------------------------
# Quick Start — full pipeline
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Quick Start — Full Pipeline")
st.caption(
    "Enter a research topic to run the complete pipeline: "
    "Literature Search → Knowledge Graph → Hypothesis Generation → GDE → Experiment Design"
)

qs_topic = st.text_input(
    "Research topic",
    placeholder="e.g., CAR-T cell therapy resistance mechanisms",
    key="qs_topic",
)

qs_col1, qs_col2 = st.columns(2)
with qs_col1:
    qs_papers = st.number_input("Max papers", min_value=5, max_value=100, value=20, step=5)
with qs_col2:
    qs_rounds = st.slider("GDE rounds", min_value=1, max_value=5, value=2)

if st.button("Run Pipeline", type="primary", disabled=not qs_topic):
    from scidex.errors import SciDEXError

    progress_container = st.status("Running SciDEX pipeline...", expanded=True)

    try:
        from scidex.pipeline import SciDEXPipeline

        def _on_progress(msg: str) -> None:
            progress_container.write(msg)

        pipeline = SciDEXPipeline(
            knowledge_accumulator=ka,
            on_progress=_on_progress,
        )
        result = pipeline.run(
            topic=qs_topic,
            max_papers=qs_papers,
            gde_rounds=qs_rounds,
        )
        st.session_state["pipeline_result"] = result

        # Summary
        n_papers = len(result.get("papers", []))
        n_hyp = len(result.get("hypotheses", []))
        n_exp = len(result.get("experiments", []))
        errors = result.get("errors", [])

        progress_container.update(
            label=f"Pipeline complete — {n_papers} papers, {n_hyp} hypotheses, {n_exp} experiments",
            state="complete",
        )

        if errors:
            for err in errors:
                st.warning(err)

        # Show results summary
        st.markdown(f"### Results for: *{qs_topic}*")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Papers Found", n_papers)
        rc2.metric("Hypotheses Generated", n_hyp)
        rc3.metric("Experiments Designed", n_exp)

        # Top hypotheses
        if result.get("hypotheses"):
            st.markdown("#### Top Hypotheses")
            for i, h in enumerate(result["hypotheses"][:5]):
                with st.expander(
                    f"[{h.get('hypothesis_type', '')}] {h.get('statement', '')[:120]}",
                    expanded=i == 0,
                ):
                    st.markdown(h.get("statement", ""))
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Confidence", f"{h.get('confidence', 0):.2f}")
                    sc2.metric("Novelty", f"{h.get('novelty_score', 0):.2f}")
                    sc3.metric("Testability", f"{h.get('testability_score', 0):.2f}")

        # GDE result
        gde = result.get("gde_result")
        if gde and gde.get("best_hypothesis"):
            st.markdown("#### Best Hypothesis (after GDE)")
            st.success(gde["best_hypothesis"].get("statement", ""))

    except Exception as exc:
        progress_container.update(label=f"Pipeline failed: {exc}", state="error")
        st.error(f"Pipeline error: {exc}")

# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------

if ka is not None:
    sessions = ka.get_session_history()
    if sessions:
        st.divider()
        st.subheader("Session History")
        for sess in reversed(sessions[-10:]):
            status = "completed" if sess.get("ended_at") else "in progress"
            st.markdown(
                f"- **{sess.get('topic', 'Unknown')}** — {status}"
                + (f" — {sess.get('summary', '')}" if sess.get("summary") else "")
            )

# ---------------------------------------------------------------------------
# Navigation guide
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Navigation")
st.markdown("""
Use the sidebar to navigate between modules:

1. **Literature Explorer** — Search and ingest papers from Semantic Scholar
2. **Knowledge Graph** — Visualize entity relationships
3. **Hypothesis Workshop** — Generate and refine hypotheses via GDE
4. **Experiment Designer** — Design protocols to test hypotheses
""")

# Sidebar stats
try:
    from scidex.literature.paper_store import PaperStore

    @st.cache_resource
    def _get_paper_store():
        return PaperStore()

    store = _get_paper_store()
    st.sidebar.metric("Papers in Store", store.count)
except Exception:
    st.sidebar.info("Paper store not yet initialized.")
