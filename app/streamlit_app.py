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

    def _get_knowledge():
        """Load fresh accumulator each time (no caching — it reads from disk
        and must reflect new pipeline runs)."""
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

# --- Session History: load previous query log from disk ---
if "query_history" not in st.session_state:
    from pathlib import Path as _QHPath
    import json as _qh_json

    _qh_path = _QHPath("data/query_history.json")
    if _qh_path.exists():
        try:
            st.session_state["query_history"] = _qh_json.loads(_qh_path.read_text())
        except Exception:
            st.session_state["query_history"] = []
    else:
        st.session_state["query_history"] = []

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

        # Import LLM chat function if available
        try:
            from scidex.llm.client import chat as llm_chat

            chat_fn = llm_chat
        except Exception as _llm_err:
            chat_fn = None
            st.warning(
                f"LLM unavailable ({_llm_err}) — experiment design will be skipped. "
                "Set `GITHUB_TOKEN` in `.env` to enable it.",
                icon="⚠️",
            )

        pipeline = SciDEXPipeline(
            knowledge_accumulator=ka,
            on_progress=_on_progress,
            chat_fn=chat_fn,
        )
        result = pipeline.run(
            topic=qs_topic,
            max_papers=qs_papers,
            gde_rounds=qs_rounds,
        )
        st.session_state["pipeline_result"] = result

        # --- Log query to session history ---
        from datetime import datetime as _dt
        from pathlib import Path as _Path
        import json as _json

        _entry = {
            "topic": qs_topic,
            "papers_found": len(result.get("papers", [])),
            "hypotheses": len(result.get("hypotheses", [])),
            "timestamp": _dt.now().isoformat(timespec="seconds"),
        }
        st.session_state.setdefault("query_history", []).append(_entry)
        _qh_save_path = _Path("data/query_history.json")
        _qh_save_path.parent.mkdir(parents=True, exist_ok=True)
        _qh_save_path.write_text(_json.dumps(st.session_state["query_history"], indent=2))

        # --- Ingest pipeline papers into PaperStore so Literature Explorer
        #     and sidebar counts stay in sync with the pipeline. ---
        if result.get("papers"):
            try:
                from scidex.literature.paper_store import PaperStore as _PS

                _store = _PS()
                _added = 0
                for p in result["papers"]:
                    try:
                        _store.add(p)
                        _added += 1
                    except Exception:
                        pass  # skip duplicates / bad data
                if _added:
                    st.toast(f"Added {_added} papers to the literature store.")
            except Exception:
                pass  # PaperStore unavailable — non-fatal

        # --- Bridge pipeline result into the session state keys that the
        #     Hypothesis Workshop (page 3) and Experiment Designer (page 4)
        #     expect, so navigating there after a pipeline run shows results. ---
        try:
            from scidex.hypothesis.models import Hypothesis, HypothesisReport
            from scidex.hypothesis.gde import GDEResult

            # Reconstruct HypothesisReport for the Hypothesis Workshop
            if result.get("hypotheses"):
                hyp_objs = [Hypothesis(**h) for h in result["hypotheses"]]
                st.session_state["hypothesis_report"] = HypothesisReport(
                    topic=qs_topic,
                    hypotheses=hyp_objs,
                    knowledge_gaps=[],
                    summary=f"Generated {len(hyp_objs)} hypotheses for '{qs_topic}'",
                )

            # Reconstruct GDEResult for the Hypothesis Workshop and Experiment Designer
            if result.get("gde_result"):
                gde_obj = GDEResult(**result["gde_result"])
                st.session_state["gde_result"] = gde_obj

                # Persist GDE result to disk so it survives page navigation / restart
                _gde_path = _Path("data/gde_result.json")
                _gde_path.write_text(_json.dumps(result["gde_result"], indent=2, default=str))

            # Persist hypothesis report to disk
            if st.session_state.get("hypothesis_report"):
                _hr = st.session_state["hypothesis_report"]
                _hr_path = _Path("data/hypothesis_report.json")
                _hr_data = {
                    "topic": _hr.topic,
                    "hypotheses": [
                        h.model_dump() if hasattr(h, "model_dump") else h.__dict__
                        for h in _hr.hypotheses
                    ],
                    "knowledge_gaps": _hr.knowledge_gaps,
                    "summary": _hr.summary,
                }
                _hr_path.write_text(_json.dumps(_hr_data, indent=2, default=str))

        except Exception:
            pass  # Non-fatal — pages will just show empty state

        # Persist knowledge graph and refresh session state so all pages
        # (KG Viewer, Hypothesis Workshop) see the new graph immediately.
        if result.get("knowledge_graph"):
            from scidex.knowledge_graph.graph import KnowledgeGraph as _KG

            kg_path = _Path("data/knowledge_graph.json")
            kg_path.parent.mkdir(parents=True, exist_ok=True)
            kg_path.write_text(_json.dumps(result["knowledge_graph"], indent=2, default=str))

            # Force all pages to use the fresh KG — overwrite even if key exists
            st.session_state["kg"] = _KG.load(kg_path)
            # Clear any cached KG visualization so the new graph renders fresh
            st.session_state.pop("kg_html", None)

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

    # Show KG stats if available — gives a richer picture than ChromaDB alone
    if "kg" in st.session_state:
        _kg_stats = st.session_state["kg"].get_statistics()
        st.sidebar.metric("KG Entities", _kg_stats.get("total_entities", 0))
        st.sidebar.metric("KG Relationships", _kg_stats.get("total_relationships", 0))
except Exception:
    st.sidebar.info("Paper store not yet initialized.")

# --- Query History sidebar ---
_qhist = st.session_state.get("query_history", [])
if _qhist:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Query History**")
    for _qh in reversed(_qhist):
        _ts = _qh.get("timestamp", "")[:16].replace("T", " ")
        st.sidebar.caption(
            f"**{_qh['topic']}** — {_qh.get('papers_found', '?')} papers, {_qh.get('hypotheses', '?')} hypotheses  \n{_ts}"
        )
