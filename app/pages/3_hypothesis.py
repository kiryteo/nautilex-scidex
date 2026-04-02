"""Hypothesis Workshop — Generate, explore, and bookmark scientific hypotheses."""

import streamlit as st
from pathlib import Path
import json

from scidex.knowledge_graph.graph import KnowledgeGraph
from scidex.hypothesis.generator import HypothesisGenerator
from scidex.hypothesis.models import HypothesisReport
from scidex.hypothesis.gde import GenerateDebateEvolve
from scidex.workspace import WorkspaceStore
from scidex.workspace.session import (
    build_comparison_payload,
    build_workspace_snapshot,
    restore_session_from_snapshot,
)

st.header("Hypothesis Workshop")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "kg" not in st.session_state:
    kg_path = Path("data/knowledge_graph.json")
    if kg_path.exists():
        st.session_state.kg = KnowledgeGraph.load(kg_path)
    else:
        st.session_state.kg = KnowledgeGraph()

if "hypothesis_report" not in st.session_state:
    st.session_state.hypothesis_report = None

if "bookmarked" not in st.session_state:
    # Load saved bookmarks
    bm_path = Path("data/bookmarked_hypotheses.json")
    if bm_path.exists():
        st.session_state.bookmarked = json.loads(bm_path.read_text())
    else:
        st.session_state.bookmarked = []

if "active_workspace_name" not in st.session_state:
    st.session_state.active_workspace_name = None

kg: KnowledgeGraph = st.session_state.kg
stats = kg.get_statistics()
workspace_store = WorkspaceStore()

# Module-level label maps (used in both the Hypothesis display and Bookmarks sections)
_TYPE_LABELS = {
    "gap_filling": "Gap Filling",
    "analogy_transfer": "Analogy Transfer",
    "contradiction_resolution": "Contradiction Resolution",
    "combination": "Swanson ABC",
    "extension": "Extension / Evolution",
}

_TYPE_EMOJI = {
    "gap_filling": "🔍",
    "analogy_transfer": "🔄",
    "contradiction_resolution": "⚡",
    "combination": "🔗",
    "extension": "📈",
}

# ---------------------------------------------------------------------------
# Sidebar — Graph info + knowledge stats
# ---------------------------------------------------------------------------

st.sidebar.subheader("Knowledge Graph")
sb1, sb2 = st.sidebar.columns(2)
sb1.metric("Nodes", stats.num_nodes)
sb2.metric("Edges", stats.num_edges)

# Knowledge accumulator stats
try:
    from scidex.knowledge.accumulator import KnowledgeAccumulator

    @st.cache_resource
    def _get_knowledge():
        return KnowledgeAccumulator()

    _ka = _get_knowledge()
    _snap = _ka.get_snapshot()
    st.sidebar.divider()
    st.sidebar.subheader("Knowledge Base")
    kb1, kb2 = st.sidebar.columns(2)
    kb1.metric("Papers", _snap.total_papers)
    kb2.metric("Hypotheses", _snap.total_hypotheses)
except Exception:
    st.sidebar.caption("Knowledge base unavailable for this session.")

if stats.num_nodes == 0:
    st.warning("The knowledge graph is empty. Go to **Knowledge Graph** page to add papers first.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("Bookmarks")
st.sidebar.metric("Saved Hypotheses", len(st.session_state.bookmarked))

st.sidebar.divider()
st.sidebar.subheader("Workspaces")
available_workspaces = workspace_store.list_workspaces()
workspace_names = [workspace.name for workspace in available_workspaces]
if workspace_names:
    selected_workspace = st.sidebar.selectbox(
        "Load saved workspace",
        options=workspace_names,
        index=0,
        key="selected_workspace_name",
    )
    if st.sidebar.button("Load Workspace"):
        snapshot = workspace_store.load(selected_workspace)
        if snapshot is not None:
            restored = restore_session_from_snapshot(snapshot)
            for key, value in restored.items():
                st.session_state[key] = value
            st.rerun()
else:
    st.sidebar.caption("No saved workspaces yet.")

workspace_name = st.sidebar.text_input(
    "Save current workspace as",
    value=st.session_state.active_workspace_name or "",
)
if st.sidebar.button("Save Workspace", disabled=not workspace_name.strip()):
    try:
        snapshot = build_workspace_snapshot(workspace_name.strip(), st.session_state)
        workspace_store.save(snapshot)
    except OSError as exc:
        st.sidebar.error(f"Failed to save workspace '{workspace_name.strip()}': {exc}")
    else:
        st.session_state.active_workspace_name = workspace_name.strip()
        st.sidebar.success(f"Saved workspace '{workspace_name.strip()}'")

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

st.subheader("Generate Hypotheses")

col1, col2 = st.columns(2)
with col1:
    topic = st.text_input(
        "Topic / research question",
        placeholder="e.g., gene therapy for breast cancer",
        key="hyp_topic",
    )
    source_entity = st.text_input(
        "Source entity for Swanson linking (optional)",
        placeholder="e.g., CRISPR, fish oil",
        key="hyp_source_entity",
    )
with col2:
    source_method = st.text_input(
        "Method for analogy search (optional)",
        placeholder="e.g., CRISPR, RNA-seq",
        key="hyp_source_method",
    )

    # Strategy hints
    st.caption(
        "**Strategies used:** Gap detection, Swanson ABC linking, "
        "Analogy transfer, Contradiction mining"
    )

generate_btn = st.button("Generate Hypotheses", type="primary", disabled=not topic)

if generate_btn:
    with st.spinner("Running hypothesis generation pipeline..."):
        try:
            # Use LLM when available for richer hypothesis generation
            try:
                from scidex.llm.client import chat as _llm_chat

                _gen_chat_fn = _llm_chat
            except Exception:
                _gen_chat_fn = None

            generator = HypothesisGenerator(chat_fn=_gen_chat_fn)
            report = generator.generate(
                kg,
                topic=topic,
                source_entity=source_entity or None,
                source_method=source_method or None,
            )
            st.session_state.hypothesis_report = report
        except Exception as exc:
            st.error(f"Hypothesis generation failed: {exc}")

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

report: HypothesisReport | None = st.session_state.hypothesis_report

if report is not None:
    st.divider()

    # Summary
    st.subheader(f"Results: {report.topic}")
    st.info(report.summary)
    if st.session_state.active_workspace_name:
        st.caption(f"Active workspace: {st.session_state.active_workspace_name}")

    # Knowledge gap analysis
    with st.expander("Knowledge Gap Analysis", expanded=True):
        if report.knowledge_gaps:
            for gap in report.knowledge_gaps:
                st.markdown(f"- {gap}")
        else:
            st.caption("No significant gaps detected.")

    # Hypotheses
    st.subheader(f"Hypotheses ({len(report.hypotheses)})")

    # Type filter
    all_types = sorted({h.hypothesis_type for h in report.hypotheses})
    selected_types = st.multiselect(
        "Filter by type",
        options=all_types,
        default=all_types,
        format_func=lambda t: f"{_TYPE_EMOJI.get(t, '💡')} {_TYPE_LABELS.get(t, t)}",
    )

    # Sort option
    sort_by = st.selectbox(
        "Sort by",
        options=["Composite Score", "Confidence", "Novelty", "Testability"],
        index=0,
    )
    sort_key = {
        "Composite Score": lambda h: h.composite_score,
        "Confidence": lambda h: h.confidence,
        "Novelty": lambda h: h.novelty_score,
        "Testability": lambda h: h.testability_score,
    }[sort_by]

    filtered = [h for h in report.hypotheses if h.hypothesis_type in selected_types]
    filtered.sort(key=sort_key, reverse=True)

    comparison_ids = st.multiselect(
        "Compare hypotheses",
        options=[h.id for h in filtered],
        default=[h.id for h in filtered[: min(2, len(filtered))]],
        format_func=lambda hypothesis_id: next(
            (
                hypothesis.statement[:100]
                for hypothesis in filtered
                if hypothesis.id == hypothesis_id
            ),
            hypothesis_id,
        ),
    )

    if comparison_ids:
        comparison_payload = build_comparison_payload(filtered, comparison_ids)
        st.markdown("### Hypothesis Comparison")
        st.table(
            [
                {
                    "Statement": item["statement"][:100],
                    "Composite": f"{item['composite_score']:.2f}",
                    "Confidence": f"{item['confidence']:.2f}",
                    "Novelty": f"{item['novelty_score']:.2f}",
                    "Testability": f"{item['testability_score']:.2f}",
                    "Support": item["evidence_summary"].get("support_count", 0),
                    "Papers": item["evidence_summary"].get("source_paper_count", 0),
                }
                for item in comparison_payload
            ]
        )
        comparison_columns = st.columns(len(comparison_payload))
        for column, item in zip(comparison_columns, comparison_payload):
            with column:
                st.markdown(f"**{item['statement']}**")
                st.metric("Composite Score", f"{item['composite_score']:.2f}")
                st.markdown("**Supporting Evidence**")
                for evidence in item["evidence_sections"]["supporting_evidence"]:
                    st.markdown(f"- {evidence}")
                if item["evidence_sections"]["source_papers"]:
                    st.markdown("**Source Papers**")
                    for paper in item["evidence_sections"]["source_papers"]:
                        st.markdown(f"- {paper}")

    for i, h in enumerate(filtered):
        emoji = _TYPE_EMOJI.get(h.hypothesis_type, "💡")
        label = _TYPE_LABELS.get(h.hypothesis_type, h.hypothesis_type)

        with st.expander(
            f"{emoji} [{label}] {h.statement[:120]}",
            expanded=i < 3,
        ):
            st.markdown(f"**Statement:** {h.statement}")

            score_cols = st.columns(3)
            with score_cols[0]:
                st.metric("Confidence", f"{h.confidence:.2f}")
            with score_cols[1]:
                st.metric("Novelty", f"{h.novelty_score:.2f}")
            with score_cols[2]:
                st.metric("Testability", f"{h.testability_score:.2f}")
            st.metric("Composite", f"{h.composite_score:.2f}")

            if h.supporting_evidence:
                st.markdown("**Supporting Evidence:**")
                for ev in h.supporting_evidence:
                    st.markdown(f"- {ev}")

            if h.source_papers:
                st.markdown("**Source Papers:** " + ", ".join(h.source_papers))

            if h.evidence_summary:
                st.markdown("**Evidence Summary:**")
                st.markdown(
                    f"- Support items: {h.evidence_summary.get('support_count', 0)}\n"
                    f"- Source papers: {h.evidence_summary.get('source_paper_count', 0)}"
                )

            # Bookmark button
            bookmarked_ids = {b["id"] for b in st.session_state.bookmarked}
            if h.id in bookmarked_ids:
                st.success("Bookmarked")
            else:
                if st.button("Bookmark", key=f"bm_{h.id}"):
                    st.session_state.bookmarked.append(h.model_dump())
                    bm_path = Path("data/bookmarked_hypotheses.json")
                    bm_path.parent.mkdir(parents=True, exist_ok=True)
                    bm_path.write_text(json.dumps(st.session_state.bookmarked, indent=2))
                    st.rerun()

    # Export as markdown
    if filtered:
        st.divider()
        md_lines = [f"# Hypotheses: {report.topic}\n"]
        for h in filtered:
            label = _TYPE_LABELS.get(h.hypothesis_type, h.hypothesis_type)
            md_lines.append(f"## [{label}] {h.statement[:80]}\n")
            md_lines.append(f"**Statement:** {h.statement}\n")
            md_lines.append(
                f"Confidence: {h.confidence:.2f} | "
                f"Novelty: {h.novelty_score:.2f} | "
                f"Testability: {h.testability_score:.2f}\n"
            )
            if h.supporting_evidence:
                md_lines.append("**Evidence:**\n")
                for ev in h.supporting_evidence:
                    md_lines.append(f"- {ev}\n")
            md_lines.append("---\n")

        st.download_button(
            "Export Hypotheses as Markdown",
            data="\n".join(md_lines),
            file_name=f"hypotheses_{report.topic.replace(' ', '_')[:30]}.md",
            mime="text/markdown",
        )

    # Swanson ABC chains visualization
    st.divider()
    st.subheader("Swanson ABC Chains")

    swanson_hypotheses = [h for h in report.hypotheses if h.hypothesis_type == "combination"]
    if swanson_hypotheses:
        for h in swanson_hypotheses:
            # Parse the chain from the statement
            st.markdown(f"**{h.statement}**")
            if h.supporting_evidence:
                cols = st.columns(len(h.supporting_evidence))
                for j, ev in enumerate(h.supporting_evidence):
                    with cols[j]:
                        st.caption(ev)
            st.markdown("---")
    else:
        st.caption(
            "No Swanson ABC chains found. Try specifying a source entity in the input above."
        )

# ---------------------------------------------------------------------------
# Generate-Debate-Evolve (GDE)
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Generate-Debate-Evolve (GDE)")
st.caption(
    "Adversarial refinement: critics debate hypotheses across rounds, "
    "survivors evolve into stronger versions."
)

if "gde_result" not in st.session_state:
    st.session_state.gde_result = None

gde_col1, gde_col2 = st.columns(2)
with gde_col1:
    gde_rounds = st.slider("Number of rounds", min_value=1, max_value=5, value=2)
with gde_col2:
    gde_survivors = st.slider("Survivors per round", min_value=1, max_value=4, value=2)

# Decide if we have hypotheses to feed in
has_report = report is not None and len(report.hypotheses) > 0

gde_btn = st.button(
    "Run Generate-Debate-Evolve",
    type="primary",
    disabled=not topic,
    help="Requires a topic. Uses previously generated hypotheses if available.",
)

if gde_btn:
    progress_messages: list[str] = []

    with st.status("Running GDE...", expanded=True) as status:

        def _on_progress(msg: str) -> None:
            progress_messages.append(msg)
            st.write(msg)

        try:
            from scidex.llm.client import chat as llm_chat

            chat_fn = llm_chat
        except Exception:
            chat_fn = None

        try:
            gde = GenerateDebateEvolve(
                max_rounds=gde_rounds,
                survivors_per_round=gde_survivors,
                chat_fn=chat_fn,
                on_progress=_on_progress,
            )

            initial = report.hypotheses if has_report else None
            gde_result = gde.run(
                knowledge_graph=kg,
                topic=topic,
                initial_hypotheses=initial,
            )
            st.session_state.gde_result = gde_result
            status.update(
                label=f"GDE complete — {gde_result.total_rounds} rounds, "
                f"{gde_result.final_hypotheses} final hypotheses",
                state="complete",
            )
        except Exception as exc:
            status.update(label=f"GDE failed: {exc}", state="error")
            st.error(f"GDE error: {exc}")

# Display GDE results
gde_result = st.session_state.gde_result
if gde_result is not None and gde_result.total_rounds > 0:
    st.markdown(
        f"**{gde_result.total_rounds} rounds** | "
        f"**{gde_result.initial_hypotheses}** initial → "
        f"**{gde_result.final_hypotheses}** final hypotheses"
    )

    # Best hypothesis
    if gde_result.best_hypothesis:
        st.success(f"**Best hypothesis:** {gde_result.best_hypothesis.get('statement', '')}")

    # Final hypotheses
    st.markdown("#### Final Hypotheses")
    for fh in gde_result.final_hypotheses_list:
        with st.expander(f"{fh.get('statement', '')[:120]}"):
            st.markdown(f"**Statement:** {fh.get('statement', '')}")
            if fh.get("supporting_evidence"):
                st.markdown("**Evidence:**")
                for ev in fh["supporting_evidence"]:
                    st.markdown(f"- {ev}")

    # Round-by-round details
    for rnd in gde_result.rounds:
        with st.expander(f"Round {rnd.round_number} details"):
            rc1, rc2 = st.columns(2)
            rc1.metric("Hypotheses In", rnd.hypotheses_in)
            rc2.metric("Survivors", rnd.hypotheses_out)

            st.markdown("**Survivors:**")
            for s in rnd.survivors:
                st.markdown(f"- {s.get('statement', '')[:100]}")

            if rnd.eliminated:
                st.markdown("**Eliminated:**")
                for e in rnd.eliminated:
                    st.markdown(f"- {e.get('statement', '')[:100]}")

            # Critic feedback
            if rnd.critic_scores:
                st.markdown("**Critic Scores:**")
                for hid, scores in rnd.critic_scores.items():
                    for cs in scores:
                        st.caption(
                            f"[{cs.get('critic_id', '')}] "
                            f"overall={cs.get('overall_score', 0):.2f} | "
                            f"novelty={cs.get('novelty', 0):.2f} | "
                            f"feasibility={cs.get('feasibility', 0):.2f}"
                        )
                        if cs.get("weaknesses"):
                            st.markdown("  Weaknesses: " + "; ".join(cs["weaknesses"][:3]))

            if rnd.evolved_hypotheses:
                st.markdown("**Evolved hypotheses:**")
                for eh in rnd.evolved_hypotheses:
                    st.markdown(f"- {eh.get('statement', '')[:100]}")

# ---------------------------------------------------------------------------
# Saved bookmarks
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Saved Hypotheses")

if st.session_state.bookmarked:
    for bm in st.session_state.bookmarked:
        label = _TYPE_LABELS.get(bm.get("hypothesis_type", ""), bm.get("hypothesis_type", ""))
        st.markdown(f"- **[{label}]** {bm.get('statement', '')}")
else:
    st.caption("No bookmarked hypotheses yet.")
