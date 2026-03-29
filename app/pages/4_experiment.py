"""Experiment Designer — Design protocols to test hypotheses."""

import streamlit as st
import json
from pathlib import Path

from scidex.errors import ExperimentError
from scidex.experiment.designer import ExperimentDesigner
from scidex.experiment.exporter import ProtocolExporter
from scidex.experiment.models import ExperimentProtocol
from scidex.experiment.uniprot_client import UniProtClient
from scidex.experiment.pubmed_client import PubMedClient
from scidex.hypothesis.models import Hypothesis

st.header("Experiment Designer")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "designed_protocols" not in st.session_state:
    st.session_state.designed_protocols = []

if "current_protocol" not in st.session_state:
    st.session_state.current_protocol = None

# ---------------------------------------------------------------------------
# Sidebar — UniProt & PubMed lookup
# ---------------------------------------------------------------------------

st.sidebar.subheader("Protein Lookup (UniProt)")
uniprot_query = st.sidebar.text_input(
    "Search UniProt",
    placeholder="e.g., TP53, BRCA1, insulin",
)
if st.sidebar.button("Search UniProt") and uniprot_query:
    with st.sidebar:
        with st.spinner("Querying UniProt..."):
            try:
                client = UniProtClient()
                results = client.search(uniprot_query, limit=5)
            except Exception as exc:
                results = []
                st.error(f"UniProt query failed: {exc}")
        if results:
            for r in results:
                gene_label = r.get("gene_name") or r.get("accession", "")
                name_label = r.get("name", "")[:50]
                with st.expander(f"{gene_label} — {name_label}"):
                    st.markdown(f"**Accession:** `{r.get('accession', '')}`")
                    st.markdown(f"**Gene:** {r.get('gene_name', 'N/A')}")
                    st.markdown(f"**Organism:** {r.get('organism', 'N/A')}")
                    func = r.get("function", "")
                    if func:
                        st.markdown(f"**Function:** {func[:400]}")
                    st.markdown(f"**Sequence length:** {r.get('sequence_length', 'N/A')} aa")
        elif not results:
            st.caption("No results found.")

st.sidebar.divider()
st.sidebar.subheader("Literature Search (PubMed)")
pubmed_query = st.sidebar.text_input(
    "Search PubMed",
    placeholder="e.g., CRISPR gene therapy cancer",
)
if st.sidebar.button("Search PubMed") and pubmed_query:
    with st.sidebar:
        with st.spinner("Querying PubMed..."):
            try:
                pm_client = PubMedClient()
                papers = pm_client.search_and_fetch(pubmed_query, max_results=5)
            except Exception as exc:
                papers = []
                st.error(f"PubMed query failed: {exc}")
        if papers:
            for p in papers:
                with st.expander(f"{p.get('title', '')[:80]}"):
                    st.markdown(f"**PMID:** `{p.get('pmid', '')}`")
                    journal = p.get("journal", "")
                    year = p.get("year", "")
                    st.markdown(f"**Journal:** {journal}" + (f" ({year})" if year else ""))
                    authors = p.get("authors", [])
                    if authors:
                        author_str = ", ".join(authors[:5])
                        if len(authors) > 5:
                            author_str += f" +{len(authors) - 5}"
                        st.markdown(f"**Authors:** {author_str}")
                    abstract = p.get("abstract", "")
                    if abstract:
                        st.markdown(f"**Abstract:** {abstract[:500]}")
        elif not papers:
            st.caption("No results found.")

st.sidebar.divider()
st.sidebar.subheader("Protocols")
st.sidebar.metric("Designed Protocols", len(st.session_state.designed_protocols))

# ---------------------------------------------------------------------------
# Input — Select or enter hypothesis
# ---------------------------------------------------------------------------

st.subheader("Hypothesis")

input_mode = st.radio(
    "Input method",
    ["From bookmarks", "From GDE results", "Enter manually"],
    horizontal=True,
)

hypothesis: Hypothesis | None = None

if input_mode == "From bookmarks":
    bookmarked = st.session_state.get("bookmarked", [])
    if bookmarked:
        options = {
            f"[{b.get('hypothesis_type', '')}] {b.get('statement', '')[:100]}": b
            for b in bookmarked
        }
        selected = st.selectbox("Select a hypothesis", list(options.keys()))
        if selected:
            bm = options[selected]
            hypothesis = Hypothesis(**bm)
    else:
        st.info(
            "No bookmarked hypotheses. Go to the Hypothesis Workshop to generate "
            "and bookmark some, or enter one manually."
        )
elif input_mode == "From GDE results":
    gde_result = st.session_state.get("gde_result")
    if gde_result is not None and gde_result.final_hypotheses_list:
        options = {
            f"{fh.get('statement', '')[:120]}": fh for fh in gde_result.final_hypotheses_list
        }
        selected = st.selectbox("Select a GDE-refined hypothesis", list(options.keys()))
        if selected:
            fh = options[selected]
            hypothesis = Hypothesis(**fh)
    else:
        st.info("No GDE results available. Run GDE in the Hypothesis Workshop first.")
else:
    manual_statement = st.text_area(
        "Hypothesis statement",
        placeholder="e.g., Inhibition of gene X reduces tumor growth in breast cancer models",
    )
    if manual_statement:
        hypothesis = Hypothesis(
            id="manual-001",
            statement=manual_statement,
            hypothesis_type="manual",
        )

# ---------------------------------------------------------------------------
# Design experiment
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Design Experiment")

col1, col2 = st.columns(2)
with col1:
    domain_context = st.text_area(
        "Domain context (optional)",
        placeholder="e.g., This involves breast cancer cell lines, CRISPR knockout...",
    )
with col2:
    constraints_text = st.text_area(
        "Constraints (optional, JSON)",
        placeholder='e.g., {"budget": "$10,000", "timeline_weeks": 8}',
    )

design_btn = st.button(
    "Design Experiment",
    type="primary",
    disabled=hypothesis is None,
)

if design_btn and hypothesis is not None:
    constraints = None
    if constraints_text.strip():
        try:
            constraints = json.loads(constraints_text)
        except json.JSONDecodeError:
            st.warning("Could not parse constraints as JSON. Ignoring.")

    with st.spinner("Designing experiment protocol..."):
        try:
            from scidex.llm.client import chat as llm_chat

            chat_fn = llm_chat
        except Exception:
            chat_fn = None

        if chat_fn is None:
            st.error("LLM not available. Set GITHUB_TOKEN to enable experiment design.")
        else:
            try:
                designer = ExperimentDesigner(chat_fn=chat_fn)
                protocol = designer.design(
                    hypothesis=hypothesis,
                    domain_context=domain_context,
                    constraints=constraints,
                )
                st.session_state.current_protocol = protocol
                st.session_state.designed_protocols.append(protocol.model_dump())
            except Exception as exc:
                st.error(f"Experiment design failed: {exc}")

# ---------------------------------------------------------------------------
# Display protocol
# ---------------------------------------------------------------------------

protocol: ExperimentProtocol | None = st.session_state.current_protocol

if protocol is not None:
    st.divider()
    st.subheader(protocol.title)

    st.markdown(f"**Objective:** {protocol.objective}")
    st.markdown(f"**Background:** {protocol.background}")
    st.markdown(f"**Timeline:** {protocol.timeline_weeks} weeks")

    # Variables
    with st.expander("Variables", expanded=True):
        if protocol.variables:
            var_data = []
            for v in protocol.variables:
                var_data.append(
                    {
                        "Name": v.name,
                        "Type": v.variable_type.value,
                        "Description": v.description,
                        "Measurement": v.measurement_method,
                        "Units": v.units,
                    }
                )
            st.table(var_data)
        else:
            st.caption("No variables defined.")

    # Controls
    with st.expander("Controls", expanded=True):
        if protocol.controls:
            ctrl_data = []
            for c in protocol.controls:
                ctrl_data.append(
                    {
                        "Name": c.name,
                        "Type": c.control_type,
                        "Description": c.description,
                        "Rationale": c.rationale,
                    }
                )
            st.table(ctrl_data)
        else:
            st.caption("No controls defined.")

    # Statistical Plan
    with st.expander("Statistical Plan", expanded=True):
        sp = protocol.statistical_plan
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Primary Test", sp.primary_test)
        with stat_col2:
            st.metric("Sample Size/Group", sp.sample_size_per_group)
        with stat_col3:
            st.metric("Power", f"{sp.power:.0%}")
        st.markdown(f"**Significance level:** {sp.significance_level}")
        st.markdown(f"**Justification:** {sp.sample_size_justification}")
        if sp.corrections:
            st.markdown(f"**Corrections:** {', '.join(sp.corrections)}")
        if sp.secondary_analyses:
            st.markdown(f"**Secondary analyses:** {', '.join(sp.secondary_analyses)}")

    # Methodology
    with st.expander("Methodology", expanded=False):
        for i, step in enumerate(protocol.methodology, 1):
            st.markdown(f"{i}. {step}")

    # Expected Outcomes & Pitfalls
    with st.expander("Expected Outcomes", expanded=False):
        for outcome in protocol.expected_outcomes:
            st.markdown(f"- {outcome}")

    with st.expander("Potential Pitfalls", expanded=False):
        for pitfall in protocol.potential_pitfalls:
            st.markdown(f"- {pitfall}")

    # Additional details
    with st.expander("Equipment, Reagents & Ethics", expanded=False):
        if protocol.equipment_needed:
            st.markdown("**Equipment:**")
            for eq in protocol.equipment_needed:
                st.markdown(f"- {eq}")
        if protocol.reagents_needed:
            st.markdown("**Reagents:**")
            for rg in protocol.reagents_needed:
                st.markdown(f"- {rg}")
        if protocol.ethical_considerations:
            st.markdown("**Ethical Considerations:**")
            for ec in protocol.ethical_considerations:
                st.markdown(f"- {ec}")

    # ---------------------------------------------------------------------------
    # Refine
    # ---------------------------------------------------------------------------

    st.divider()
    st.subheader("Refine Protocol")

    feedback = st.text_input(
        "Feedback for refinement",
        placeholder="e.g., Increase sample size, add a sham control group",
    )
    refine_btn = st.button("Refine")

    if refine_btn and feedback:
        with st.spinner("Refining protocol..."):
            try:
                from scidex.llm.client import chat as llm_chat

                chat_fn = llm_chat
            except Exception:
                chat_fn = None

            if chat_fn is None:
                st.error("LLM not available.")
            else:
                try:
                    designer = ExperimentDesigner(chat_fn=chat_fn)
                    refined = designer.refine(protocol, feedback)
                    st.session_state.current_protocol = refined
                    # Update the last protocol in the list
                    if st.session_state.designed_protocols:
                        st.session_state.designed_protocols[-1] = refined.model_dump()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Refinement failed: {exc}")

    # ---------------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------------

    st.divider()
    exporter = ProtocolExporter()
    md_content = exporter.to_markdown(protocol)

    st.download_button(
        label="Export as Markdown",
        data=md_content,
        file_name=f"{protocol.title.replace(' ', '_')[:50]}.md",
        mime="text/markdown",
    )
