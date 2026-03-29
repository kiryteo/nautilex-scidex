# SciDEX: The Scientific Discovery Exchange

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-249%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

**An agentic platform that transforms scientific literature into novel, testable hypotheses through structured adversarial debate.**

SciDEX ingests papers from Semantic Scholar, builds a knowledge graph of entities and relationships, generates hypotheses using four complementary strategies, refines them through a multi-round Generate-Debate-Evolve loop, and produces rigorous experiment protocols -- all from a single research query.

---

## Architecture

```
                           SciDEX Pipeline
 ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐
 │  1. INGEST   │    │  2. EXTRACT  │    │  3. HYPOTHESIZE          │
 │              │    │              │    │                          │
 │ Semantic     │───>│ Knowledge    │───>│ Gap Detection            │
 │ Scholar API  │    │ Graph        │    │ Swanson ABC Linking      │
 │ + SPECTER2   │    │ (NetworkX)   │    │ Cross-Domain Analogy     │
 │ + ChromaDB   │    │ Entity &     │    │ Contradiction Mining     │
 │              │    │ Relation     │    │                          │
 │ 214M papers  │    │ Extraction   │    │         │                │
 └──────────────┘    └──────────────┘    └─────────┼────────────────┘
                                                   │
                                                   v
                                    ┌──────────────────────────┐
                                    │  4. GENERATE-DEBATE-     │
                                    │     EVOLVE (GDE)         │
                                    │                          │
                                    │  Generate hypotheses     │
                                    │       │                  │
                                    │  3 Critics debate        │
                                    │  (method/domain/innov.)  │
                                    │       │                  │
                                    │  Tournament selection    │
                                    │       │                  │
                                    │  Evolve survivors        │
                                    │       │                  │
                                    │  Repeat N rounds         │
                                    └───────┼──────────────────┘
                                            │
                                            v
                           ┌──────────────────────────────┐
                           │  5. EXPERIMENT DESIGN         │
                           │                               │
                           │  Structured protocols          │
                           │  Variables, controls, stats    │
                           │  UniProt + PubMed integration  │
                           │  Markdown + PDF export         │
                           └───────┬──────────────────────-┘
                                   │
                                   v
                           ┌──────────────────────────────┐
                           │  6. KNOWLEDGE ACCUMULATION    │
                           │                               │
                           │  4-layer persistent store      │
                           │  Raw > Domain > Outcomes > Meta│
                           │  Cross-session learning        │
                           └───────────────────────────────┘
```

---

## Features

- **Semantic Scholar integration** -- search 214M papers with pre-computed SPECTER2 embeddings, rate-limited API client with local file caching
- **ChromaDB vector store** -- semantic similarity search over ingested papers using SPECTER2 or fallback embeddings
- **Knowledge graph extraction** -- regex-based entity recognition (genes, proteins, diseases, methods) with co-occurrence relationship building on NetworkX
- **Swanson ABC linking** -- discover hidden indirect connections: if A relates to B and B relates to C, does A relate to C?
- **Four hypothesis strategies** -- gap detection, cross-domain analogy, Swanson linking, and contradiction mining
- **Generate-Debate-Evolve (GDE)** -- multi-round adversarial refinement with three critic perspectives, tournament selection, and hypothesis evolution
- **Experiment design** -- LLM-generated structured protocols with variables, controls, sample size justification, and statistical analysis plans
- **UniProt and PubMed integration** -- ground hypotheses in real protein data and full-text literature
- **Protocol export** -- Markdown and PDF (via WeasyPrint) export of experiment protocols
- **Knowledge accumulation** -- 4-layer persistent store (raw, domain, outcomes, meta) that learns across sessions
- **Multi-page Streamlit app** -- Literature Explorer, Knowledge Graph Viewer, Hypothesis Workshop, and Experiment Designer

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A [GitHub token](https://github.com/settings/tokens) for LLM access (GitHub Copilot Models)
- A [Semantic Scholar API key](https://www.semanticscholar.org/product/api) (optional but recommended)

### Install

```bash
git clone <repo-url> && cd scidex
uv sync
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# Required -- GitHub token for LLM access (GitHub Copilot Models via OpenAI-compatible API)
GITHUB_TOKEN=your_github_token_here
# Or dynamically:
# export GITHUB_TOKEN=$(gh auth token)

# Optional -- Semantic Scholar API key (higher rate limits: 100 req/sec vs 1 req/sec)
S2_API_KEY=your_s2_api_key_here
```

### Launch

```bash
uv run streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501` with four pages: Literature Explorer, Knowledge Graph, Hypothesis Workshop, and Experiment Designer.

To run the full pipeline programmatically:

```python
from scidex.pipeline import SciDEXPipeline

pipeline = SciDEXPipeline()
result = pipeline.run(topic="CAR-T cell therapy resistance mechanisms", max_papers=20, gde_rounds=3)
```

---

## Project Structure

```
src/scidex/
├── pipeline.py                  # End-to-end orchestrator (search → KG → hypotheses → GDE → experiments)
├── errors.py                    # Domain-specific exception hierarchy
├── llm/
│   ├── client.py                # GitHub Copilot Models client (OpenAI-compatible, retry logic)
│   └── llm_config.py            # Model configuration
├── literature/
│   ├── s2_client.py             # Semantic Scholar API client (rate-limited, file-cached)
│   ├── paper_store.py           # ChromaDB vector store for paper embeddings
│   └── ingestion.py             # Search → fetch → store pipeline with progress tracking
├── knowledge_graph/
│   ├── extractor.py             # Regex-based entity extraction (genes, proteins, diseases, methods)
│   ├── relations.py             # Relationship builder (structural + co-occurrence)
│   ├── graph.py                 # NetworkX DiGraph with query, path-finding, serialization
│   ├── visualization.py         # Graph rendering for Streamlit
│   └── models.py                # Entity, Relationship, KnowledgeGraphStats models
├── hypothesis/
│   ├── generator.py             # Orchestrates all 4 hypothesis strategies
│   ├── gap_detector.py          # Find structural holes in the knowledge graph
│   ├── swanson_linker.py        # Swanson ABC literature-based discovery
│   ├── analogy_engine.py        # Cross-domain method transfer
│   ├── contradictions.py        # Contradiction mining across papers
│   ├── critic.py                # LLM-powered multi-perspective hypothesis evaluation
│   ├── tournament.py            # Tournament selection (top-N survival)
│   ├── evolver.py               # Combine survivors + critic feedback → improved hypotheses
│   ├── gde.py                   # Generate-Debate-Evolve loop orchestrator
│   └── models.py                # Hypothesis, HypothesisReport, GapAnalysis models
├── experiment/
│   ├── designer.py              # LLM-generated structured experiment protocols
│   ├── exporter.py              # Markdown + PDF export (WeasyPrint)
│   ├── uniprot_client.py        # UniProt protein lookup API
│   ├── pubmed_client.py         # PubMed literature search via NCBI E-utilities
│   └── models.py                # ExperimentProtocol, Variable, Control, StatisticalPlan
└── knowledge/
    ├── accumulator.py           # 4-layer persistent knowledge store with session tracking
    └── models.py                # KnowledgeEntry, KnowledgeLayer, KnowledgeSnapshot

app/
├── streamlit_app.py             # Main page: dashboard, quick-start pipeline, session history
└── pages/
    ├── 1_literature.py          # Literature Explorer: search, ingest, browse papers
    ├── 2_knowledge_graph.py     # Knowledge Graph Viewer: entity exploration, visualization
    ├── 3_hypothesis.py          # Hypothesis Workshop: generate, GDE, bookmark hypotheses
    └── 4_experiment.py          # Experiment Designer: design protocols, export PDF
```

---

## The Generate-Debate-Evolve (GDE) Pipeline

GDE is the core innovation, inspired by [Google's AI Co-Scientist](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/). It applies adversarial refinement to transform initial hypotheses into rigorously vetted, high-quality scientific claims.

### How It Works

```
Round 0: GENERATE
  HypothesisGenerator produces initial hypotheses from 4 strategies:
  - Gap detection (structural holes in the knowledge graph)
  - Swanson ABC linking (hidden indirect connections)
  - Cross-domain analogy (method/concept transfer)
  - Contradiction mining (conflicting findings → resolution hypotheses)

Round 1..N: DEBATE → SELECT → EVOLVE

  DEBATE
  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
  │ Methodologist   │  │ Domain Expert   │  │ Innovator       │
  │                 │  │                 │  │                 │
  │ Experimental    │  │ Mechanism       │  │ Breakthrough    │
  │ rigor, stats,   │  │ plausibility,   │  │ potential, new  │
  │ confounds       │  │ prior work      │  │ directions      │
  └───────┬─────────┘  └───────┬─────────┘  └───────┬─────────┘
          │                    │                     │
          └────────────────────┼─────────────────────┘
                               │
                    Each scores: novelty, feasibility,
                    testability, significance (0-1)
                    + strengths, weaknesses, improvements
                               │
                               v
  SELECT (Tournament)
  - Aggregate critic scores across all perspectives
  - Top N hypotheses survive; rest are eliminated
                               │
                               v
  EVOLVE
  - Combine surviving hypotheses with critic feedback
  - Generate refined, improved variants
  - Feed back into next debate round
```

After the final round, the hypothesis with the highest aggregate critic score is selected as the best candidate for experiment design.

---

## Module Deep Dives

### Literature Ingestion

The `S2Client` wraps the Semantic Scholar Graph API (214M papers, 2.49B citations). It enforces rate limiting via a thread-safe `_RateLimitedSession` (1 request/second without API key) and caches individual paper responses as JSON files in `data/cache/` to avoid redundant fetches. Papers include pre-computed SPECTER2 embeddings from the S2 API, eliminating the need for local embedding computation.

`PaperStore` persists papers in a ChromaDB collection with cosine similarity. When SPECTER2 embeddings are available for all papers in a batch, they are used directly; otherwise ChromaDB computes embeddings from abstract text.

### Knowledge Graph

`EntityExtractor` uses curated regex patterns -- not LLM calls -- for fast entity recognition. It detects gene names (2-6 uppercase letters, excluding common English abbreviations), proteins (suffix-based: -ase, -in, -gen, etc.), 60+ diseases, and 40+ scientific methods. `RelationshipBuilder` creates structural relationships (author-wrote-paper, paper-cites-paper) and co-occurrence relationships from abstract text.

The `KnowledgeGraph` class wraps a NetworkX DiGraph with convenience methods for BFS subgraph extraction, shortest-path finding, and JSON serialization.

### Swanson ABC Linking

Implements Don Swanson's literature-based discovery model. Given a starting entity A, it finds co-occurring entities B, then for each B finds co-occurring entities C that are *not* directly connected to A. These A-B-C chains suggest novel indirect relationships. Confidence scales with the number of independent intermediaries, and novelty scales with graph distance.

### Hypothesis Critics

Three LLM-powered critics evaluate each hypothesis from distinct perspectives:

| Critic | Focus | Evaluates |
|--------|-------|-----------|
| **Methodologist** | Experimental rigor | Statistical power, confounds, testability, vague claims |
| **Domain Expert** | Scientific validity | Mechanism plausibility, consistency with prior work, novelty vs. prior art |
| **Innovator** | Breakthrough potential | New research directions, paradigm challenges, transformative impact |

Each critic returns structured scores (novelty, feasibility, testability, significance -- all 0-1) plus written strengths, weaknesses, and suggested improvements. This multi-perspective approach is validated by research showing LLM feedback overlaps with human reviewers more than humans overlap with each other (Liang et al., 2023).

### Experiment Design

`ExperimentDesigner` generates structured protocols via LLM with:
- Independent, dependent, controlled, and confounding variables
- Positive, negative, vehicle, and sham controls
- Statistical plan: primary test, significance level, power analysis, sample size justification, multiple comparison corrections
- Step-by-step methodology, expected outcomes, and potential pitfalls
- Equipment, reagents, timeline, and ethical considerations

`ProtocolExporter` renders protocols as Markdown or PDF (via WeasyPrint). `UniProtClient` and `PubMedClient` provide real-time lookups to ground experiment design in actual protein data and published methods.

### Knowledge Accumulation

The `KnowledgeAccumulator` maintains a 4-layer persistent store:

| Layer | Contents | Examples |
|-------|----------|----------|
| **Raw** | Ingested papers | Titles, abstracts, metadata |
| **Domain** | Extracted entities | Genes, proteins, diseases, methods |
| **Outcomes** | Generated results | Hypotheses, experiment protocols |
| **Meta** | Cross-session patterns | Research trends, methodological insights |

All entries are deduplicated by deterministic hashing. Session tracking records research topics and summaries for continuity across sessions.

---

## Demo Walkthrough

This example traces a complete workflow for immunology research.

### 1. Search for Papers

Enter **"T cell exhaustion checkpoint immunotherapy"** in the Literature Explorer. SciDEX queries Semantic Scholar and retrieves 20 papers with abstracts, citation counts, and SPECTER2 embeddings. Papers are stored in ChromaDB for semantic search.

### 2. Build Knowledge Graph

Navigate to the Knowledge Graph page. The extractor identifies entities from paper abstracts:
- **Genes**: PD1, CTLA4, LAG3, TIM3, TOX, NFAT
- **Diseases**: melanoma, lung cancer
- **Methods**: flow cytometry, scRNA-seq, CRISPR
- **Proteins**: various checkpoint receptors

Relationships are built from co-occurrence patterns and structural metadata (authorship, citations).

### 3. Detect Gaps and Generate Hypotheses

In the Hypothesis Workshop, click **"Generate Hypotheses"**. The four strategies find:
- **Gap detection**: TOX and metabolic reprogramming are both well-connected but not directly linked
- **Swanson linking**: PD1 -> epigenetic modification -> mitochondrial dysfunction (A-B-C chain through 3 intermediaries)
- **Analogy**: CRISPR screening methods from cancer biology could be applied to exhaustion reversal
- **Contradiction**: conflicting reports on whether PD1 blockade restores vs. remodels exhausted T cells

### 4. Refine via GDE

Click **"Run GDE"** with 3 rounds. Watch the adversarial loop:
- **Round 1**: 3 critics score all hypotheses. The methodologist flags the PD1-epigenetics chain as lacking a clear experimental readout. The domain expert rates it high on plausibility. Tournament selects the top 2.
- **Round 2**: Evolved hypotheses incorporate critic feedback -- the PD1-epigenetics hypothesis now specifies ATAC-seq as the measurement approach.
- **Round 3**: Final selection produces a refined hypothesis with confidence 0.78 and testability 0.85.

### 5. Design Experiment

Select the top hypothesis and click **"Design Experiment"**. SciDEX generates a protocol:
- **IV**: PD1 blockade + epigenetic modifiers (BET inhibitors)
- **DV**: Chromatin accessibility (ATAC-seq peaks), T cell cytotoxicity
- **Controls**: isotype antibody (negative), anti-PD1 alone, BET inhibitor alone
- **Stats**: Two-way ANOVA, n=10/group, Bonferroni correction
- **Timeline**: 12 weeks

Export as PDF for lab review.

---

## Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Language | Python 3.11+ | Core platform |
| Package Manager | uv | Dependency management |
| Literature API | Semantic Scholar Graph API | Paper search, metadata, SPECTER2 embeddings |
| Vector Store | ChromaDB | Semantic similarity search over papers |
| Knowledge Graph | NetworkX | In-memory directed graph with query and serialization |
| LLM | GitHub Copilot Models (GPT-4o) | Entity extraction, hypothesis generation, critique, experiment design |
| Frontend | Streamlit | Multi-page interactive application |
| Visualization | Plotly, pyvis | Knowledge graph and data visualization |
| PDF Export | WeasyPrint | Experiment protocol PDF generation |
| Protein Data | UniProt REST API | Protein sequences and annotations |
| Literature | PubMed E-utilities | Full-text article search |
| Data Models | Pydantic | Structured validation for hypotheses, protocols, and knowledge entries |

---

## LLM Integration

SciDEX uses **GitHub Copilot Models** via the OpenAI-compatible API endpoint at `models.inference.ai.azure.com`. The default model is **GPT-4o**.

LLM is used for:
- Hypothesis critique (3 independent critic agents per debate round)
- Hypothesis evolution (combining survivors with critic feedback)
- Contradiction mining (identifying conflicting findings across papers)
- Experiment protocol generation (structured JSON output)
- Testability score refinement

All LLM calls include retry logic with exponential backoff for rate limits and transient server errors. Entity extraction and relationship building use regex patterns, not LLM calls, for speed and determinism.

---

## API Keys

### GitHub Token (Required)

Used for LLM access via GitHub Copilot Models.

```bash
export GITHUB_TOKEN=$(gh auth token)
# or set manually in .env
```

### Semantic Scholar API Key (Optional)

Without a key, the S2 API is rate-limited to approximately 1 request/second. With a key, the limit increases to 100 requests/second.

Apply at: https://www.semanticscholar.org/product/api

```bash
# Add to .env
S2_API_KEY=your_key_here
```

The S2 client enforces rate limiting internally via a thread-safe throttle mechanism, so you do not need to manage request timing manually.

---

## Testing

```bash
# Run the full test suite
uv run pytest

# Run with verbose output
uv run pytest -v

# Run a specific module's tests
uv run pytest tests/test_s2_client.py
```

The test suite contains **249 tests** across 15 test files covering all modules. Tests use mock LLM functions injected via the `chat_fn` parameter, so they run without API tokens.

---

## Team

Built at the **Nautilex Hackathon**, Allen Institute.

**Challenge Owner**: Kris Ganjam, OCTO
