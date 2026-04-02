# SciDEX Evidence, Comparison, and Workspaces Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add evidence-grounded hypothesis cards, side-by-side hypothesis comparison/ranking, and saved workspaces that let users resume SciDEX research sessions.

**Architecture:** Keep the current Streamlit page flow intact and add three small seams: a JSON-backed workspace store for persistence, a hypothesis ranking/evidence enrichment layer for display quality, and page-level UI wiring that reads/writes those structured objects through session state. Reuse existing hypothesis models, knowledge graph JSON, and exporter patterns rather than introducing a new database or a new app shell.

**Tech Stack:** Python 3.11, Pydantic, Streamlit, pytest, existing SciDEX JSON persistence patterns

---

## Boundaries

### Modules
| Module | Responsibility | Owns |
|--------|---------------|------|
| `scidex.workspace` | Persist and retrieve named workspace snapshots | `src/scidex/workspace/` |
| `scidex.hypothesis` | Hypothesis models, evidence enrichment, ranking helpers | `src/scidex/hypothesis/` |
| `app/pages/3_hypothesis.py` | Generate, compare, bookmark, save, and resume hypothesis work | `app/pages/3_hypothesis.py` |
| `app/pages/4_experiment.py` | Consume saved/bookmarked hypotheses and show active workspace context | `app/pages/4_experiment.py` |

### Interfaces (contracts between modules)
| From → To | Interface | Shape |
|-----------|-----------|-------|
| `app/pages/3_hypothesis.py` → `scidex.workspace` | `WorkspaceStore.save(snapshot)` / `list_workspaces()` / `load(name)` | sync |
| `app/pages/3_hypothesis.py` → `scidex.hypothesis` | `enrich_report(report, kg, topic)` / `rank_hypotheses(hypotheses)` | sync |
| `app/pages/4_experiment.py` → `scidex.workspace` | `get_active_workspace()` or session snapshot dict | sync |
| `scidex.workspace` → filesystem | JSON file persistence under project data dir | sync |

### Data Flow
User generates hypotheses on the hypothesis page. The page enriches each hypothesis with structured evidence and computed ranking metrics, stores the enriched report in session state, and optionally persists the full page snapshot as a named workspace. Loading a workspace restores the session state objects used by the hypothesis and experiment pages. Comparison UI reads from the enriched hypotheses and renders a table plus side-by-side evidence sections.

### Invariants
- Workspace persistence must stay JSON-serializable and local-file based.
- Existing hypothesis fields (`supporting_evidence`, `source_papers`) remain backward-compatible.
- If no workspace exists, the app behaves exactly as it does today.
- Ranking must be deterministic and must not require LLM access.

### Task 1: Workspace persistence backend [Wave 1]

**Files:**
- Create: `src/scidex/workspace/__init__.py`
- Create: `src/scidex/workspace/models.py`
- Create: `src/scidex/workspace/store.py`
- Test: `tests/test_workspace_store.py`

**Step 1: Write the failing tests**

Add tests for:
- saving a named workspace snapshot with topic, report, graph JSON, bookmarks, GDE result, and current protocol
- listing workspaces in reverse chronological order
- loading a saved workspace by name
- rejecting duplicate names by replacing the existing snapshot instead of duplicating it

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workspace_store.py -v`
Expected: FAIL because `scidex.workspace` does not exist yet.

**Step 3: Write minimal implementation**

Create a small Pydantic workspace snapshot model and a JSON-backed `WorkspaceStore` modeled after `KnowledgeAccumulator`, with methods:
- `save(snapshot)`
- `list_workspaces()`
- `load(name)`

Use one JSON file and keep the schema flat/simple.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_workspace_store.py -v`
Expected: PASS

### Task 2: Evidence enrichment and ranking [Wave 2]

**Files:**
- Modify: `src/scidex/hypothesis/models.py`
- Create: `src/scidex/hypothesis/ranking.py`
- Modify: `src/scidex/hypothesis/generator.py`
- Test: `tests/test_hypothesis_generator.py`
- Create: `tests/test_hypothesis_ranking.py`

**Step 1: Write the failing tests**

Add tests for:
- enriched hypotheses include structured evidence sections derived from existing evidence/paper fields
- ranking assigns a deterministic composite score and sorts strongest hypotheses first
- evidence enrichment preserves existing hypothesis fields

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hypothesis_generator.py tests/test_hypothesis_ranking.py -v`
Expected: FAIL because the new evidence/ranking structures do not exist.

**Step 3: Write minimal implementation**

Add lightweight structured evidence fields to the hypothesis model and a small ranking helper that computes a composite score from confidence, novelty, testability, evidence count, and source paper count. Call the enrichment/ranking pass from `HypothesisGenerator.generate()` before returning the report.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hypothesis_generator.py tests/test_hypothesis_ranking.py -v`
Expected: PASS

### Task 3: Hypothesis page UI for compare + save/load [Wave 3]

**Files:**
- Modify: `app/pages/3_hypothesis.py`
- Modify: `app/pages/4_experiment.py`
- Modify: `app/streamlit_app.py`
- Optionally modify: `src/scidex/experiment/exporter.py` if workspace context needs export support
- Test: `tests/test_experiment.py`
- Create: `tests/test_workspace_ui_contracts.py`

**Step 1: Write the failing tests**

Add tests for:
- workspace snapshots can be translated to/from the page session payloads used by hypotheses/experiments
- comparison payload exposes top-ranked hypotheses and includes evidence sections for display
- experiment page can hydrate a `Hypothesis` from saved workspace/bookmark data

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_experiment.py tests/test_workspace_ui_contracts.py -v`
Expected: FAIL because the UI contract helpers do not exist yet.

**Step 3: Write minimal implementation**

Update the hypothesis page to:
- show a composite-score sort option
- allow selecting 2-3 hypotheses for side-by-side comparison
- render structured evidence cards for each selected hypothesis
- save and load named workspaces

Update the experiment page to surface the active workspace and read the restored hypothesis/session state.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_experiment.py tests/test_workspace_ui_contracts.py -v`
Expected: PASS

### Task 4: Full verification and cleanup [Wave 4]

**Files:**
- Modify only if needed based on test failures

**Step 1: Run focused suite**

Run: `uv run pytest tests/test_workspace_store.py tests/test_hypothesis_generator.py tests/test_hypothesis_ranking.py tests/test_experiment.py tests/test_workspace_ui_contracts.py -v`
Expected: PASS

**Step 2: Run full suite**

Run: `uv run pytest`
Expected: PASS with existing 249 tests plus new ones.

**Step 3: Self-review for smallest adequate diff**

Check for unnecessary abstraction or duplicated session-state helpers. Collapse anything obviously overbuilt.

## Wave Summary

| Wave | Tasks | Can Parallelize? |
|------|-------|-----------------|
| Wave 1 | 1 | N/A |
| Wave 2 | 2 | N/A |
| Wave 3 | 3 | N/A |
| Wave 4 | 4 | N/A |
