# Workspace Persistence

Last updated: 2026-04-01

## Key Facts
- [source] Workspace snapshots now persist `designed_protocols` and the hypothesis input widget values (`hyp_topic`, `hyp_source_entity`, `hyp_source_method`) in `src/scidex/workspace/models.py` and `src/scidex/workspace/session.py`.
- [source] `WorkspaceStore.save()` only updates in-memory state after the JSON write succeeds; failed writes raise `OSError` instead of silently reporting success.
- [source] Session helpers are imported from `scidex.workspace.session`, while `scidex.workspace` only re-exports `WorkspaceStore` and `WorkspaceSnapshot`.

## Gotchas
- [synthesis] ALWAYS treat workspace saves as atomic: write first, then replace `_workspaces`, or the UI can claim a save succeeded when disk persistence failed.
- [synthesis] ALWAYS give persisted Streamlit inputs explicit widget keys if they must survive a workspace restore.
- [synthesis] NEVER monkeypatch a bound `Path.write_text` on a `PosixPath` instance in pytest; patch `Path.write_text` on the class instead so teardown can restore it.

## Related Files
- `src/scidex/workspace/store.py` — atomic save and error propagation.
- `src/scidex/workspace/session.py` — snapshot/restore payload contract.
- `app/pages/3_hypothesis.py` — workspace save/load UI and keyed inputs.
- `app/pages/4_experiment.py` — protocol page restore/hydration path.
- `tests/test_workspace_store.py` — persistence failure and import-surface regression tests.

## Sources
- `src/scidex/workspace/models.py`
- `src/scidex/workspace/store.py`
- `src/scidex/workspace/session.py`
- `app/pages/3_hypothesis.py`
- `app/pages/4_experiment.py`
- `tests/test_workspace_store.py`
