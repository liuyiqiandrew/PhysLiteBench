# Screened candidates

These tasks did not meet the retained benchmark cutoff of at most one unhinted pass in three. Their source, completed controls and validators are preserved. The manifest's `archived_path` identifies the latest revision; some entries point to a separate revision archive to preserve an earlier copy here. The manifest records results and source hashes; all exploratory batches remain in `../../results/candidates.json` and the original ignored job directories.

Run an archived task from the repository root with `python3 scripts/run_science.py archives/screened/tasks/<task> --agent oracle --trials 1`. For its completed shortcut add `--solution-model archives/screened/scripts/<task_with_underscores>_baseline.py`. Individual scientific validators run from `archives/screened/scripts/`; their default output is under this archive's `jobs/`. The shared first-wave validator remains at `scripts/validate_candidates.py` and resolves both retained and archived paths.

Archiving changes location only, not the task instruction, public environment or private grading. This archive is not part of the retained task set.
