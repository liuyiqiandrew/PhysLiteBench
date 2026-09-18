# Repository Guidelines

## Structure

The three tasks are `tasks/qubit-control/`, `tasks/thermal-bodies/`, and `tasks/reaction-diffusion/`. Each has public files in `environment/`, private verification in `tests/`, and a reference solution in `solution/`. Keep private answers, tests, and author documentation out of agent images.

Read `TASKS.md` for physics, `docs/BENCHMARK.md` for grading and validation, and `DASHBOARD.md` for the latest results. `scripts/` contains the runner, validators, and completed shortcut controls. `results/` holds shareable summaries; new run artifacts go in ignored `jobs/`.

## Development and Testing

Install uv and start Docker. From this directory:

```bash
docker info
python3 scripts/run_science.py tasks/thermal-bodies --agent oracle --trials 1
python3 scripts/run_science.py tasks/thermal-bodies --model gpt-5.6-luna \
  --trials 3 --concurrency 3 --reasoning-effort high
```

The runner pins Harbor 0.21.0 and Codex CLI 0.154.0. It generates fresh job names; never reuse a historical job directory. Keep authentication outside the repository.

Tests use pytest with `test_*.py` files and `test_*` functions. Run tests per task: separate tasks use the same module names. Validate the physical oracle and completed shortcut independently. The oracle must pass; the shortcut must pass calibration and parameter checks but fail hidden predictions. Unfinished starters may fail public tests. Local validator commands are in `docs/BENCHMARK.md`.

## Style and Benchmark Integrity

Keep design and implementation minimal. Use straightforward prose. Match nearby Python, use four-space indentation and `snake_case` names, and state physical units and assumptions. Preserve task interfaces. No formatter or linter is configured.

Freeze inputs and grading during each evaluation batch. Regenerate calibration only intentionally. Preserve prior artifacts and inspect final code and trajectories before classifying a failure. Distinguish physical mistakes from mathematical, numerical, coding, and infrastructure errors. Keep dashboard comparisons limited to the newest batches and retain their provenance.

## Review Guidance

No Git history establishes a commit convention. Use short imperative subjects and focused changes. Review descriptions should explain the physical behavior changed, validation performed, and relevant run evidence. Link related issues when applicable.
