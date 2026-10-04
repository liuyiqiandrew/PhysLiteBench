# Benchmark interface and validation

The active target is now **ten tasks at 0/3 each** in unhinted Luna-high trials, with predominantly physical-model failures. **3 of 10 currently qualify**: insulating-mhd-duct, magnetic-tracer and reaction-diffusion. The other seven require revision or replacement. [Current goal status](../results/zero-three-goal-status.json) records source checks and the stricter protocol; the completed batches below remain historical evidence.

PhysLiteBench tests whether a coding agent revises a physical approximation that fits calibration but fails elsewhere in the stated apparatus. Ten tasks were selected through a conditional screen with one reviewed physical failure each (0/1). A separate fresh three-trial confirmation found 6 of ten at or below one pass; [the dashboard](../DASHBOARD.md) separates both batches from archived scores.

## Agent task

Each task supplies an `instruction.md`, a public Docker environment, calibration data, a Python predictor and public tests. Its README defines the apparatus, allowed preparations, observables, units, parameter ranges and API. The instruction explicitly permits modifying or replacing every prediction function and helper while preserving the documented API. Private cases use the same stated apparatus with preparations outside the calibration subset.

The ordinary image contains only `environment/`. Private `tests/`, `solution/`, `AUTHOR.md` and `hint.md` stay outside it. The runner adds the hint only to the instruction when `--hint` is requested. The frozen environment, grading and configuration must remain byte-identical between paired ordinary and hinted batches.

Each task uses a model with `fit` returning itself and a prediction method documented in its public README. Inputs and shapes vary by apparatus. Follow each task's interface rather than assuming a shared prediction signature. Public tests check interface and calibration; the private verifier owns its own calibration copy and physical reference.

## Reward and diagnosis

A trial receives reward 1 only if every required check passes. The verifier tests interface, finite outputs, calibration goodness of fit, parameter recovery and multiple physical predictions. Most new tasks require reduced calibration chi-square below 1.5 and relative parameter error below 3%; hidden tolerances and normalization are task-specific. Their author notes explain numerical convergence and separation from measurement noise.

A failed unfinished fit is not the intended shortcut control. Every candidate includes a completed calibration fit for both a correct oracle and the supplied physical approximation. Both must recover the parameter and fit calibration; only the wrong physical approximation should fail the distinguishing experiments.

Review final code, public messages, commands and verifier metrics before classifying an agent failure. Separate incorrect physical assumptions from algebra, discretization, coding, optimizer and infrastructure failures. Diagnostic repairs run in memory at the retained fitted parameter; they never modify the original submission or its recorded reward.

## Reproduction

Use Docker and uv. From the repository root:

```bash
python3 scripts/run_candidate_matrix.py magnetic-tracer \
  --conditions oracle shortcut plain --workers 1 --label fresh-three
```

The matrix runner checks the oracle and completed shortcut, then runs exactly three unhinted Luna high trials. The oracle reward should be 1 and the completed shortcut reward 0. Infrastructure errors stop that task for review. New job names must be unique; never overwrite an existing result. Staged candidates can use `--task-root staging/REVISION/tasks --control-root staging/REVISION/scripts`.

The historical conditional screen used `run_neutrality_screen.py`: an initial pass triggered two more trials, while an initial failure stopped at 0/1. The initial outcome was never discarded. The fresh confirmation used `run_candidate_matrix.py --conditions plain` for all ten retained tasks; it did not use conditional stopping or hints.

An entirely unstarted initial or follow-up batch in the conditional runner may be replaced with `--retry-unstarted-job JOB` after review confirms a network setup failure, no agent execution, no verifier result and no native session. The runner verifies that source, grading, instruction and configuration still match the frozen task and preserves the failed attempts. A replacement initial trial follows the same conditional rule; replacement follow-ups run only the missing trials. Reports distinguish total attempts from actual model trials. A failure after the agent starts is never excluded by this rule.

Pinned setup: Python 3.13, Harbor 0.21.0, Codex CLI 0.154.0, NumPy 2.3.3, SciPy 1.16.3 and pytest 8.4.2 in new task images. The agent/verifier timeouts are 600/60 seconds. Authentication remains outside the repository.

Scientific validators run independently of agent trials:

```bash
uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 \
  --with pytest==8.4.2 python scripts/validate_magnetic_tracer.py
```

Check each task's author notes for its exact command and report. Ordinary validation reads fixed data; `--generate` intentionally replaces both public and private calibration copies. New candidate validators use 256 noisy calibration fits. Some also evaluate every hidden case on every draw; others check hidden sensitivity at the observed parameter extrema. These are different checks and are labeled as such in the reports.

Independent references use a different derivation or representation from the oracle: examples include constrained pair-friction elimination, conservative heat-flux balance, entropy/strain constitutive systems, finite-volume current closure and Fourier-moment evolution. Check relevant conservation laws, stable/positive limits and numerical refinement as well as prediction agreement.

## Evidence and selection

The runner preserves a frozen task snapshot, native sessions, trajectory, final `/app` artifacts, source diffs, verifier metrics and timing under ignored `jobs/`. [The full candidate ledger](../results/candidates.json) summarizes all indexed batches and merges separate per-trial reviews. Regenerate it and the selection status with:

```bash
python3 scripts/summarize_candidates.py
python3 scripts/summarize_neutrality.py
```

Selection includes every neutral-instruction batch matching the current environment, grading, instruction and configuration hashes. Do not rerun unchanged candidates until a favorable zero appears. A scientific revision is evaluated separately and its predecessor preserved. The original results and paired hint audits are historical and preserved in the pre-neutral archive. A current 0/1 failure is not interchangeable with an archived 0/3.

The user authorized archiving tasks above the 1/3 cutoff. [Screened tasks](../archives/screened/README.md) remain runnable by passing their archive path to `run_science.py`; their source hashes and evaluation IDs are in the archive manifest. Their individual validators and baselines moved with them. The shared first-wave validator resolves retained and archived paths.

One or three conditional trials measure a development outcome, not a population success probability. Selection against Luna and reuse of calibration observations are disclosed. No held-out generalization or future 3/3-failure guarantee is claimed.

For a new task's files, assumptions, review and validation requirements, use [ADDING_TASKS.md](ADDING_TASKS.md). Prior versions of this guide are preserved in the documentation archive.
