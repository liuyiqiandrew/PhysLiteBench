# PhysLiteBench

The active target is now **ten tasks at 0/3 each** in unhinted Luna-high trials, with predominantly physical-model failures. **3 of 10 currently qualify**: insulating-mhd-duct, magnetic-tracer and reaction-diffusion. The other seven require revision or replacement. [Current goal status](results/zero-three-goal-status.json) records source checks and the stricter protocol; the completed batches below remain historical evidence.

A lightweight benchmark for physical reasoning in scientific code. Each task supplies an apparatus, calibration data, and a predictor with an unfinished fit. The supplied physical approximation fits calibration but fails on another allowed preparation. A successful agent must check the physical model as well as fit the numbers.

All active tasks use a neutral instruction that explicitly permits replacing any prediction function or helper while preserving the documented API. The previous ten-task 0/3 result belongs to the archived instructions; it is not a result for the current sources.

The fresh confirmation batch is complete: **6 of 10 tasks meet the at-most-1/3 pass cutoff**. There were **13 passes in 30 new unhinted Luna-high trials**. 3 tasks scored 0/3. Above cutoff: `electric-dipole-force` (3/3), `entropy-anomaly` (2/3), `spin-strain-response` (3/3), `superconducting-heat` (2/3). See [fresh-batch results](results/retained-three-r1-results.json) and [the task-by-task dashboard](DASHBOARD.md).

The first neutral-instruction screen produced 29 passing rewards in 32 trials. Ten tasks exceeded the desired cutoff. One passing reward also had an agent timeout and is reported separately. The requested conditional screen is complete. Ten retained tasks each have **0 passes in 1 unhinted Luna-high trial (0/1)**, with every failure reviewed as a physical-model failure. That screen ran one unhinted trial per changed task, then exactly two more only after an initial pass. A stopped first-run failure is reported as **0/1**, never as 0/3. Every trial and prior revision is retained.

See [the current dashboard](DASHBOARD.md), [the earlier source-matched screening results](results/neutrality-status.json), and [the complete initial neutral screen](results/neutrality-initial-screen.json). [All candidate batches](results/candidates.json) preserve the development history. The old tasks, scores and documentation are in [the pre-change archive](archives/pre-neutral-instructions).

[Task explanations](TASKS.md), [interfaces and validation](docs/BENCHMARK.md), and [adding a task](docs/ADDING_TASKS.md) describe the author bundle. A private physics hint is appended only with `--hint`; the neutral screening runs do not use it.

## Setup

Install Python 3, [uv](https://docs.astral.sh/uv/getting-started/installation/), and [Docker](https://docs.docker.com/get-started/get-docker/). Start Docker and check `docker info`. Run the commands below from this directory.

The runner uses uv to select Python 3.13 and Harbor 0.21.0. Dockerfiles pin the scientific dependencies; no project-wide virtual environment is required.

## Run a task

First check the reference solution in Docker:

```bash
python3 scripts/run_science.py tasks/magnetic-tracer --agent oracle --trials 1
```

The expected reward is **1**. Replace `tasks/magnetic-tracer` with another active task directory to run it. The `nop` agent leaves the starter unchanged and is expected to receive **0**, because its fit is unfinished.

For Codex trials, install the Codex CLI and sign in with `codex login`. Authentication stays outside this repository: the runner uses `~/.codex/auth.json`, or the path specified by `CODEX_AUTH_JSON_PATH`. The container's CLI is pinned to 0.154.0. For a separate three-trial batch:

```bash
python3 scripts/run_science.py tasks/magnetic-tracer \
  --model gpt-5.6-luna --reasoning-effort high \
  --trials 3 --concurrency 3
```

Choose a model available to your account with `--model`. Independent task batches can also run concurrently. Each trial has a 600-second agent limit and a 60-second verifier limit.

Runs receive fresh timestamped names under `jobs/`. An explicit `--job-name` must be unused. The runner retains the tested task snapshot, trajectories, final source, diffs, metrics, and timing. Generated jobs are excluded from version control.

To check the completed but physically incorrect model, use an oracle run with a shortcut supplied as its solution:

```bash
python3 scripts/run_science.py tasks/magnetic-tracer --agent oracle --trials 1 \
  --solution-model scripts/magnetic_tracer_baseline.py
```

This control should fit calibration and recover the parameter, then fail only hidden predictions. See [validation instructions](docs/BENCHMARK.md) for controls and local checks.

## Contents

`tasks/` contains the ten retained tasks from the completed neutral-instruction screen. `archives/screened/` contains earlier screened families, and revision archives preserve every superseded source. An additional [unevaluated backup](archives/unevaluated/README.md) is preserved separately. `scripts/` holds the runner, validators and completed wrong-model controls. `results/` contains shareable metrics, reviews and diagnostic repairs; full local traces remain in ignored `jobs/` directories. The original three tasks' raw jobs are in the adjacent historical archive.

Each task is an author bundle: Harbor exposes only the instruction and public Docker environment to the solving agent. Private grading, reference solutions and author notes must stay outside that image. Prior documentation is preserved under [archives/pre-curation-docs](archives/pre-curation-docs/README.md).
