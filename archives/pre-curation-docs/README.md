# PhysLiteBench

A lightweight benchmark for physical reasoning in scientific code.

Harbor tasks test whether a coding agent checks a model's physical assumptions before trusting a successful calibration fit. Each task supplies an apparatus description, calibration data, and a predictor with an unfinished fit. The predictor agrees with calibration but makes the wrong prediction for other allowed preparations.

| Task | Physical issue | Revision |
|---|---|---|
| [Spin probes](tasks/qubit-control/environment/README.md) | Two probes experience the same fluctuating field; their responses are correlated. | 6 |
| [Gas chambers](tasks/thermal-bodies/environment/README.md) | A free piston equalizes pressure without keeping it constant over time. | 9 |
| [Electrodiffusion](tasks/reaction-diffusion/environment/README.md) | Two salts sharing an anion must respond to one common electric field. | 6 |

## Candidate expansion and physics hints

Fourteen additional candidates use matched plain and hinted three-trial
GPT-5.6 Luna high-effort batches. Each candidate has a completed
physical oracle, a completed wrong-model control, independent verification,
calibration data, and an author-owned `hint.md`. The hint is appended only to the
agent instruction; data, starter, tests, model, reasoning effort, and time limits
remain unchanged.

| Candidate | Physical distinction |
|---|---|
| `hall-bar` | Insulating sidewalls constrain transverse current, not electric field. |
| `moving-inductor` | Core motion changes flux linkage and exchanges mechanical work. |
| `effusive-beam` | Aperture crossing events sample velocities with a flux weight. |
| `real-gas-expansion` | The equation of state constrains volume-dependent internal energy. |
| `induced-dipole` | Creating a dipole costs internal polarization energy. |
| `radiative-plates` | Reflected radiation contributes to subsequent irradiation. |
| `magnetic-equilibrium` | A self-consistent stationary root may be thermodynamically unstable. |
| `spatial-diffusion` | Isothermal variable mobility requires noise-induced drift. |
| `hydrodynamic-beads` | Independent equilibrium positions can have coupled dynamics. |
| `poroelastic-solid` | Trapped pore fluid stiffens volumetric deformation. |
| `competitive-adsorption` | Two species compete for one population of adsorption sites. |
| `compensated-conductor` | Two carrier populations share one transverse electric field. |
| `fluid-coexistence` | Phase separation replaces an unstable homogeneous equation of state. |
| `first-order-magnet` | A field-aligned, locally stable state can still have higher free energy. |

Run a matched pair from this directory:

```bash
python3 scripts/run_science.py tasks/hall-bar --model gpt-5.6-luna --reasoning-effort high --trials 3
python3 scripts/run_science.py tasks/hall-bar --model gpt-5.6-luna --reasoning-effort high --trials 3 --hint
```

The new-task evidence is in [the candidate results](results/candidates.json),
[calibration validation](results/candidate-validation.json), and the
[physics review](docs/candidates/physics-review.md). None achieved the requested
0/3 unhinted result. Hall-bar and spatial diffusion showed the intended physical
failures, with unhinted passes of 1/3 and 2/3 respectively; both scored 3/3 with
their hints. Effusion's one failure was an implementation mistake. The other
eleven candidates passed all three unhinted trials. These remain exploratory
candidates; see [the dashboard](DASHBOARD.md#candidate-expansion-matched-physics-hints)
for all observed results and their limitations.

Start with [TASKS.md](TASKS.md) for an explanation from physical intuition through worked equations. [The benchmark guide](docs/BENCHMARK.md) explains interfaces, grading, and validation. [DASHBOARD.md](DASHBOARD.md) contains the latest model comparisons and trajectory findings.

Contributing a new system? Follow [Adding a physics task](docs/ADDING_TASKS.md) for the directory layout, author/agent boundary, controls, evaluation, and PR requirements.

## Setup

Install Python 3, [uv](https://docs.astral.sh/uv/getting-started/installation/), and [Docker](https://docs.docker.com/get-started/get-docker/). Start Docker and check `docker info`. Run the commands below from this directory.

The runner uses uv to select Python 3.13 and Harbor 0.21.0. Dockerfiles pin the scientific dependencies; no project-wide virtual environment is required.

## Run a task

First check the reference solution in Docker:

```bash
python3 scripts/run_science.py tasks/qubit-control --agent oracle --trials 1
```

The expected reward is **1**. Replace `tasks/qubit-control` with `tasks/thermal-bodies` or `tasks/reaction-diffusion` to run either other task. The `nop` agent leaves the starter unchanged and is expected to receive **0**, because its fit is unfinished.

For Codex trials, install the Codex CLI and sign in with `codex login`. Authentication stays outside this repository: the runner uses `~/.codex/auth.json`, or the path specified by `CODEX_AUTH_JSON_PATH`. The container's CLI is pinned to 0.154.0.

```bash
python3 scripts/run_science.py tasks/qubit-control \
  --model gpt-5.6-luna --reasoning-effort high \
  --trials 3 --concurrency 3
```

Choose a model available to your account with `--model`. Independent task batches can also run concurrently. Each trial has a 600-second agent limit and a 60-second verifier limit.

Runs receive fresh timestamped names under `jobs/`. An explicit `--job-name` must be unused. The runner retains the tested task snapshot, trajectories, final source, diffs, metrics, and timing. Generated jobs are excluded from version control.

To check the completed but physically incorrect model, use an oracle run with a shortcut supplied as its solution:

```bash
python3 scripts/run_science.py tasks/qubit-control --agent oracle --trials 1 \
  --solution-model scripts/qubit_noise_baseline.py
```

This control should fit calibration and recover the parameter, then fail only hidden predictions. See [validation instructions](docs/BENCHMARK.md) for all three controls and local checks.

## What is included

```text
tasks/
  qubit-control/       Shared-field spin task
  thermal-bodies/      Free-piston gas task
  reaction-diffusion/  Shared-anion transport task
  .../                Additional candidates listed above
scripts/               Runners, validators, and completed shortcut controls
docs/                  Benchmark and grading guide
results/               Latest evaluation statistics and selected diagnostics
```

Each task contains `instruction.md`, `task.toml`, `environment/`, `tests/`, and `solution/`. This is an author bundle with private verifiers and reference solutions. Harbor exposes only the instruction and Docker environment to the solving agent; keep that boundary when using another harness.

This directory is self-contained and can be shared on its own. Earlier tasks, development notes, and full historical run logs are archived separately. The bundled dashboard identifies the runs behind its summaries; it does not include their full transcripts. Candidate evaluations retain their full local artifacts in ignored `jobs/` directories.
