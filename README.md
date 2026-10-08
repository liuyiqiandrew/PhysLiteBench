# PhysLiteBench

This branch publishes **ten qualified physics tasks**, each with **0 passes in 3 completed unhinted Luna High trials**. The model was `gpt-5.6-luna` with reasoning effort `high`. Every qualifying failure was reviewed as a physical-model failure. Exact source hashes and trial IDs are in [the result manifest](results/qualified-tasks.json).

| Task and public apparatus | Revision | Passes | Reviewed physical failures |
|---|---:|---:|---:|
| [Entropy anomaly](tasks/entropy-anomaly/environment/README.md) | 7 | 0/3 | 3 |
| [Finite-layer phoresis](tasks/finite-layer-phoresis/environment/README.md) | — | 0/3 | 3 |
| [Hierarchical glass overlaps](tasks/hierarchical-glass-overlaps/environment/README.md) | 2 | 0/3 | 3 |
| [Insulating MHD duct](tasks/insulating-mhd-duct/environment/README.md) | — | 0/3 | 3 |
| [Magnetic tracer](tasks/magnetic-tracer/environment/README.md) | — | 0/3 | 3 |
| [Reaction–diffusion](tasks/reaction-diffusion/environment/README.md) | — | 0/3 | 3 |
| [Ring-current fluctuations](tasks/ring-current-fluctuations/environment/README.md) | 1 | 0/3 | 3 |
| [Rotating reservoir](tasks/rotating-reservoir/environment/README.md) | 8 | 0/3 | 3 |
| [Single-file memory](tasks/single-file-memory/environment/README.md) | — | 0/3 | 3 |
| [Superconducting heat](tasks/superconducting-heat/environment/README.md) | 6 | 0/3 | 3 |

A dash means the inherited result record identifies the revision by its source hashes. The three-trial counts cover completed scientific evaluations; separate setup/infrastructure attempts for phoresis and single-file memory remain listed in the manifest.

Each task has a public apparatus, starter, calibration data and interface tests under `environment/`, private grading and an independent physical reference under `tests/`, and the correct implementation under `solution/`. Completed approximation controls and validators are under `scripts/`. Follow each task's public README for its exact API and its AUTHOR.md for derivations and validation details.

The final addition is hierarchical-glass-overlaps revision 2. It predicts labelled replica partitions in a quenched Gaussian hierarchy. All three solutions fitted calibration correctly but retained the wrong parent–descendant correlation law. Correcting only that physical law at the original fitted widths repaired every graded group. The physical oracle passed all nine Docker tests; the completed approximation passed four and failed the five intended prediction groups. Its earlier revision had 1/3 passes and did not qualify.

## Run a task

Start Docker, install uv, and use the pinned runner:

```bash
python3 scripts/run_science.py tasks/hierarchical-glass-overlaps --agent oracle --trials 1
python3 scripts/run_science.py tasks/hierarchical-glass-overlaps \
  --model gpt-5.6-luna --reasoning-effort high --trials 3 --concurrency 3
```

The runner pins Harbor 0.21.0 and Codex CLI 0.154.0. Ordinary runs use no physics hint. Keep authentication outside the repository. Task scientific sources are unchanged from the qualified Desktop snapshots; publishing ran source-integrity checks rather than new paid trials.

Raw jobs, native transcripts, private staging work and the full review archive remain local. This branch publishes the task sources, completed controls, validators and concise result records. Existing historical task folders inherited from the base branch are excluded from the ten-task qualified roster above. See [repository contribution guidance](docs/ADDING_TASKS.md).
