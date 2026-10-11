# PhysLiteBench

This branch publishes **thirteen qualified physics tasks**, each with **0 passes in 3 completed unhinted Luna High trials**. The model was `gpt-5.6-luna` with reasoning effort `high`. Every qualifying failure was reviewed as a physical-model failure. Exact source hashes and trial IDs are in [the result manifest](results/qualified-tasks.json).

| Task and public apparatus | Revision | Passes | Reviewed physical failures |
|---|---:|---:|---:|
| [Entropy anomaly](tasks/entropy-anomaly/environment/README.md) | 7 | 0/3 | 3 |
| [Finite-ion cloud](tasks/finite-ion-cloud/environment/README.md) | 3 | 0/3 | 3 |
| [Finite-layer phoresis](tasks/finite-layer-phoresis/environment/README.md) | 2 | 0/3 | 3 |
| [Hard-rod current](tasks/hard-rod-current/environment/README.md) | 2 | 0/3 | 3 |
| [Hierarchical glass overlaps](tasks/hierarchical-glass-overlaps/environment/README.md) | 2 | 0/3 | 3 |
| [Insulating MHD duct](tasks/insulating-mhd-duct/environment/README.md) | 2 | 0/3 | 3 |
| [Magnetic tracer](tasks/magnetic-tracer/environment/README.md) | 2 | 0/3 | 3 |
| [Reaction–diffusion](tasks/reaction-diffusion/environment/README.md) | 7 | 0/3 | 3 |
| [Ring-current fluctuations](tasks/ring-current-fluctuations/environment/README.md) | 1 | 0/3 | 3 |
| [Rotating reservoir](tasks/rotating-reservoir/environment/README.md) | 8 | 0/3 | 3 |
| [Single-file memory](tasks/single-file-memory/environment/README.md) | 1 | 0/3 | 3 |
| [Superconducting heat](tasks/superconducting-heat/environment/README.md) | 6 | 0/3 | 3 |
| [Swirling heat tails](tasks/swirling-heat-tails/environment/README.md) | 1 | 0/3 | 3 |

The three-trial counts cover completed scientific evaluations; separate setup/infrastructure attempts for phoresis and single-file memory remain listed in the manifest.

Each task has a public apparatus, starter, calibration data and interface tests under `environment/`, private grading and an independent physical reference under `tests/`, and the correct implementation under `solution/`. Completed approximation controls and validators are under `scripts/`. Follow each task's public README for its exact API and its AUTHOR.md for derivations and validation details.

Hierarchical-glass-overlaps revision 2 completed the original ten-task roster. It predicts labelled replica partitions in a quenched Gaussian hierarchy. All three solutions fitted calibration correctly but retained the wrong parent–descendant correlation law. Correcting only that physical law at the original fitted widths repaired every graded group. The physical oracle passed all nine Docker tests; the completed approximation passed four and failed the five intended prediction groups. Its earlier revision had 1/3 passes and did not qualify.

The three requested additions are swirling-heat-tails r1, hard-rod-current r2 and finite-ion-cloud r3. Their reviewed failures respectively omit rare endpoint-energy constraints, current-conditioned rod composition, and the shared-angle constraint in a three-ion cell. Correcting only each response at its retained fitted parameter repairs every hidden group. Each oracle passes 7/7 Docker tests; each completed approximation passes four checks and fails the three intended prediction groups. See the [heat results](results/expansion-swirling-heat-tails-r1-results.json), [rod results](results/expansion-hard-rod-current-r2-results.json), [ion results](results/expansion-finite-ion-cloud-r3-results.json), and [expansion status](results/physics-expansion-status.json). All 66 trials across 22 revisions are retained; nineteen excluded revisions remain in the [expansion archive](archives/physics-expansion-20261011/README.md).

## Run a task

Start Docker, install uv, and use the pinned runner:

```bash
python3 scripts/run_science.py tasks/hierarchical-glass-overlaps --agent oracle --trials 1
python3 scripts/run_science.py tasks/hierarchical-glass-overlaps \
  --model gpt-5.6-luna --reasoning-effort high --trials 3 --concurrency 3
```

The runner pins Harbor 0.21.0 and Codex CLI 0.154.0. Ordinary runs use no physics hint. Keep authentication outside the repository. Hidden data and solutions are in this repository, and a trial agent with web access could read them while the repository is public. Before rerunning trials, make the repository private or disable agent web access.

Four tasks changed after qualification without new trials: entropy anomaly, insulating MHD duct, rotating reservoir and superconducting heat. Their verifiers now grade against frozen reference values; old and new verifiers gave identical results on every checked submission. The superconducting-heat starter no longer carries the true conductance as a default. The manifest still records the qualified snapshots in `source_sha256` and lists every changed file under `post_qualification_changes`; see [post-qualification maintenance](docs/BENCHMARK.md#post-qualification-maintenance).

Raw jobs, native transcripts and private staging work remain local. This branch publishes task sources, completed controls, validators, result records and the reviewed expansion archive. The inherited task folders `tasks/hydrodynamic-heating` and `tasks/thermoelastic-rod` are not in the qualified roster above. See [repository contribution guidance](docs/ADDING_TASKS.md).
