# PhysLiteBench

The Desktop roster now contains **thirteen physics tasks**: the ten locally qualified snapshots below plus three additions from the remote branch. The [full roster](results/task-roster.json) records each exact revision, source hashes and evaluation status. The latest evidence does **not** establish 0/3 passes for all thirteen.

## Three imported tasks

| Imported task | Revision | Original selection passes | Latest GPT-5.6 Luna High passes |
|---|---:|---:|---:|
| [Swirling heat tails](tasks/swirling-heat-tails/environment/README.md) | 1 | 0/3 | **1/3** |
| [Hard-rod current](tasks/hard-rod-current/environment/README.md) | 2 | 0/3 | **0/3** |
| [Finite-ion cloud](tasks/finite-ion-cloud/environment/README.md) | 3 | 0/3 | **0/3** |

The three additions were merged from `origin/luna-high-qualified-tasks` at `6f4ca59`. Their fresh confirmation retained all nine completed trials with no infrastructure errors or retries. Swirling heat tails has a passing solution; hard-rod current has three clean physical failures; finite-ion cloud has two clean physical failures and one mixed physical/mathematical failure. These are the remote results, not new local benchmark runs. See the [confirmation results](results/expansion-confirmation-20261011-1709-results.json) and [independent review](results/expansion-confirmation-20261011-1709-review.md).

Their web search was disabled, but shell networking remained available. This differs from the original ten's restricted-network qualification. The imported review found no external retrieval in the emitted trajectories. Original selection results and the passing confirmation outcome are both preserved.

## Original ten qualified tasks

The preserved local set contains **ten qualified physics tasks**, each with **0 passes in 3 completed unhinted GPT-5.6 Luna High trials**. All thirty selected trials completed without infrastructure exceptions, and their failures were reviewed as physical-model errors. The [result manifest](results/qualified-tasks.json) identifies the exact source hashes, trial IDs and evidence.

| Task and public apparatus | Revision | GPT-5.6 Luna High passes |
|---|---:|---:|
| [Entropy anomaly](tasks/entropy-anomaly/environment/README.md) | 9 | 0/3 |
| [Finite-layer phoresis](tasks/finite-layer-phoresis/environment/README.md) | 2 | 0/3 |
| [Hierarchical glass overlaps](tasks/hierarchical-glass-overlaps/environment/README.md) | 2 | 0/3 |
| [Insulating MHD duct](tasks/insulating-mhd-duct/environment/README.md) | 2 | 0/3 |
| [Magnetic tracer](tasks/magnetic-tracer/environment/README.md) | 9 | 0/3 |
| [Reaction–diffusion](tasks/reaction-diffusion/environment/README.md) | 10 | 0/3 |
| [Ring-current fluctuations](tasks/ring-current-fluctuations/environment/README.md) | 1 | 0/3 |
| [Rotating reservoir](tasks/rotating-reservoir/environment/README.md) | 11 | 0/3 |
| [Single-file memory](tasks/single-file-memory/environment/README.md) | 1 | 0/3 |
| [Superconducting heat](tasks/superconducting-heat/environment/README.md) | 11 | 0/3 |

The October 11 fresh rerun confirmed five previous revisions and found at least one pass on each of the other five. Entropy anomaly R9, magnetic tracer R9, reaction–diffusion R10, rotating reservoir R11 and superconducting heat R11 are the newly qualified revisions. Each received independent scientific validation, oracle/finished-approximation controls, exactly three fresh trials, source and trajectory review, and same-parameter causal repairs. All fifteen repairs passed. Earlier passing revisions and unsuccessful development batches remain in the [development ledger](results/qualification-20261011/development-ledger.json); the [original fresh rerun](results/fresh-ten-luna-high-20261011.json) remains unchanged.

These tasks were selected and refined against GPT-5.6 Luna. Observed 0/3 outcomes do not establish a zero future pass rate. The separate five-model comparison uses frozen task versions and reports its own runtime and access limitations in [the dashboard](DASHBOARD.md).

Each task supplies public apparatus, starter code, calibration and interface tests under `environment/`; private grading and an independent reference under `tests/`; and a correct solution under `solution/`. Completed approximation controls and validators are under `scripts/`. The public README defines the API; AUTHOR.md records the derivation. Private material stays outside ordinary agent images.

## Run a task

Start Docker, install uv, and use the pinned runner:

```bash
python3 scripts/run_science.py tasks/hierarchical-glass-overlaps --agent oracle --trials 1
python3 scripts/run_science.py tasks/hierarchical-glass-overlaps   --model gpt-5.6-luna --reasoning-effort high --trials 3 --concurrency 3   --restricted-network --agent-setup-timeout-multiplier 3
```

Qualification of the original ten used Harbor 0.21.0, Codex CLI 0.154.0, 600 seconds for solving and 60 seconds for verification. `--restricted-network` reproduces the runtime allowlist used in qualification: only OpenAI authentication/API hosts are reachable, and agent web search is disabled. It changes only the temporary runtime configuration and preserves source task bytes. Keep authentication outside the repository.

Task sources in both manifests exactly match their evaluated snapshots. The ten-task qualification manifest remains unchanged; the full roster adds the three imported snapshots. Prior canonical tasks and documentation are preserved in [the October 11 archive](archives/pre-qualification-20261011). Earlier post-qualification maintenance is historical evidence. Raw native jobs remain local; selected submissions, metrics, diffs and reviews are in [the evidence folder](results/qualification-20261011).

The inherited folders `tasks/hydrodynamic-heating` and `tasks/thermoelastic-rod` are outside this thirteen-task roster. See [benchmark protocol](docs/BENCHMARK.md) and [contribution guidance](docs/ADDING_TASKS.md).
