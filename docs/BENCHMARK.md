# Benchmark author guide

These three tasks test whether an agent checks a supplied physical model rather than only completing its parameter fit. Each starter contains a working predictor and an unfinished fit method. A completed shortcut fits the calibration but makes wrong predictions for other preparations. [TASKS.md](../TASKS.md) explains the physics and exact solutions.

## Current tasks and interfaces

| Task | Revision and physical assumption to repair | Interface |
|---|---|---|
| [qubit-control](../tasks/qubit-control/environment/README.md) | 6: two spins experience one fluctuating field; the shortcut treats their noise independently. | `QubitModel.fit(runs)` stores positive `.gamma` (s⁻¹); `predict(experiments)` returns shape `(N,)`. |
| [thermal-bodies](../tasks/thermal-bodies/environment/README.md) | 9: two gases share a moving piston inside a rigid vessel; the shortcut treats their common pressure as constant. | `ThermalModel.fit(runs)` stores positive `.conductance` (W/K); `predict(t, initial_temperature)` returns shape `(N, 2)`. |
| [reaction-diffusion](../tasks/reaction-diffusion/environment/README.md) | 6: two salts share a counterion; the shortcut extends separate binary-salt diffusion to a mixture. | `TransportModel.fit(data)` stores positive `.diffusivity` (m²/s); `predict(t, x, initial)` returns shape `(T, X, 2)`. |

All fit methods return `self`. The linked apparatus documents define input schemas, units, and preparation conventions. The transport directory retains its original name; the current system has no chemical reactions.

## Agent and author boundary

The agent receives the task instruction and the files copied by `environment/Dockerfile`: apparatus README, starter `model.py`, public tests, and calibration data. Docker builds only that environment context. Author documentation, `solution/`, `tests/`, and repository scripts are outside the agent workspace. Harbor supplies the private tests during verification after the agent finishes. Private calibration copies keep grading independent of edits to public data.

The shareable repository contains the answers for reproducibility. Run tasks through Harbor; do not give the solving agent the entire checkout or this guide. Edit the model freely within the stated API: the task asks for a complete model, not just the missing fit method.

## Grading and controls

Reward is 1 only when every public and private test passes. All tasks require reduced calibration χ² below 1.5 and relative fitted-parameter error below 5%. Reduced χ² divides the uncertainty-weighted squared residual sum by the number of scalar measurements minus one.

| Task | Additional private prediction requirement |
|---|---|
| Spin | Each of two groups has absolute probability RMSE below 0.03. This is not a relative percentage error. |
| Piston | Each of three cases has `norm(prediction − truth) / norm(truth − 293 K) < 0.05`, pooling times and both gases. |
| Electrolyte | Each of three cases has NRMSE below 0.05 for both species. Each species is normalized by its reference trajectory's departure from its initial spatial mean; the case score is the larger species error. |

The exact oracle must pass everything. Completed shortcuts must pass the API, calibration, and parameter checks, while failing hidden predictions. These controls are [spin](../scripts/qubit_noise_baseline.py), [piston](../scripts/thermal_piston_baseline.py), and [electrolyte](../scripts/reaction_baseline.py). A `nop` run leaves the fit unfinished and is not this control.

Classify an agent result as an intended physical failure only after inspecting its final source, public trajectory, and verifier metrics: the fit and API must work, and the retained physical assumption must explain the prediction error. Record code/math errors, timeouts, and infrastructure failures separately. Public tests passing alone does not establish physical correctness.

## Reproduction

Run from the repository root with Docker running and `uv` available. The task images pin Python 3.13, uv 0.12.5, NumPy 2.3.3, SciPy 1.16.3, and pytest 8.4.2. The [runner](../scripts/run_science.py) pins Harbor 0.21.0 and Codex CLI 0.154.0; agent/verifier limits are 600/60 seconds. Codex runs require authentication as described in the [setup README](../README.md).

For example, check both piston controls, then run three agents concurrently:

```bash
python3 scripts/run_science.py tasks/thermal-bodies --agent oracle --trials 1 --concurrency 1
python3 scripts/run_science.py tasks/thermal-bodies --agent oracle --solution-model scripts/thermal_piston_baseline.py --trials 1 --concurrency 1
python3 scripts/run_science.py tasks/thermal-bodies --model gpt-5.6-luna --reasoning-effort high --trials 3 --concurrency 3
```

Substitute either other task and its shortcut path; `--model` selects another available Codex model. Each run creates a fresh job containing its frozen task, logs, metrics, final files, and diffs.

Run the independent-reference and 256-noise-realization checks without Harbor:

```bash
validate() {
  OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 \
    --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 "$@"
}
validate scripts/validate_qubit_noise.py
validate scripts/validate_thermal_piston.py
validate scripts/validate_reaction.py
```

These commands write validation reports under `jobs/`; they need no existing results. Repeating a validator replaces its reports; Harbor batches use separate fresh job directories. Use the checked-in calibration data. Data-regeneration options are for intentional benchmark changes, not ordinary reproduction.
