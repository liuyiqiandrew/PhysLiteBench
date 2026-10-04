# Benchmark author guide

These tasks test whether an agent checks a supplied physical model rather than only completing its parameter fit. Each starter contains a working predictor and an unfinished fit method. A completed shortcut fits the calibration but makes wrong predictions for other preparations. [TASKS.md](../TASKS.md) explains the physics and exact solutions.

For a new system, follow [Adding a physics task](ADDING_TASKS.md). The original three tasks retain their established revisions. Fourteen additional candidates start at revision 1; local scientific controls are recorded separately from Harbor and model-evaluation results.

## Current tasks and interfaces

| Task | Revision and physical assumption to repair | Interface |
|---|---|---|
| [qubit-control](../tasks/qubit-control/environment/README.md) | 6: two spins experience one fluctuating field; the shortcut treats their noise independently. | `QubitModel.fit(runs)` stores positive `.gamma` (s⁻¹); `predict(experiments)` returns shape `(N,)`. |
| [thermal-bodies](../tasks/thermal-bodies/environment/README.md) | 9: two gases share a moving piston inside a rigid vessel; the shortcut treats their common pressure as constant. | `ThermalModel.fit(runs)` stores positive `.conductance` (W/K); `predict(t, initial_temperature)` returns shape `(N, 2)`. |
| [reaction-diffusion](../tasks/reaction-diffusion/environment/README.md) | 6: two salts share a counterion; the shortcut extends separate binary-salt diffusion to a mixture. | `TransportModel.fit(data)` stores positive `.diffusivity` (m²/s); `predict(t, x, initial)` returns shape `(T, X, 2)`. |

All fit methods return `self`. The linked apparatus documents define input schemas, units, and preparation conventions. The transport directory retains its original name; the current system has no chemical reactions.

### Seven revision-1 candidates

These tasks share `Model.fit(records)` and `Model.predict(experiments)`.
Each calibration record has an `input` dictionary, a scalar measured `value`,
and its independent Gaussian standard deviation `sigma`. `fit` returns `self`
and stores the parameter below. `predict` returns a finite NumPy array of shape
`(len(experiments),)`. The apparatus links specify input ranges and units.

| Task | Fitted attribute | Prediction inputs and output |
|---|---|---|
| [hall-bar](../tasks/hall-bar/environment/README.md) | `.mobility`, m²/(V s) | `electric_field`, `magnetic_field` → signed longitudinal current density, A/m². |
| [moving-inductor](../tasks/moving-inductor/environment/README.md) | `.resistance`, ohms | `time`, `initial_current`, `voltage`, `x0`, `amplitude`, `omega` → current, A. Each experiment starts a fresh run at time zero. |
| [effusive-beam](../tasks/effusive-beam/environment/README.md) | `.temperature`, K | `component`, `threshold` in m/s → velocity-component CDF. |
| [real-gas-expansion](../tasks/real-gas-expansion/environment/README.md) | `.heat_capacity`, J/(mol K) | `protocol="heating"` with initial/final temperatures and volume → heat, J; `protocol="expansion"` with initial temperature and initial/final volumes → final temperature, K. |
| [induced-dipole](../tasks/induced-dipole/environment/README.md) | `.polarizability`, units of 10⁻²⁴ C m²/V | `observable`, `field_offset`, `gradient`, `position` → dipole in units of 10⁻²¹ C m, or force in units of 10⁻¹⁵ N. |
| [radiative-plates](../tasks/radiative-plates/environment/README.md) | `.emissivity`, dimensionless | `temperature_1`, `temperature_2`, `emissivity_2` → net flux leaving plate 1, W/m². |
| [magnetic-equilibrium](../tasks/magnetic-equilibrium/environment/README.md) | `.coupling`, the specified energy unit | `temperature`, nonzero `field` → magnetization per spin. Units have kB=1. |

### Optional backup interfaces

Seven additional revision-1 candidates use the same scalar-record
`Model.fit(records)` / `predict(experiments)` API. They are alternatives under
development, with independently reviewed local controls.

| Task | Parameters and observables | Hidden requirement |
|---|---|---|
| [spatial-diffusion](../tasks/spatial-diffusion/environment/README.md) | `.diffusivity` in micrometer²/s; `contrast`, `first`, `second`, `time`, `mode` → cosine moment. | Each of three contrast groups has absolute moment RMSE below 0.03. |
| [hydrodynamic-beads](../tasks/hydrodynamic-beads/environment/README.md) | `.stiffness` in pN/micrometer; `measurement`, `bead`, `separation`, `factors`, plus `shift` and `time` for dynamics → equilibrium variance in micrometer² or mean displacement in micrometers. | Each of three trap-shift groups has RMS-normalized error below 0.04. |
| [poroelastic-solid](../tasks/poroelastic-solid/environment/README.md) | `.young_modulus` in MPa; six tensor `strain` entries and requested `component` → total stress in MPa. | Each of three loading groups has RMS-normalized error below 0.04. |
| [competitive-adsorption](../tasks/competitive-adsorption/environment/README.md) | `.affinity_a` and `.affinity_b` in L/mmol; `protocol`, `species`, `concentration_a`, `concentration_b` → occupancy or displacement fraction. | Each of four species/protocol groups has RMS-normalized error below 0.04. |
| [compensated-conductor](../tasks/compensated-conductor/environment/README.md) | `.mobility` in m²/(V s); `electric_field`, `magnetic_field`, `density_positive`, `density_negative` → signed bulk longitudinal current density in A/m². | Each of three density-imbalance groups has RMS-normalized error below 0.04. |
| [fluid-coexistence](../tasks/fluid-coexistence/environment/README.md) | `.attraction` in the specified reduced units; `temperature` and mean molar `volume` → equilibrium pressure in reduced units. | Each of three subcritical isotherms has RMS-normalized error below 0.04. |
| [first-order-magnet](../tasks/first-order-magnet/environment/README.md) | `.coupling` in the specified energy unit; `temperature` and nonzero `field` → equilibrium magnetization per spin, with known four-spin coupling Q=1.2. | Each of three low-temperature groups has absolute magnetization RMSE below 0.025. |

For all except spatial diffusion and the first-order magnet, RMS normalization is `RMSE(prediction − truth)/RMS(truth)`
within each hidden group. Each fitted parameter must be within 3% and calibration
reduced chi-square below 1.5. Adsorption subtracts two fitted parameters when
computing reduced chi-square; every other task subtracts one.

These local controls are described in the
[backup physics review](candidates/backup-physics-review.md). Spatial diffusion
and hydrodynamic beads use 256 calibration-noise realizations followed by hidden
checks at the observed parameter extrema; fluid coexistence uses the same scope.
Poroelasticity, adsorption, and the compensated conductor evaluate both fitted
controls on all hidden groups for every one of their 256 noise realizations.
The first-order magnet does the same, additionally checking that the shortcut
remains a field-aligned local minimum separated in free energy from equilibrium.
Its [independent review](../results/first-order-independent-review.json) and
[author derivation](../tasks/first-order-magnet/AUTHOR.md) record those checks.
None of these procedures measures agent difficulty.

## Agent and author boundary

The agent receives the task instruction and the files copied by `environment/Dockerfile`: apparatus README, starter `model.py`, public tests, and calibration data. Docker builds only that environment context. Author documentation, `solution/`, `tests/`, and repository scripts are outside the agent workspace. Harbor supplies the private tests during verification after the agent finishes. Private calibration copies keep grading independent of edits to public data.

The shareable repository contains the answers for reproducibility. Run tasks through Harbor; do not give the solving agent the entire checkout or this guide. Edit the model freely within the stated API: the task asks for a complete model, not just the missing fit method.

For the new tasks, `hint.md` is author-owned and outside the image.
The runner's `--hint` condition appends its contents to the instruction in a
temporary task copy. It changes no starter code, calibration, dependencies,
private tests, time limit, model, or reasoning effort. The retained instruction,
frozen task, and `physics_hint` field in `run-timing.json` identify the condition.

## Grading and controls

Reward is 1 only when every public and private test passes. All tasks require reduced calibration χ² below 1.5. Relative fitted-parameter error must be below 5% for the original three tasks and below 3% for each fitted parameter of the revision-1 candidates. Reduced χ² divides the uncertainty-weighted squared residual sum by the number of scalar measurements minus the number of fitted parameters.

| Task | Additional private prediction requirement |
|---|---|
| Spin | Each of two groups has absolute probability RMSE below 0.03. This is not a relative percentage error. |
| Piston | Each of three cases has `norm(prediction − truth) / norm(truth − 293 K) < 0.05`, pooling times and both gases. |
| Electrolyte | Each of three cases has NRMSE below 0.05 for both species. Each species is normalized by its reference trajectory's departure from its initial spatial mean; the case score is the larger species error. |
| Hall bar | Each of three magnetic-field groups has `RMSE(prediction − truth) / RMS(truth) < 0.04`. |
| Moving inductor | Each of three moving-core groups has `RMSE(prediction − truth) / RMS(truth) < 0.04`. |
| Effusive beam | Each of three axial-threshold groups has absolute CDF RMSE below 0.025. |
| Real-gas expansion | Each of three expansion groups has `RMS((prediction − truth) / abs(truth − initial_temperature)) < 0.04`. All hidden expansions have nonzero cooling. |
| Induced dipole | Each of three field-gradient groups has `RMSE(prediction − truth) / RMS(truth) < 0.04`. |
| Radiative plates | Each of three gray-panel groups has `RMSE(prediction − truth) / RMS(truth) < 0.04`. |
| Magnetic equilibrium | Each of three low-temperature groups has absolute magnetization RMSE below 0.025. |

The exact oracle must pass everything. Completed shortcuts must pass the API, calibration, and parameter checks, while failing hidden predictions. These controls are [spin](../scripts/qubit_noise_baseline.py), [piston](../scripts/thermal_piston_baseline.py), and [electrolyte](../scripts/reaction_baseline.py). A `nop` run leaves the fit unfinished and is not this control.

Classify an agent result as an intended physical failure only after inspecting its final source, public trajectory, and verifier metrics: the fit and API must work, and the retained physical assumption must explain the prediction error. Record code/math errors, timeouts, and infrastructure failures separately. Public tests passing alone does not establish physical correctness.

The seven new shortcut controls are
[Hall](../scripts/hall_bar_baseline.py),
[inductor](../scripts/moving_inductor_baseline.py),
[effusion](../scripts/effusive_beam_baseline.py),
[gas](../scripts/real_gas_expansion_baseline.py),
[dipole](../scripts/induced_dipole_baseline.py),
[radiation](../scripts/radiative_plates_baseline.py), and
[magnet](../scripts/magnetic_equilibrium_baseline.py).
Their [local scientific validation](../results/candidate-validation.json)
records oracle passes and shortcut hidden failures with both controls passing
calibration and parameter recovery. This is local control evidence, not a Harbor
or model success-rate claim.

That validation uses 256 independent calibration-noise realizations per task.
Calibration-generation seeds are 9200–9206 in the table's task order; noise-check
seeds add 10000. Every sampled fit passed the calibration threshold, and the
largest parameter error was 1.24%. Both controls have the same calibration
predictor, so the validator fits the oracle on every realization and cross-checks
the shortcut fit on the first and last. It evaluates all hidden groups at the
smallest and largest fitted parameters observed across those 256 runs; oracle
and shortcut remain separated there. This is a parameter-sensitivity check,
not 256 complete hidden-prediction evaluations or an exhaustive parameter bound.
The [independent physics review](candidates/physics-review.md) covers derivations,
units, stable gas endpoints, flux conservation, and magnetic root residuals.

## Reproduction

Run from the repository root with Docker running and `uv` available. The task images pin Python 3.13, uv 0.12.5, NumPy 2.3.3, SciPy 1.16.3, and pytest 8.4.2. The [runner](../scripts/run_science.py) pins Harbor 0.21.0 and Codex CLI 0.154.0; agent/verifier limits are 600/60 seconds. Codex runs require authentication as described in the [setup README](../README.md).

For example, check both piston controls, then run three agents concurrently:

```bash
python3 scripts/run_science.py tasks/thermal-bodies --agent oracle --trials 1 --concurrency 1
python3 scripts/run_science.py tasks/thermal-bodies --agent oracle --solution-model scripts/thermal_piston_baseline.py --trials 1 --concurrency 1
python3 scripts/run_science.py tasks/thermal-bodies --model gpt-5.6-luna --reasoning-effort high --trials 3 --concurrency 3
```

Substitute another task and its shortcut path; `--model` selects another available Codex model. Each run creates a fresh job containing its frozen task, logs, metrics, final files, and diffs.

Run both Harbor controls for every new task with:

```bash
for candidate in hall-bar moving-inductor effusive-beam real-gas-expansion \
                 induced-dipole radiative-plates magnetic-equilibrium; do
  python3 scripts/run_science.py "tasks/$candidate" \
    --agent oracle --trials 1 --concurrency 1
  python3 scripts/run_science.py "tasks/$candidate" \
    --agent oracle --solution-model "scripts/${candidate//-/_}_baseline.py" \
    --trials 1 --concurrency 1
done
```

After controls pass, run each frozen candidate without and with its physics
hint under the same agent setup. For example:

```bash
python3 scripts/run_science.py tasks/hall-bar --model gpt-5.6-luna \
  --reasoning-effort high --trials 3 --concurrency 3
python3 scripts/run_science.py tasks/hall-bar --model gpt-5.6-luna \
  --reasoning-effort high --trials 3 --concurrency 3 --hint
```

The development target is ideally 0/3 unhinted and 3/3 hinted. It is not an
established result for these tasks. Preserve every exploratory batch and review
each final submission and trajectory; report the newest frozen confirmation
batch separately. The hint should supply the missing physics, not fitted values
or hidden cases.

Run the independent-reference and 256-noise-realization checks without Harbor:

```bash
validate() {
  OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 \
    --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 "$@"
}
validate scripts/validate_qubit_noise.py
validate scripts/validate_thermal_piston.py
validate scripts/validate_reaction.py
validate scripts/validate_candidates.py
```

These commands write validation reports under `jobs/`; they need no existing results. Repeating a validator replaces its reports; Harbor batches use separate fresh job directories. Use the checked-in calibration data. Data-regeneration options are for intentional benchmark changes, not ordinary reproduction.

`validate_candidates.py` checks the seven revision-1 tasks and defaults to
256 calibration-noise realizations. It writes
`jobs/candidate-validation/summary.json`; the reviewed shareable copy is
`results/candidate-validation.json`. Pass task names to select candidates, and
use `--generate` only for an intentional calibration change before freezing a
revision. Public and private calibration copies are regenerated together.

The seven optional backups have separate validators:

```bash
validate scripts/validate_spatial_diffusion.py
validate scripts/validate_hydrodynamic_beads.py
validate scripts/validate_poroelastic_solid.py
validate scripts/validate_competitive_adsorption.py
validate scripts/validate_compensated_conductor.py
validate scripts/validate_fluid_coexistence.py
validate scripts/validate_first_order_magnet.py
```

Their reports go to `jobs/<task>-validation/summary.json`. Run their Harbor
controls using the same loop above with task names `spatial-diffusion`,
`hydrodynamic-beads`, `poroelastic-solid`, `competitive-adsorption`,
`compensated-conductor`, `fluid-coexistence`, and `first-order-magnet`; their
completed shortcuts follow the same underscore-named script convention.
Each task also has an author-owned `hint.md` for the same instruction-only
comparison.
