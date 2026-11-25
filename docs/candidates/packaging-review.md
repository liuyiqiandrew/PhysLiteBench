# Packaging and reporting audit

Reviewed 2026-10-01 and updated 2026-10-02, without rerunning agent batches or
updating evaluation results. This audit covers the original seven revision-1
additions and seven optional additions:

`hall-bar`, `moving-inductor`, `effusive-beam`, `real-gas-expansion`,
`induced-dipole`, `radiative-plates`, `magnetic-equilibrium`,
`spatial-diffusion`, `hydrodynamic-beads`, `poroelastic-solid`,
`competitive-adsorption`, `compensated-conductor`, `fluid-coexistence`, and
`first-order-magnet`. The last task was added after the initial thirteen-task
audit; its final source and data were audited separately before Harbor.

No task packaging blocker was found. The two reporting limitations identified
in the first audit have been fixed and rechecked. Completion of all requested Harbor
conditions and interpretation of agent failures remain separate requirements.

## Files and image boundary

Every task has its instruction, hint, Harbor configuration, public apparatus
README, starter, public tests and calibration, private tests and metadata,
independent reference, private calibration copy, complete oracle, and solution
installer. Each has a completed shortcut script. The first seven use
`validate_candidates.py`; each optional task has a standalone validator.

AST parsing succeeds for all task Python files. Imports are confined to the
standard library, installed NumPy/SciPy/pytest, the submitted `model`, and the
task-local reference loaded by the verifier. There are no imports from another
task, repository scripts, or scratch files, and no task symlinks.

All fourteen environment directories have exactly these intended public files,
apart from generated Python/test caches:

- `Dockerfile`
- `README.md`
- `model.py`
- `test_public.py`
- `data/calibration.json`

Dockerfiles copy those public files explicitly. They use the same pinned Python
3.13 base, uv 0.12.5, NumPy 2.3.3, SciPy 1.16.3, and pytest 8.4.2. Author
notes, hints, true-parameter metadata, solutions, and hidden tests are outside the
build context. A text scan found no public references to private paths, true
parameter fields, author notes, hints, or repository scripts.

All Harbor configurations retain `/app` artifacts, exclude caches and Git
metadata, and specify a 600-second agent limit and 60-second verifier limit.
The installers replace only `/app/model.py`. Verifier scripts write reward
`1` or `0` after pytest, including a failing pytest exit, and preserve separate
metrics. No task depends on another task's files at runtime.

## Calibration, interfaces, and scientific controls

Public and private calibration files are byte-identical for all fourteen tasks.
Their input records also match the current reference calibration generators
exactly, including the intended two independent repetitions. All listed
measurement standard deviations are positive.

The scalar-record `Model.fit(records)` / `predict(experiments)` API is consistent
across these tasks. Fitted attributes, output units, shapes, initial conditions,
and allowed preparation ranges are public. The adsorption task fits two
parameters; its calibration chi-square correctly uses two fitted degrees of
freedom. The others fit one parameter. Completed oracle and shortcut fit methods
are present; unfinished fits remain confined to starters.

The [seven-task physics review](physics-review.md) and
[six-backup review](backup-physics-review.md) independently inspect each oracle
against its separate reference, calibration equivalence, and discriminating
experiments. The fourteenth task has its own
[independent first-order-magnet review](../../results/first-order-independent-review.json).
Hidden cases test:

| Task | Hidden physical distinction | Scale and limit |
|---|---|---|
| Hall bar | Sidewall constraint changes longitudinal response at nonzero field | Group RMS relative current, 0.04 |
| Moving inductor | Core motion changes flux at fixed current | Group RMS relative current, 0.04 |
| Effusive beam | Outward normal velocities are flux weighted | Absolute CDF RMSE, 0.025 |
| Real-gas expansion | Conserved energy includes attraction energy | Per-observation cooling-relative RMS, 0.04 |
| Induced dipole | Polarization work changes mechanical force | Group RMS relative force, 0.04 |
| Radiative plates | Gray surfaces repeatedly reflect radiation | Group RMS relative flux, 0.04 |
| Magnetic equilibrium | Valid roots can be unstable thermodynamic states | Absolute magnetization RMSE, 0.025 |
| Spatial diffusion | Isothermal drag gradients preserve uniform equilibrium | Absolute moment RMSE, 0.03 |
| Hydrodynamic beads | Fluid-mediated response of the unshifted bead | Group RMS relative displacement, 0.04 |
| Poroelastic solid | Trapped fluid adds volumetric stiffness | Group RMS relative stress, 0.04 |
| Competitive adsorption | Shared sites change occupancy and displacement | Group RMS relative fraction, 0.04 |
| Compensated conductor | Imbalance prevents cancellation of the shared Hall field | Group RMS relative current, 0.04 |
| Fluid coexistence | Global equilibrium replaces the homogeneous pressure loop | Group RMS relative pressure, 0.04 |
| First-order magnet | A field-aligned local minimum can be metastable | Absolute magnetization RMSE, 0.025 |

All scale denominators are nonzero on their actual hidden groups. In particular,
expansion is normalized by nonzero cooling rather than absolute temperature,
and the initially uniform diffusion case uses an absolute scale. Hydro's
unshifted-bead response is small but nonzero; independent SI integration resolves
it far more accurately than the grading threshold.

The recorded local scientific controls pass calibration and parameter recovery.
Oracles pass hidden thresholds; completed shortcuts fail every hidden group.
The validators check independent physical limits and references, not simply
copies of the oracle formula. Numerical reference discrepancies are well below
thresholds, including the finite-volume convergence check for spatial diffusion
and the near-critical safeguard for fluid coexistence.

Measurement uncertainties and fitted-parameter sensitivity are also checked.
Every task has a recorded 256-realization calibration-noise validation. The
first seven, spatial diffusion, hydrodynamic beads, and fluid coexistence then
check hidden predictions at observed fitted-parameter extrema. Poroelasticity,
adsorption, the compensated conductor, and the first-order magnet evaluate every
hidden group for both fitted controls on every realization. Across those checks, the largest oracle
hidden error is approximately 0.01772 for fluid coexistence, below its 0.04
threshold; the shortcut separation remains. These checks establish margins on
the tested uncertainty realizations, not exhaustive global error bounds.

The first-order magnet also checks physical validity of the completed shortcut
on all 256 fitted realizations: the small root remains field aligned, has
residual below 5e-17 and curvature above 0.1105, yet its free-energy excess is
at least 0.00965. All competing minima remain at least 0.00571 above equilibrium.
The public preparation excludes phase-transition lines, avoiding ambiguity at
exact coexistence. Final-source local pytest controls yield seven oracle passes
and four shortcut passes with exactly three hidden failures. Its data and source
were frozen before Harbor; no claim about agent difficulty follows from this.

Ordinary validators read checked-in calibration and write reports under `jobs`.
Calibration regeneration is gated by `--generate` and updates public and private
copies together using recorded seeds. It is not an implicit part of validation.

## Runner and report audit

`run_science.py` first copies a task into a temporary directory. With `--hint`,
it changes only that copy's instruction by appending the exact author hint,
prefixed by `Physics hint:`. Environment, tests, and configuration are untouched.
The matrix runner requests `gpt-5.6-luna` at `high` effort in both agent
conditions, with three trials and concurrency three; both controls use one
oracle-agent trial. The same Codex CLI 0.154.0 and Harbor 0.21.0 are pinned.
Frozen tasks, final instructions, timing metadata, final files, and diffs are
retained in fresh job directories.

The matrix runner rejects nonzero process exits, Harbor errors, unexpected trial
counts, incorrect oracle/shortcut aggregate rewards, and failed control
calibration or parameter checks. The metric lookup handles both single-parameter
and maximum multi-parameter error fields. It does not classify agent failures;
the separate source/trajectory review is necessary.

The final summarizer was run read-only with output directed to a temporary file,
without changing `results`. Its output matches the published
`results/candidates.json`: 56 completed batches, 112 retained trials, and
14 matched plain/hint pairs. These contain 14 passing oracle controls,
14 completed shortcuts that pass calibration and parameter recovery but fail
every hidden group, and 84 reviewed agent trials. Plain agents pass 38/42;
hinted agents pass 42/42. Reviews classify the four failures as three physical
model failures and one implementation failure; no reviews remain pending.
There are no infrastructure errors. All fourteen dashboard rows match the
retained trial rewards and classifications.

The independent final audit reconciled every trial count, reward, exception,
verifier metric, available review hash, and timing condition. Every pair has
identical environment and private tests, the exact appended hint, matching
native Luna/high contexts and CLI 0.154.0, and identical `task.toml` files
with 600/60-second limits. Current environment, tests, solution, instruction,
hint, and configuration files match each task's evaluated plain snapshot;
all fourteen current public/private calibration copies are byte-identical.

The initial audit identified two report limitations; both are now resolved:

1. `summarize_candidates.py` now records the `task.toml` hash and parses both
   timeout values. Its paired checks require identical configuration and the
   expected 600/60-second limits.
2. The summary now requires exactly three completed, exception-free, binary-reward
   trials in each condition. Native model/effort and CLI checks are gated by this
   completeness check, preventing vacuous success when trial files are absent.

All fourteen completed paired comparisons pass these checks along with identical
environment/tests and the exact appended hint. The full fourteen-task evaluation
is complete. No candidate achieved 0/3 plain passes; completion does not establish
the requested difficulty target.

Neither script silently turns an agent modeling failure into a pass. The
summarizer attaches separate review classifications and preserves exceptions.
It should continue to be treated as an evidence collector, not as proof that the
user's requested difficulty target has been achieved.
