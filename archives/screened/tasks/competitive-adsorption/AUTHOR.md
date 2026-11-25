# Competitive-adsorption candidate, revision 1

This author file is outside the Docker build context. Agent difficulty has not
yet been measured. Target: three GPT-5.6 Luna high-effort trials without a hint,
then three with `hint.md` appended under the same agent configuration.

The public apparatus has one population of equivalent sites, one molecule per
site, no lateral interactions, and constant reservoir concentrations. Two
unknown affinities are separately identified by pure-species calibration.

Mass action gives theta_A=K_A*c_A*theta_empty and
theta_B=K_B*c_B*theta_empty. One shared normalization produces denominator
1+K_A*c_A+K_B*c_B. The shortcut instead applies a separate Langmuir denominator
to each species, which is exact on all calibration records. It consequently
predicts zero displacement when the competing species is introduced. The
oracle normalizes the three statistical weights; the reference solves the
three mass-action/normalization equations as a linear system.

Hidden concentrations remain moderate: even the shortcut's combined occupancy
stays below 0.858, so it satisfies a simple occupancy-bound check. Grading covers
both species' equilibrium occupancy and the loss in occupancy after addition
of the competitor. The latter is explicitly a fraction of all sites, not the
fraction of previously occupied sites lost.

The verifier uses private calibration and requires reduced chi-squared below
1.5 (192 scalar records minus two fitted parameters), each affinity within 3%,
and each hidden case's RMSE divided by reference RMS below 0.04. Measurement
sigma is 0.006 times the largest noiseless calibration fraction. Across 256
noise draws the oracle's maximum hidden error is 0.657%, while the shortcut's
minimum is 25.6%; displacement error is 100%. Calibration seed: 9311;
noise-validation seed: 19311.

Run from the repository root:

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_competitive_adsorption.py
python3 scripts/run_science.py tasks/competitive-adsorption --agent oracle --trials 1
python3 scripts/run_science.py tasks/competitive-adsorption --agent oracle --trials 1 --solution-model scripts/competitive_adsorption_baseline.py
```

The validator reads checked-in data unless `--generate` is explicit. It checks
both controls on all hidden cases for all 256 noisy calibrations; empty and
pure-species limits; equal-species symmetry; probability normalization;
competition monotonicity; and displacement as occupancy loss. Reports go to
`jobs/competitive-adsorption-validation/summary.json`. Harbor controls and agent
trials are separate evaluation steps managed by the parent task.
