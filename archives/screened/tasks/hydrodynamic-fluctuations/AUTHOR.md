# Hydrodynamic fluctuations, revision 1

This candidate hardens the hydrodynamic-beads idea. The starter already has the correct deterministic Oseen coupling. The missing physics is the matching correlation of thermal velocity increments, so fixing the mean drift alone cannot solve it. Original hydrodynamic-beads revisions and run artifacts are unchanged.

For K = 0.3 diag(factors), the mean evolves as m(t) = shift - exp(-MKt) shift. Mean-only calibration identifies the viscosity, which scales every entry of M inversely. The thermal covariance obeys

    dC/dt = -MK C - C KM + 2 kBT M,  C(0) = 0.

Its exact physical solution is C(t) = Ceq - E Ceq E^T, where Ceq = kBT K^-1 and E = exp(-MKt). The implementation diagonalizes the symmetric similarity transform sqrt(K) M sqrt(K) and integrates the covariance source in that basis. The shortcut uses the same drift and exact covariance integrator but replaces M by diag(M) only in the noise covariance. This is a valid independent-velocity-noise model, but it violates the common liquid's fluctuation-dissipation relation. Both controls fit the calibration identically.

The cross-covariance initially grows at rate 2 kBT M12 and tends to zero in the Gibbs state. The shortcut instead produces a negative cross-covariance with a nonzero stationary value. Hidden tests measure this finite-time quantity for equal and unequal traps. No unknown coupling or unadvertised interaction is introduced. Constant mobility, leading Oseen order, zero initial position covariance, a single equilibrated bath, and neglected inertia are all public.

The independent reference integrates mean and covariance force-balance equations using SI mobility and stiffness. The author checks also use equal-trap normal modes, Gibbs equilibrium, the initial diffusion slope, the uncoupled limit, and positivity. This is an idealized version of the coupled Brownian dynamics studied by [Berut et al.](https://arxiv.org/abs/1611.10219); the task is restricted to one equilibrium temperature.

Validation uses seed 9321 and 200 calibration records with Gaussian errors at 0.6% of the largest mean displacement. The private parameter tolerance is 3%; calibration reduced chi-square must be below 1.5. Hidden error is RMSE divided by the reference RMS for each group, with limit 0.04. This tolerance is well above fitted-parameter and numerical errors.

Checked-in calibration: fitted viscosity 0.0011014205 Pa s, relative error 0.129%, reduced chi-square 1.07371. Oracle hidden NRMSE is at most 0.001284; shortcut NRMSE is at least 5.32057. All 256 additional noise realizations pass calibration and parameter checks; maximum parameter error is 0.659%. At observed parameter extrema, oracle hidden NRMSE remains below 0.006566 and shortcut NRMSE stays above 5.31559. Independent ODE discrepancy is below 9.4e-15 square micrometers; Gibbs and normal-mode checks are below 7e-18.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 scripts/validate_hydrodynamic_fluctuations.py hydrodynamic-fluctuations
```

Add `--generate` only to intentionally regenerate both calibration copies. The report is `jobs/hydrodynamic-fluctuations-validation/summary.json`. Local isolated-copy pytest: physical oracle 7 passed; completed shortcut 4 passed and 3 hidden prediction tests failed. Harbor controls and paired gpt-5.6-luna/high 3-trial evaluation are coordinated separately; scientific validity does not establish agent difficulty.
