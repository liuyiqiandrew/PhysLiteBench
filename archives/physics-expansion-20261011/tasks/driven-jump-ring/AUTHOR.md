# Driven jump ring, revision 1

This candidate tests correlations between the counts in consecutive periods
of a continuously operating molecular ring. The public three-state jump law
obeys local detailed balance: on bond `i`, the forward/backward rate ratio is
`exp(E_i-E_(i+1)+F/3)`, and the product of ratios around the ring is `exp(F)`.
The bath thermal energy is one. Stage switches do not cause a jump or reset.
The counter gives every clockwise jump weight `1/3` and every reverse jump
weight `-1/3`. The observable is the long-time cumulant-generating function
per complete period, not the generating function of a single period.

## Physical models

For a column probability vector, the tilted generator retains the physical
escape rates on its diagonal and multiplies off-diagonal rates by
`exp(chi/3)` or `exp(-chi/3)`. Let `M_chi` be the chronological product of
the stage propagators `exp(duration*L_chi)`.

The correct asymptotic answer is `log(rho(M_chi))`, where `rho` is the positive
Perron eigenvalue. Conditioning on unusually large or small accumulated
counts changes the distribution of states at period boundaries. The state
retained at one boundary also influences the counts in later periods.

The completed approximation computes the exact single-period propagator and
the exact physical periodic stationary state `p`, but gives
`log(ones @ M_chi @ p)`. This is the correct long-time result for an apparatus
that independently redraws its state from `p` at each period boundary. It
retains all within-period dynamics and gives the right mean current. The
specified apparatus does not perform that renewal, and its fluctuation
statistics differ. The approximation is a complete stochastic model with
nonzero predictions.

## Calibration and references

Calibration uses spatially uniform energies and barriers within each stage.
The clockwise and reverse rates, `a` and `b`, then do not depend on the
occupied state. The count has the independent Poisson-difference law

```
psi(chi) = sum_stages duration *
    (a*(exp(chi/3)-1) + b*(exp(-chi/3)-1)).
```

Both controls agree exactly throughout the allowed attempt-rate interval.
This expression is linear in the one unknown attempt rate, so the weighted
least-squares fit is unique. There are 120 readings at 24 settings, with
independent fixed sigma `0.00015` and seed 611103. The true rate is 1.1.

The independent reference conditions on a Poisson number of attempted jumps
in each stage. Its discrete step includes physical null events and weighted
clockwise/reverse jumps. It sums the positive series, bounds its remaining
tail using the matrix one-norm, then propagates normalized weighted states
through many complete periods. The limiting single-period mass growth gives
the answer. It uses neither a matrix exponential nor an eigenvalue solver.
Refinement changes both the series budget and convergence tolerance, and
starts from a different initial state.

The oracle and reference are checked on 72 random domain/parameter cases.
Additional checks cover zero counting field, cycle-origin invariance,
equilibrium detailed balance, and the static fluctuation symmetry
`psi(chi)=psi(-F-chi)`. Hidden reference values are frozen for fast grading;
ordinary validation recomputes them. Hidden preparations vary period length,
barrier modulation, static heterogeneity and protocol order. These are all
within the public domain.

Calibration requires reduced chi-square below 1.5; parameter recovery requires
relative error below 3%. Each hidden group uses RMSE divided by the true group
RMS, below 4%. Across 256 independent calibration-noise realizations (seed
611107), the validator checks parameter recovery and all hidden group errors.
The exact scalar fit is used for these repetitions after validating both
control implementations against the analytical calibration law.

Run:

```
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 \
  --with pytest==8.4.2 python scripts/validate_driven_jump_ring.py
```

`--generate` intentionally replaces both calibration copies and frozen hidden
values. Validation is recorded in
`results/expansion-driven-jump-ring-r1-validation.json`.

## Evaluation

The target is exactly three frozen, unhinted `gpt-5.6-luna` High trials, with
the standard 600/60-second agent/verifier limits. Retain every outcome. Only
reviewed physical failures qualify: assuming renewed periods or using the
ordinary stationary distribution for the biased path ensemble is a physical
mistake. Reversing matrix order, tilting the escape diagonal, confusing counts
per time with counts per period, or an eigensolver bug must be diagnosed
separately. A same-parameter repair can test causality when appropriate.

The qualified exclusion-ring task concerns interacting density fluctuations;
this task has one molecule and only three states. The issue is continuity
across counting periods. It does not isolate the adiabatic geometric term.
Primary context for driven counting statistics is
[Sinitsyn and Nemenman](https://arxiv.org/abs/q-bio/0612018). This finite-stage
apparatus and independent-cycle comparison are defined and derived here.
