# Counterion cloud, revision 1

The dilute limit of a neutral cylindrical Poisson–Boltzmann cell changes
when a nonzero counterion layer remains near the charged rod. Weak-charge
calibration identifies the bare line density but does not test this layer.

Let lambda=u*tau0 and xi=lB*lambda. At finite log-radius L=log(R/a), set
w=log(4*pi*lB*r^2*n(r)). The PB equation and electric boundary conditions
give w''=exp(w), w'(0)=2(1-xi), w'(L)=2. Its first integral is
C=w'^2-2 exp(w). The outer pressure ratio is
(1-C/4)*(1-exp(-2L))/(2xi). Equivalently, the exact contact identity is

    phi_R=(1-a^2/R^2)*(1-xi/2+pi*a^2*n(a)/lambda).

At fixed positive a, canonical finite-cell equilibrium followed by
R/a -> infinity gives phi=1-xi/2 for xi<=1 and phi=1/(2xi) for xi>=1.
The inner contact density tends to zero only in the first regime. Above
threshold, a^2*n(a)=(xi-1)^2/(2*pi*lB) in the limit. This is the stated
mean-field model; it does not claim a finite-concentration or correlated
ion prediction.

The completed shortcut drops the inner contact term at all line charges.
It is a consistent all-ions-dilute-away pressure closure, not an exact
supercritical PB solution. Its continued response 1-xi/2 remains positive
throughout the public domain. Its relative error above threshold is
(xi-1)^2. The actual pressure is higher than this linear continuation.

The true tau0 is 1.3 nm^-1. Calibration has 24 preparations, ten independent
replicates each, sigma .001, seed 618001. All calibration xi<=.7 over the
entire allowed parameter interval, so weighted linear regression uniquely
identifies tau0. Hidden groups have xi from 1.36864 through 1.625, plus
shared subcritical anchors. Radius changes check the prescribed limit.
Prediction error is group RMS error divided by truth RMS; the gate is 4%.
Fitted oracle error is below .04%, while the shortcut errors are 17.15%,
25.10%, and 33.35%. The 256 noise checks use seed 618009 and retain the gap.

The private reference solves the finite-cell first integral and extrapolates
pressures from L=800,1600,3200,6400 using inverse-log powers 2,3,4. It never
uses either limiting response branch. Refinement doubles all four L.
Independent collocation solves the PB boundary-value problem at L=12 and
checks neutrality. Validation explicitly covers xi=.999,1,1.001 because
convergence is nonuniform near the threshold. The largest tested limit
error is below 1e-8, well below the grading tolerance. Eighty random and
boundary cases and 41 noiseless parameter recoveries also pass.

Public files state the canonical cell, the order of limits, dielectric,
ion valence, mean-field approximation, normalization and API. The diagnosis,
solution, hidden cases and hint stay outside the ordinary agent image.
The concept review is in results/expansion-cylindrical-counterion-concept-review.md.

Physics source: Deserno and Holm, *Cell model and Poisson-Boltzmann theory:
A brief introduction*, Sections 2 and 4,
https://arxiv.org/abs/cond-mat/0112096.

Validate from the repository root:

```bash
UV_CACHE_DIR=/tmp/physlite-uv-cache uv run --offline --no-project --python 3.13 \
  --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 \
  python scripts/validate_counterion_cloud.py
python3 scripts/run_science.py tasks/counterion-cloud --agent oracle --trials 1
python3 scripts/run_science.py tasks/counterion-cloud --agent oracle --trials 1 \
  --solution-model scripts/counterion_cloud_baseline.py
```

Data regeneration requires the validator's explicit `--generate` flag.
Freeze all task files and run exactly three gpt-5.6-luna High plain trials,
concurrency three, built-in web search disabled. Shell networking remains
available. Review source and trajectories before classifying any failure.
