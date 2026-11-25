# Magnetic tracer, revision 2

Revision 1 passed all three trials after the neutral instruction change. Its
complete source, controls, validator, reports and three trial reviews are in
`archives/magnetic-tracer-neutral-v1/`. Revision 2 has not yet been evaluated
with an agent.

The new physical experiment measures the signed mechanical area accumulated by
a trapped underdamped charged tracer, then takes the controlled small-mass
limit. The public apparatus specifies the complete finite-mass dynamics,
initial Maxwell velocities independent of the prescribed position covariance,
and measurement-before-limit order. There is no stochastic-calculus convention
left implicit in the measured area. Position moments are also measured.

The completed shortcut already has the exact limiting position OU dynamics,
including full symmetric and antisymmetric Lorentz mobility. It then evaluates
the ordinary Stratonovich area of that position process. This is a coherent
coarse-grained measurement approximation: its position predictions are correct,
but a path observable involving fast velocities need not commute with the
small-mass limit.

With K=diag(kx,ky), let M=[[gamma,B],[-B,gamma]]/(gamma²+B²), C=<r r^T>, and
Y=<r v^T>. The limiting covariance evolves with drift -M K and stationary value
T K^-1. The finite-mass cross-covariance balance gives

```
Y -> (T I - C K) M^T,
d<area>/dt = Y_xy - Y_yx.
```

The T M^T contribution is -2BT/(gamma²+B²). The shortcut misses this term. In
thermal equilibrium its limiting position SDE predicts a positive signed area
rate +2BT/(gamma²+B²), although the finite-mass mechanical area has zero mean.
The hidden nonequilibrium preparations have nonzero physical areas, so the
solution is not a constant-zero predictor.

Calibration uses zero field, several trap/preparation settings, all three
position moments and tilted-preparation area measurements. Both models are
then exactly equivalent for every allowed friction; the relaxation curves
identify the common friction. Calibration was deliberately regenerated from
the independent finite-mass reference with seed 871023. Every scalar readout
has the same fixed instrumental sigma=0.0006, independent of the parameter.
Public and private copies match.

The independent reference evolves the full four-dimensional phase-space OU
process. It propagates the covariance increment relative to exact thermal
equilibrium and evaluates the integral of x*v_y-y*v_x with a four-dimensional
Lyapunov primitive. Only after evaluating the finite-mass observable does it
extrapolate three masses to zero. It does not use the area-correction formula.
The validator checks this primitive against direct covariance-plus-area ODE
integration, and checks mass refinement, equilibrium area, field-reversal
reflection, positive position covariances and calibration equivalence.

`results/magnetic-area-r2-validation.json` records the completed checks:

- Independent finite-mass predictions agree with the oracle within 2.92e-9;
  halving all masses reduces the maximum discrepancy to 3.63e-10.
- The finite-mass primitive agrees with direct ODE integration to 1.42e-13.
- All 256 noise realizations pass calibration and friction recovery; maximum
  parameter error is 0.225%. The oracle passes every sample, with maximum
  hidden error 0.000941. The shortcut fails every sample and every group;
  its smallest hidden error is 0.6083 against the unchanged 0.025 threshold.
- The original local test harness gives seven passes for the oracle and four
  passes plus three hidden failures for the completed shortcut.

The instruction, public API checks, private grading assertions, parameter
range, instrument sigma, prediction tolerance and runtime limits are unchanged.
The apparatus, readout inputs, preparations and calibration were intentionally
revised. Source provenance is in `results/magnetic-area-r2-source-provenance.json`.
Peer review is under `magnetic-tracer-r2` in
`results/constraint-hardening-physics-review.json`. The parent coordinates
fresh Docker controls and agent evaluations.

The neutral-instruction conditional screen is complete: oracle 1/1, completed
shortcut 0/1, and the initial Luna-high trial 0/1. Per the approved policy no
follow-ups were run after this initial failure; this is not a three-trial rate.
Index: `jobs/matrix-neutral-r2-20261002-181402/index.json`. Trial
`magnetic-tracer__Wfi3Bro` retained the positional Stratonovich area while
correctly fitting friction and predicting covariances. The public trajectory
checks only the zero-field calibration. At its fixed fitted friction
0.8004295129, subtracting the fast-velocity area term changes no calibration
prediction and reduces hidden errors 1.668/1.383/.609 to
.0000825/.0000497/.000225. This supports a physical singular-limit failure.
The audit and causal repair are saved in
`results/neutrality-brownian-trial-reviews.json` and
`results/magnetic-area-r2-Wfi3Bro-fixed-parameter-repair.json`; native artifacts
are unchanged.
