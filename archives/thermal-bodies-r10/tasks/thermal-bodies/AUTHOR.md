# Thermal bodies, revision 10

The previous revision passed all three trials after the approved neutral
instruction change. Its complete task, controls, validator and three trial
reviews are preserved in `archives/thermal-bodies-neutral-v1/`. Revision 10 also passed all three unhinted Luna-high trials under the neutral
instruction (initial yC5xMcU, follow-ups PsoeaXV and rB44UDF). Each submission
derived the missing nonlinear pressure-work factor; these are genuine physics
passes. Reviews are in `results/neutrality-brownian-trial-reviews.json`, with
controller provenance `jobs/matrix-neutral-r2-20261002-175653/index.json`.

This revision supplies the correct heat capacities for both linear temperature
modes. Its remaining approximation treats those modes independently. To resolve
the missing nonlinear work term, the apparatus now has a known inter-gas link
of 5e-5 W/K and allows initial temperatures from 80 to 1200 K. The README
explicitly defines an ideal monatomic gas with constant heat capacities and no
phase or internal-state changes over this range, and retains rapid mechanical
and internal equilibration. These are physical model changes, not tighter
prediction tolerances. The approved neutral instruction, API, 5% hidden
threshold, public tests, private test assertions and runtime limits are unchanged.

Let S=T1+T2 and Δ=T1−T2. Pressure balance and fixed total volume imply
Vi=V*Ti/S. Applying the first law separately to each chamber gives

```
n Cp dTi/dt − n R (Ti/S) dS/dt = Qi,
n Cv dS/dt = −g(S−2 Tb),
dΔ/dt = −(g+2 K) Δ/(n Cp) + (R/Cp)(dS/dt) Δ/S.
```

The starter solves the exact S equation and drops only the final coupling in
the Δ equation. It is the consistent linearization about equal bath-temperature
gases and conserves the correct total gas energy. The full finite-amplitude
solution multiplies the independent difference-mode exponential by
`(S(t)/S(0))^(R/Cp)`.

Calibration still prepares S(0)=2Tb in every run, so the nonlinear work term
vanishes exactly and both models have the same identifiable conductance.
Calibration was deliberately regenerated for the changed fixed link with seed
20260921, using 121 observations from 0 to 600 s per run. The independent
instrument uncertainty is a fixed 0.03 K for every temperature and time, the
same instrument value as before. It does not depend on the noiseless response.
Public and private measurements are identical copies.

The physical oracle integrates the coupled two-chamber capacity matrix. The
private reference instead uses the independent analytic sum/difference
solution. `scripts/validate_thermal_piston.py` checks both, the first law,
positive chamber volumes, total-volume conservation, and nonnegative entropy
production at the temperature-range corners and additional scored profiles.
It also runs 256 calibration-noise realizations with seed 1729.

Results are in `results/thermal-r10-validation.json`:

- Both controls pass calibration and parameter recovery in all 256 samples.
  The largest conductance error is 0.081%.
- The oracle passes every sample; its largest hidden NRMSE is 0.000519.
- The completed independent-mode control fails all three groups in every
  sample. Its smallest errors are 0.07282, 0.07088 and 0.11508, against 0.05.
- The capacity-matrix oracle and analytic reference agree within 1.15e-8 K.
  Chamber first-law residuals are below 2.50e-16 W; all checked temperatures
  and volumes are positive. Total volume is constant to 1.09e-19 m³.
- The unchanged local pytest harness gives eight passes for the oracle and
  five passes plus three hidden failures for the completed control. The
  unfinished fit in the starter is not counted as a physical failure.

`results/thermal-r10-source-provenance.json` records current source hashes and
byte identity of the instruction, grading assertions and runtime configuration.
Peer review is in `results/constraint-hardening-physics-review.json` under
`thermal-bodies-r10`. The parent coordinates fresh Docker controls and all
agent evaluations.
