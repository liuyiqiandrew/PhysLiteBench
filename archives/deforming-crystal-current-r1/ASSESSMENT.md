# Deforming crystal current, revision1

The scientific package is independently approved and frozen. The final peer
report is attached under results/. No model or Docker evaluation has occurred.

The apparatus measures current into a shorted material electrode during slow
homogeneous deformation of a polar crystal. The complete source correctly
solves internal equilibrium and differentiates laboratory polarization density;
it interprets that derivative as lead current. The independent reference
relaxes the internal coordinates separately and solves grounded-slab Poisson
charge response. Exact longitudinal calibration identifies the stiffness;
ordinary transverse/mixed drives distinguish the two readouts.

There are288 calibration readings over18 settings, with fixed sigma=.001.
All256 noisy calibration checks pass as intended. Maximum oracle prediction
error is.000242 and minimum shortcut diagnostic error is.4888, against.04.
Across the full fitted interval every diagnostic group remains at least.375
apart; physical and source signals both remain positive and nonzero. The
unchanged longitudinal anchors pass both controls.

Validation includes all64 deformation-box corners at three stiffnesses and12
signed rate basis directions, plus64 interior controls:2368 responses. The
maximum independent reference discrepancy is8.62e−12. The internal free energy
is globally convex, and the charge-termination branch is safe over the full
box. Charge conservation, signed drive reversal, rate linearity and fixed-frame
covariance pass. Poisson mesh refinement is qualified: the integrated induced
charge is naturally mesh-exact when charge deposition preserves its first moment.

The local oracle passes7/7 tests in.625 seconds; the completed source passes
calibration, parameter and anchor checks and fails exactly three diagnostics
in.606 seconds. The full author validator took17.76 seconds. Every noise outcome
and all prototype alternatives are retained. The exact original prototype,
its author manifest and peer are preserved under `prototype/`.

The neutral instruction and600/60-second limits are unchanged. The public
starter has a None parameter and unfinished fit. The image receives only the
public apparatus, model, interface test and calibration data. The public text
contains no current conversion or numerical method restriction. Final source
and report hashes are in `results/deforming-crystal-current-r1-source-provenance.json`.
Empirical difficulty remains unmeasured.
