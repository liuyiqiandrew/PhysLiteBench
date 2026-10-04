# Nematic wall torque, revision 1

Frozen after scientific validation, isolated completed controls and independent
final source/public-input/grading approval. No model evaluations have been run.
Root alone schedules Docker controls and fresh unhinted trials. The final review
is `results/physics-review-nematic-wall-torque-r1.json`.

The completed source solves the correct nonlinear director state but computes
anchoring work from bulk splay/bend energy. The full material energy includes a
divergence contribution that changes curved-wall work while leaving that state
unchanged. Flat-cell calibration identifies the same modulus in both models.
The readout is supplied work by distributed tilt controls, not net axial torque.

All 256 noisy calibrations pass for both controls. Correct predictions pass;
the source fails the three curved-wall groups with errors .612–.831. Independent
energy minimization and boundary virtual work agree within 5e-7 relative over
64 settings. Local controls give 7/7 versus four passes and three intended
prediction failures. The public physical preparation selects a stable smooth
branch without claiming uniqueness of every three-dimensional stationary state.

The stage preserves the reduced-unit prototype, records its unit rescaling and
initial supplementary-limit numerical fix, and includes 18 source/control files
with scientific and local reports. Canonical tasks and shared documentation were
not changed. See AUTHOR for physics, overlap review and causal classification.
