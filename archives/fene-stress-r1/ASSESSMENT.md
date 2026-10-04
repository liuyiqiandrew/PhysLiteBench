# FENE stress, revision 1

Frozen after scientific validation, isolated completed controls and independent
final peer approval. No model evaluations have been run; root alone schedules
Docker controls and fresh unhinted trials. The final review is
`results/physics-review-fene-stress-r1.json`.

The completed source uses a mathematically valid Peterlin conformation closure.
It fits signed, stable Hookean rheometry exactly and uses all finite-spring
controls. The actual nonlinear molecular-force distribution gives different
mechanical tensile stress. The microscopic bead force/noise, stationary flow,
finite-extension boundary and stress-per-molecule normalization are public.

All 256 noisy fits pass for both controls. The correct model passes predictions;
the source fails the three finite-extension groups with errors .580–.643 at the
.04 gate. Independent direct Cartesian force integration agrees within 6.31e-8
over 56 domain checks. Local controls give 7/7 versus 4 passes and 3 intended failures,
both under .41 seconds.

The stage contains 18 source/control files, scientific/local reports and preserved
prototype history. An initial outside-domain limit report contains a documented
NaN from Jacobi normalization overflow; the final prototype and scientific
report are finite and pass. This old exploratory report is not task data or a
grading reference. Canonical tasks and shared documentation were not changed.
