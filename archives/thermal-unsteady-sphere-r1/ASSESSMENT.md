# Thermal unsteady sphere, revision 1

Status: approved and frozen after scientific validation, local controls and
independent final source review. No model evaluation has been run.

This is a thermal revision of the archived unsteady-sphere family. Both controls
have the complete unsteady Stokes response and particle inertia. Rich driven
calibration cannot distinguish a white point-force model from the equilibrium
fluid's force spectrum. The latter is checked independently by integrating
exterior-fluid strain noise and surface traction. The broader overlap with
fluid fluctuation-dissipation tasks is documented in the author and archive audit.

The 224 calibration readings contain 14 settings repeated 16 times, with fixed
instrument sigma .002 µm/s. The neutral instruction is unchanged. The source
has an ordinary unfinished fit and no default truth. Public prose defines the
two-sided angular-frequency velocity spectrum, equilibrium preparation and
finite-frequency continuum without a correction formula or solver restriction.

All 256 noise draws pass calibration, parameter recovery and oracle predictions;
every completed source fails all three diagnostic groups. Worst oracle error is
.000125; minimum source diagnostic error is .36320 against the .04 gate.
Fifty-six domain cases agree with the independent reference within 1.1e-13.
Local oracle passes 7/7; the source passes four checks and fails the three intended
thermal-spectrum checks. Each control completes in under half a second.

The original prototype is preserved in `prototype/`. Final validation, local
controls, history, archive audit and 18-file source provenance are under
`results/`. Final peer approval is
`results/physics-review-thermal-unsteady-sphere-r1.json`. Root owns frozen Docker
controls and model evaluation. Scientific
separation does not establish empirical difficulty.
