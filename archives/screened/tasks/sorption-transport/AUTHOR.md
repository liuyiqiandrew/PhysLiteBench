# Sorption transport, revision 1

Pure-species fronts identify the membrane exchange coefficient. The completed shortcut solves both valid single-species Langmuir-retarded diffusion equations independently, fitting all calibration records. A shared finite site pool changes not just equilibrium occupancy but the full two-species storage Jacobian. A competitor gradient can release initially uniform dissolved A as it moves. Hidden tests use that effect and opposing fronts.

The oracle evolves dissolved concentrations using the inverse two-by-two storage Jacobian in every chamber. The independent reference evolves conserved dissolved-plus-bound amounts and inverts local site equilibrium at each right-hand-side evaluation. The chamber model is finite dimensional as physically specified; there is no mesh approximation or unresolved spatial PDE.

The prediction limit is 1.2% relative RMS in dissolved concentration; calibration noise is 0.0015 mmol/L. Noisy calibration has 192 independent observations. Validate with scripts/validate_sorption_transport.py; data regeneration requires --generate. All r1 controls must be checked before freezing or evaluating agents.
