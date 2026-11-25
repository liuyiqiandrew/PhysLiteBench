# Plasma compression, revision 1

The isotropic adiabatic shortcut uses the correct density, frozen magnetic flux and signed Maxwell momentum stress. It assumes continued ion pressure isotropy. The collisionless, gyro- and bounce-adiabatic apparatus instead preserves separate perpendicular and parallel velocity actions. Thus p_perp=p0/(a^4*b), p_parallel=p0/(a^2*b^3). When a=b both exactly equal p0*(n/n0)^(5/3), so isotropic calibration identifies p0 without distinguishing the closures.

The end pistons supply a concrete longitudinal bounce invariant. Ideal induction alone does not establish longitudinal particle compression. External actuators prescribe geometry and may do work; mechanical force equilibrium is not imposed. A virtual bulk-rest-frame surface measures the momentum tensor, not an unsupported physical-wall force. The magnetic contribution is B²/2−B_normal², with positive compressive sign.

The independent reference integrates normal velocity squared over the initial three-dimensional Maxwellian, preserving each sample's magnetic moment and longitudinal bounce action. It computes magnetic stress from vector contractions. The oracle uses CGL moment invariants directly. The validation also checks the complete stretch/pressure domain against kinetic mirror and firehose thresholds, not only the initial beta.

This tests the pressure-tensor consequence of magnetic and bounce actions. It shares the conserved-actions theme with the archived harmonic collisionless-trap candidate; it should not be represented as an unrelated statistical mechanism if both were ever retained.

Primary theory: [Webb et al., Action principles and conservation laws for Chew–Goldberger–Low anisotropic plasmas (2022)](https://arxiv.org/abs/2202.06399). The fixed reduced magnetic permeability is 1; no electromagnetic material response is fitted.

Calibration seed 139845 and 256 independent noise draws with seed 537822 give fitted p0=0.055003412 and reduced chi-square 1.04122. Measurement sigma is 0.0003; the hidden RMS-relative-error limit is 0.015. Across all noise draws, the oracle error is below 0.000970 and the shortcut error exceeds 0.03271. Kinetic quadrature agreement and pressure-work checks are below 4e-16. The maximum kinetic mirror and firehose indices over the allowed domain are 0.30142 and 0.27989, respectively, against thresholds of 1. Actual local controls give seven passes for the oracle and four passes plus three hidden failures for the shortcut.

Run `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_plasma_compression.py`. Data regeneration requires `--generate`. The shareable validation report is results/plasma-compression-validation.json.
