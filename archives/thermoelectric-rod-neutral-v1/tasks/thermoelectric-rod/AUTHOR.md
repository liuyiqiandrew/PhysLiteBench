# Thermoelectric rod: revision 1

Zero-current temperature relaxation identifies thermal conductivity, and
terminal voltage measures the integrated Seebeck response. These observations
do not measure the heat carried by a nonzero electrical current. The starter
uses the Seebeck coefficient correctly in voltage, solves Fourier conduction
and Joule heating correctly, and omits reversible Thomson heating.

For the specified reciprocal material, Pi=T*S. With the public electric-field
convention, local energy conservation gives

    q = T*S(T)*J - k*T_x,
    c*T_t = -q_x + J*E = k*T_xx + rho*J² - T*S'(T)*J*T_x.

The two terms proportional to S*J*T_x cancel. Fixed end temperatures make
contact Peltier heat part of the reservoir exchange; they do not remove bulk
Thomson heat. The voltage integral depends only on the two end temperatures,
so correct voltage predictions alone cannot detect the missing heat term.
See the constitutive and energy equations in
[The Thomson effect and the ideal equation on thermoelectric coolers](https://www.sciencedirect.com/science/article/abs/pii/S0360544213003629).
The task specifies an ideal material and uses generated measurements.

Calibration uses k=1.25 W/(m K), seed 9401, temperature uncertainty .04 K,
and voltage uncertainty 2e-5 V. The independent reference evolves conservative
heat-flux divergence plus electrical work on an eightfold refined mesh.
Both completed models give exactly the same zero-current predictions and fit
k=1.25208180, chi-square .98352. Hidden errors are temperature RMSE divided by
the RMS departure from the linear reservoir-temperature profile; the limit
is .04. Oracle errors are .00078–.00139, and shortcut errors .51–3.76. Both
models predict terminal voltage correctly, isolating the thermal failure.

The oracle evolves the explicit Thomson equation on the requested grid.
The independent reference keeps flux and work terms separate and uses fourfold
refinement. Eightfold refinement changes temperature by at most .000140 K;
the coarse oracle agrees with the independent energy form within .00271 K.
Reversing the current and spatial direction reverses temperature profiles
within 2.3e-13 K. Zero-current predictions coincide exactly.

All 256 calibration-noise trials (seed 19401) pass calibration and parameter
checks; maximum conductivity error is .391%. Hidden predictions at both fitted
parameter extrema give oracle error below .00330 and shortcut error above .509.
These are extrema sensitivity checks, not full hidden Monte Carlo evaluation.
Ordinary `scripts/validate_thermoelectric_rod.py` preserves data; `--generate`
updates the identical public and private copies intentionally.

An independent teammate reviewed the constitutive assumptions, signs, reservoir
boundary condition, source models, and validation evidence before Harbor.

## Reviewed evaluation

The frozen revision scores 0/3 unhinted and 3/3 hinted on GPT-5.6 Luna high, Codex CLI 0.154.0 and Harbor 0.21.0, with 600/60-second agent/verifier limits. All three ordinary failures pass calibration and parameter recovery and are reviewed physical-model failures. Fixed-parameter repairs recover the correct hidden predictions. See `results/candidates.json` and `results/hardening-status.json` for job IDs, per-trial reviews and matched-source checks. These are development-batch outcomes, not a future failure-rate guarantee.
