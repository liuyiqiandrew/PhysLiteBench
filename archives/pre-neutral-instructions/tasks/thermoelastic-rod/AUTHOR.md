# Thermoelastic rod, revision 2

The error is treating fixed total length as a local constant-stress condition.
Revision 1 used uniform expansion and gave plain 2/3, hinted 3/3. Its complete
sources, controls, validation, and reviews are in archives/thermoelastic-rod-r1;
its original jobs remain unchanged. Revision 2 uses known spatial expansion
and a calibration that observes only an antisymmetric thermal subspace.

Write alpha(x)=.002[1+.65 cos(2 pi x/L)]. Local linear thermodynamics gives

    ds = (c_e/T0) dtheta + E alpha(x) depsilon,
    sigma = E[epsilon-alpha(x) theta].

Uniform reaction stress and zero mean strain imply

    epsilon = alpha(x) theta - mean(alpha theta),
    c_sigma(x) theta_dot - T0 E alpha(x) mean(alpha theta_dot) = k theta_xx,
    c_sigma(x) = c_e + T0 E alpha(x)^2.

The completed shortcut solves the spatially varying local c_sigma problem,
including all its modal mixing, but omits the weighted nonlocal term. This is
a constitutive approximation, not a numerical shortcut. For any temperature
profile f, the exact capacity quadratic form is

    c_e mean(f^2) + T0 E [mean(alpha^2 f^2)-mean(alpha f)^2] >= c_e mean(f^2).

Thus the constrained material is stable. Alpha and c_sigma are even about the
midpoint. Insulated odd initial temperature remains odd, so mean(alpha theta)
vanishes throughout every calibration run. Both controls are exactly equivalent
on this restricted observation, and the decay identifies conductivity. Hidden
finite-body contacts break parity; a separate insulated even preparation probes
the same coupling without a contact or changing the total heat.

The oracle uses 64 normalized cosine modes. It integrates the local heat
capacity matrix and subtracts T0 E V times the outer product of the expansion
moments. A generalized symmetric eigensystem evolves this positive capacity
matrix with Fourier conduction and a contact port. The independent reference
evolves cell entropy and body energy, solving local temperature, strain, and
uniform stress from a separate constitutive system. It uses finite-volume heat
fluxes with half-cell conduction in series with the contact.

The conserved linear energy is

    V [mean(c_sigma theta)-T0 E mean(alpha) mean(alpha theta)] + C_b theta_b.

The uniform rod capacity is V[c_e+T0 E Var(alpha)]=7.535 J/K. It is neither
the spatial average of c_sigma nor the uniform-expansion revision's 5 J/K.
Conduction and contact entropy production are second order in temperature
departures, consistently with the stated linear thermoelastic balance.

All outputs are departures in K. Hidden groups use absolute RMSE below .025 K,
so nearly zero moments do not produce a normalization artifact. Departures at
most 1 K keep the ideal large expansion coefficient within small strain.
Calibration seed 9313 uses k=145 W/(m K), 160 independent readings, and sigma
.00147268 K. Noise seed 19313 generates 256 additional realizations. Both
checked-in fits give k=144.80125 and reduced chi-square .92306. Oracle hidden
RMSE is at most .000586 K; shortcut errors are .05486 to .10040 K.

Refining the independent reference from 128 to 256 cells changes outputs by
at most 2.18e-5 K. Oracle/reference maximum pointwise error is .001322 K,
less than one measurement sigma and far below the hidden threshold. Closed
energy error is below 3e-12 J and the predicted final equilibrium error below
9e-12 K. Calibration shortcut discrepancy is 5.6e-17 K, and its weighted
strain stays below 4.7e-20. The smallest capacity eigenvalue is 5 J/K.

All 256 calibration fits pass; maximum relative parameter error is .549%.
Hidden evaluation at the observed fitted-parameter extrema gives oracle RMSE
below .000766 K and shortcut RMSE above .05470 K. These are 256 calibration
noise fits followed by two extrema sensitivity checks, not 256 full hidden
runs or an exhaustive noise bound. Local verifier controls give oracle 7/7;
the shortcut passes four interface/calibration/parameter checks and fails
all three hidden groups.

Run scripts/validate_thermoelastic_rod.py for ordinary frozen-data validation;
--generate intentionally replaces both calibration copies. The report is
jobs/thermoelastic-rod-validation/summary.json and is copied into
results/constraint-hardening-science.json. A teammate reviewed the spatial
constitutive law, weighted constraint, initial entropy, independent reference,
calibration parity, and units. Revision-2 agent evaluation is complete: 0/3 unhinted and 3/3 hinted, with all unhinted failures reviewed as physical.

## Reviewed evaluation

The frozen revision scores 0/3 unhinted and 3/3 hinted on GPT-5.6 Luna high, Codex CLI 0.154.0 and Harbor 0.21.0, with 600/60-second agent/verifier limits. All three ordinary failures pass calibration and parameter recovery and are reviewed physical-model failures. Fixed-parameter repairs recover the correct hidden predictions. See `results/candidates.json` and `results/hardening-status.json` for job IDs, per-trial reviews and matched-source checks. These are development-batch outcomes, not a future failure-rate guarantee.
