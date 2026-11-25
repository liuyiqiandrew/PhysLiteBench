# Magnetic transfer between Cartesian heat baths

Revision 1 is staged and unevaluated. No claim about Luna difficulty has been established. Root controls promotion and model evaluations.

The supplied model solves the complete finite-mass Gaussian mechanics, including the magnetic field, both thermal noise channels and all position-velocity correlations. It reports the power transferred through the harmonic potential in the selected Cartesian channel as that reservoir's calorimetric power. This is an exact channel calculation but an incorrect identification of the measured energy exchange. Calibration at zero magnetic field cannot distinguish the two.

Let Y=<x v^T>, V=<v v^T>, and K=[[k,h],[h,k]]. The bath-to-particle stationary powers are

    Q_x = gamma*(T_x/m - V_xx) = (K Y)_xx - B V_xy,
    Q_y = gamma*(T_y/m - V_yy) = (K Y)_yy + B V_xy.

The Lorentz term has zero total power but transfers kinetic energy between the two Cartesian components. A calorimeter surrounding each physical Langevin reservoir includes this exchange. The shortcut retains only diag(KY). It uses every apparatus input and preserves positive covariance, total heat balance, field-reversal symmetry and the heat-flow direction throughout the public domain. No numerical or fitting error is needed for its failure.

For these symmetric traps, an independent stationary moment reduction gives

    Q_x = gamma*(T_x-T_y)/(2m)
          * (B^2 + m*h^2/k)/(gamma^2+B^2+m*h^2/k),
    Q_x,shortcut = gamma*(T_x-T_y)*h^2
                   / [2k*(gamma^2+B^2+m*h^2/k)].

At B=0 both reduce to gamma*h^2*(T_x-T_y)/[2*(k*gamma^2+m*h^2)]. The calibration reads the hot reservoir, so all exact signals are positive and strictly decrease with gamma. Across the entire public range k*gamma^2-m*h^2 is at least .15. This proves global identifiability; the validator also checks 41 noiseless parameter values and a 401-point calibration loss scan.

The verifier does not solve a covariance equation. It integrates the independent positive bath-to-bath spectral transmission

    Q_x = 2 gamma^2 (T_x-T_y)/pi
          * integral_0^infinity omega^2 |G_xy(omega)|^2 d omega,
    G^-1 = K - m omega^2 I - i gamma omega I + i B omega J,
    J = [[0,1],[-1,0]].

For this K, |G_xy|^2 has numerator h^2+B^2*omega^2. This formula follows by substituting the independent Langevin force spectra into gamma*(T_x/m-<v_x^2>) and subtracting its equal-temperature equilibrium value. It is independent of the solution's four-dimensional Lyapunov calculation. The validator additionally compares both with the closed stationary moment formula.

Calibration has 288 independent Gaussian observations, 12 settings repeated 24 times, with fixed sigma=.0006. The uncertainty is independent of the unknown drag and response. Generation seed is 78113; the 256 noise realizations use 78117. The true drag is 1.13. The nominal fit is 1.1301095873 with reduced chi-square 1.06645 for both controls. All 256 calibration/parameter checks and physical predictions pass, and every completed shortcut fails all three hidden groups. The largest physical error is .000926; the smallest shortcut error is .8201 against the unchanged .04 relative RMS criterion. Coherent ±3 sigma calibration offsets preserve separation.

The independent frequency reference and analytic moment solution agree with the Lyapunov oracle to 1.54e-14 over 96 off-grid cases. First-law/channel balance, field reversal, equal-temperature zero heat and exact zero-field calibration agree to 2.2e-14. Every checked covariance is positive and every drift is stable. Positivity and stability throughout the domain also follow from positive K and positive viscous damping. Local tests give oracle 8/8 and shortcut 5 passing checks plus exactly 3 hidden prediction failures, within one second each. Agent and verifier limits remain 600 and 60 seconds.

This is related to the retained entropy-anomaly and magnetic-tracer tasks by its Lorentz force, but it has no mass limit, spatial temperature profile or stochastic-area observable. It tests attribution of energy to two distinct thermal reservoirs while retaining exact mechanics. The earlier small-field joint-limit prototype is preserved separately and was rejected because its exact limiting positional state necessarily drops the magnetic control. No benchmark source or evaluation was created for that rejected variant.

Primary sources:

- Abdoli et al., *Brownian magneto-gyrator as a tunable microengine*, https://arxiv.org/abs/2110.05284, and the published *Tunable Brownian magneto heat pump*, https://pmc.ncbi.nlm.nih.gov/articles/PMC9352690/. These derive Lorentz-mediated kinetic heat transfer between Cartesian reservoirs. The signs and symmetric-trap formulas above are derived directly from this task's specified equations.
- Filliger and Reimann, *Brownian Gyrator: A Minimal Heat Engine on the Nanoscale*, https://doi.org/10.1103/PhysRevLett.99.230602, for the tilted harmonic two-bath apparatus.

Validation: `results/magnetic-bath-transfer-validation.json` relative to the staged root. Reproduce with `python scripts/validate_magnetic_bath_transfer.py`; data regeneration requires `--generate`. The independent source peer and final file hashes accompany the staged results. No task instructions prescribe a solution method, and no private solution or hint is copied into the public image.

## Reviewed evaluation outcome

The neutral-r1 conditional screen produced three genuine physical successes: initial Az9juNs and followups dsAgbm8 and j96nraC. Every final source replaces trap-mediated channel power with the correct thermostat heat gamma*(T_i/m-<v_i^2>) while preserving the full Gaussian dynamics and a valid shared-friction fit. All three pass eight verifier checks; maximum hidden relative error is 5.79e-5. This candidate misses the intended difficulty target and is archived. Earlier staged/unevaluated labels in frozen provenance record the historical state before evaluation. The complete native commands, trajectories, saved sources, controls and metrics are preserved in this archive.
