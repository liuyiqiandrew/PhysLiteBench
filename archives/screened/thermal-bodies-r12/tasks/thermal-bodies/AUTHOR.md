# Thermal calorimeters revision 12

The complete preceding colored-reservoir task, controls, validator, reports and all three reviewed Luna-high passes are preserved in `archives/thermal-bodies-r11/manifest.json`. Those agents all rebuilt the correct full covariance. This revision supplies the full covariance and changes the physical measurement to finite-time outer-thermostat heat fluctuations. It retains the neutral instruction, fit API, 3% parameter gate, reduced-chi-square gate, 5% hidden error gate and runtime limits.

The couplers now have finite known mass M. Their link stiffness is k=gamma/tau; tau is the independently known ratio of drag to link stiffness, not an assumed exponential memory kernel after introducing inertia. With state (q,p,y,P),

    qdot=p,             pdot=-Kq+k(y-q),
    ydot=P/M,           dP=[-k(y-q)-gamma P/M]dt+sqrt(2 gamma T)dW.

Both supplied controls solve this exact eight-dimensional stationary Gaussian process, including all correlations. The completed shortcut also integrates the mechanical spring-power fluctuations exactly. Thus restoring cross-mode noise or improving its numerical solver does not repair the hidden physical error.

The outer calorimeter is separated from a main oscillator by a physical spring and coupler. The intervening energy is

    E_i = k(y_i-q_i)^2/2 + P_i^2/(2M).

With W_i the integrated spring power into the main oscillator, the first law gives Q_i=W_i+E_i(t)-E_i(0). The source uses the exact variance of W_i as its heat prediction. Stationary mean heat and mean work coincide, so both the mode-correlation and mean-power calibration measurements are exactly shared for every gamma. Finite-time heat variance also needs the energy-difference variance and its covariance with W_i. Keeping finite M avoids the kinetic-energy boundary ambiguity of overdamped calorimetry.

The oracle computes work correlations and endpoint correlations from Gaussian Wick contractions. If A is the symmetric spring-power matrix, B is the coupler-energy matrix, S is stationary covariance, and Phi(t) is the drift propagator, then a quadratic cross covariance is

    Cov[x(t)^T B x(t), x(0)^T A x(0)]
       = 2 Tr[B Phi(t) S A S Phi(t)^T].

The independent reference instead follows the actual thermostat's Ito heat increments,

    dQ_i = [gamma T_i/M - gamma P_i²/M²]dt
           + sqrt(2 gamma T_i)*P_i/M dW_i.

It integrates closed equations for the mean heat, heat-weighted second moments E[Q_i xx^T], and E[Q_i²], in a different canonical state ordering. The same-noise cross term is included. It contains no work-plus-endpoint reconstruction.

Calibration is deliberately regenerated for the finite-mass apparatus using the existing seed 876031: 192 mode correlations at fixed sigma 0.003 and 32 stationary mean-power records at fixed sigma 0.0005. Both uncertainties are independent of gamma and the noiseless response. The public and private copies are identical. Fitting the frozen data gives gamma 0.700383615 and chi-square 1.10565 for both controls.

`results/thermal-r12-validation.json` contains 256 independent calibration-noise fits and complete hidden rescorings. All oracle/calibration/parameter checks pass; no shortcut passes. The maximum oracle hidden error is 0.000738, while the minimum shortcut group error is 0.7023 against 0.05. All hidden groups fail robustly. The unchanged harness gives oracle 7/7 and shortcut 4 passed/3 hidden failures; see `results/thermal-r12-local-controls.json`.

The independent heat-moment reference agrees within 5.1e-12 on hidden cases and 3.2e-12 at finite-mass domain corners. The 48-to-96 quadrature change is 1.4e-12. Direct moment tolerance refinement, short-time thermal noise, covariance positivity, equilibrium Gibbs covariance, stationary inner/outer heat balance, total heat balance, deterministic energy dissipation and noiseless parameter recovery across the allowed interval all pass. Materials independently reviewed the source and an off-grid case in `results/materials-hardening-physics-review.json` under `thermal-bodies-r12`.

The apparatus specifies all physical stored energies and the location/sign of the calorimeters. It does not supply the missing heat/work conversion or the private Ito heat formula. The initial stationary joint state includes the couplers and their correlations. No degrees of freedom are silently eliminated from the measured heat.

No agent outcome is claimed before this revision's frozen trials finish. Reproduce with `python scripts/validate_thermal_colored.py`; add `--regenerate` only for deliberate calibration regeneration. Run both isolated public/private controls with `python scripts/check_thermal_colored_controls.py`. Source and archive checks are in `results/thermal-r12-source-provenance.json`.
