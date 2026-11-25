# Entropy anomaly — revision 1

This candidate tests whether eliminating fast variables preserves calorimetric observables. The completed shortcut correctly computes the stationary positional density and probability current, including the temperature-gradient term in the Smoluchowski flux. It then evaluates the positional path-entropy rate. That rate fits every uniform-temperature calibration measurement exactly but misses heat transported by fast velocities in a nonuniform bath.

The apparatus specifies an underdamped Langevin particle with exactly one velocity degree of freedom. The reported quantity is the stationary bath entropy from its local Stratonovich heat, averaged for a long time at positive mass before taking mass to zero. Specifying this order is necessary: setting mass to zero in the equation of motion first changes the thermodynamic observable. External heat required to maintain the bath profile is excluded.

The phenomenon is described in [Celani et al., Anomalous thermodynamics at the micro-scale](https://arxiv.org/abs/1206.1742). The task uses a one-dimensional periodic coordinate, so its kinetic contribution has coefficient 1/2, rather than the three-dimensional coefficient 5/6.

## Calibration and physical separation

At uniform temperature the exact finite-mass velocity distribution has mean F/gamma and variance T/m. Its bath entropy rate is F^2/(gamma*T). Fitting reciprocal friction by weighted least squares therefore identifies the only unknown. Both completed controls use precisely this calibration predictor. The public parameter range is gamma in [0.7,1.6]; the private value is 1.1.

For nonuniform temperature the normalized stationary positional state satisfies

    gamma*J = F*rho - (T*rho)',     J constant.

The oracle evaluates

    rate = integral gamma*J^2/(T*rho) dx
           + integral rho*(T')^2/T dx / (2*gamma).

The shortcut retains only the first integral. It uses every supplied force and temperature-profile input, conserves probability, and predicts the correct stationary position distribution and current. Its failure is not a missing noise-induced drift or a numerical error in the positional equation. All hidden cases have nonzero force, so the shortcut makes nonzero, substantive predictions.

A moment derivation also fixes the sign and dimension-dependent factor without assuming the entropy formula. Write M_n(x)=integral v^n p(x,v) dv. Stationary Kramers balance gives

    m*M_2' = F*rho - gamma*J,
    M_2 - T*rho/m = F*J/gamma - m*M_3'/(2*gamma),
    m*M_3 -> [F*T*rho - (T^2*rho)']/gamma + 2*T*J
           = 3*T*J - rho*T*T'/gamma.

The finite-mass heat convention gives gamma*integral(M_2/T-rho/m) dx. Substitute the second relation, integrate by parts, and use the limiting third moment. The periodic integral of J*T'/T vanishes, leaving the positional contribution and the positive kinetic term above.

At zero force rho is proportional to 1/T, the positional current is zero, and the exact limiting rate for the sinusoidal profile is

    T0*k^2*(1-sqrt(1-a^2))/(2*gamma).

This is checked independently. Uniform temperature gives F^2/(gamma*T) and zero kinetic contribution.

## Independent private reference

The verifier does not use the oracle entropy formula. It solves the stationary finite-mass Kramers equation in a Fourier spatial representation and a Gaussian-weighted Hermite expansion in w=sqrt(m/T0)*v. Streaming and force couple adjacent velocity modes; temperature-dependent diffusion couples modes separated by two. The null equation is replaced by probability normalization.

From the resulting coefficients c_n(x), the reference evaluates the finite-mass calorimetric rate directly:

    gamma/m * integral [T0*(c0+sqrt(2)*c2)/T(x) - c0] dx.

It uses 95 spatial nodes and 48 velocity modes. Three positive masses h,h/2,h/4, with h=.004*gamma^2/(T0*k^2), are extrapolated quadratically to zero. The mass scale controls the ratio of velocity relaxation time to thermal spatial diffusion time. The combination is rate(h)/3 - 2*rate(h/2) + 8*rate(h/4)/3. Exact uniform-bath calibration is evaluated analytically.

The validator halves the mass scale and increases resolution to 127 spatial nodes and 64 velocity modes. It separately compares direct finite-mass heat with the third-moment heat balance and checks decreasing negative spectral tail mass. The truncated reference has small tail oscillations, not an exactly positive density at finite resolution; their integrated negative mass falls from 5.30e-6 at 24 velocity modes to 8.57e-9 at 48 and 8.52e-11 at 64. The relevant heat observable is already converged far below the grading tolerance.

## Validation

Run from the repository root:

    OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 scripts/validate_entropy_anomaly.py

Calibration regeneration requires the explicit `--generate` flag. Seed 9330 produces 200 measurements. The fitted friction is 1.0996005454 with reduced chi-square 1.04895. Both controls agree with exact calibration to 6.7e-16. All 256 independent calibration-noise realizations pass calibration and parameter checks; maximum relative parameter error is 0.00470. Hidden predictions are additionally checked at the two observed parameter extrema, rather than described as exhaustive hidden Monte Carlo tests.

The physical oracle's hidden normalized RMS errors are about 0.000363. The shortcut scores 0.412, 0.609, and 0.452. The observed parameter-extrema check has maximum oracle error 0.00472 and minimum shortcut error 0.409. Each group uses RMS truth as its scale and a 0.04 threshold, well above calibration and numerical uncertainty.

Independent finite-mass reference disagreement is 2.53e-8 on hidden cases; halving its mass scale changes predictions by 2.21e-8 and increasing both basis sizes by 5.72e-12. Allowed-range corner disagreement is below 1.8e-7. Direct heat and third-moment heat agree within 1.5e-12. Probability normalization, force reversal, exact constant-force current, zero-force density, and inverse-friction scaling are checked.

The full report is `jobs/entropy-anomaly-validation/summary.json`; shareable evidence is `results/entropy-anomaly-science.json`. Independent source reviews are recorded in `results/constraint-hardening-physics-review.json` and `results/materials-entropy-kinetic-review.json`. Local isolated controls must pass 7/7 tests for the oracle and pass calibration/parameter tests while failing the three hidden groups for the completed shortcut. Harbor controls and Luna difficulty evaluation are separate, root-scheduled checks; no model failure rate is claimed from numerical validation alone.
