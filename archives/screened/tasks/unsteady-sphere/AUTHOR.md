# Unsteady sphere, revision 1

This candidate tests whether a steady calibration is incorrectly promoted to a dynamical constitutive law. The unknown liquid viscosity is identified by steady force/terminal-velocity measurements. Those data do not probe the momentum stored in the surrounding liquid or the diffusion of vorticity after a velocity change. The public apparatus states linear unsteady fluid momentum balance, no slip, and the far-field condition; it does not give the resulting drag law.

For the exp(-i omega t) convention, solving the exterior unsteady Stokes equations gives

    k = sqrt(-i omega rho / eta),  Re(k) > 0,
    Z_fluid = 6*pi*eta*a * [1 + a*k + (a*k)^2/9],
    V_hat = F_hat / (Z_fluid - i omega m_particle).

The square-root term represents viscous history. The quadratic term is -i omega times half the displaced fluid mass. Both follow from the prescribed fluid equations. Small convective Reynolds number permits neglect of u dot grad(u); it does not justify dropping rho partial_t u. The completed shortcut uses exactly the response of a particle with its stated mass and constant steady Stokes drag. Its mathematical calculation and calibration fit are correct, but its hydrodynamic force is incomplete at finite frequency.

Calibration measures V=F/(6*pi*eta*a). Both controls fit the same inverse-viscosity linear coefficient, so they agree on every noiseless calibration input. The hidden in-phase and quadrature responses probe frequency-dependent fluid momentum. No new boundary or forcing rule is introduced. Force magnitude is at most 1e-9 N; even using the largest steady mobility, the radius-based convective Reynolds number is below 0.0034 and the oscillatory displacement divided by radius below 0.0027 at the minimum nonzero frequency. The incompressible, small-amplitude, linear model is explicit.

The independent reference does not insert the closed drag formula. It solves the two exterior streamfunction modes numerically for the no-slip conditions:

    psi = V*a^2*h(r/a)*sin(theta)^2,
    h(q) = C/q + D*exp[-s(q-1)]*(1+1/(s*q)), s=a*k,
    h(1)=1/2, h'(1)=1.

It obtains pressure from the irrotational mode, obtains viscous stress from surface derivatives, and integrates the axial traction by angular Gaussian quadrature. The validator independently transforms the causal t^(-1/2) history kernel numerically, checks added mass from the kinetic energy of the potential dipole flow, verifies positive real impedance, and checks the zero-fluid-density, DC, and high-frequency limits. The relevance of history forces even in low-Reynolds-number oscillatory flow is demonstrated experimentally in [An experimental study on stationary history effects in high-frequency Stokes flows](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/an-experimental-study-on-stationary-history-effects-in-highfrequency-stokes-flows/FB0CF77C4679442E565B7DA5CA69AA61); the task uses the unbounded single-sphere continuum model.

Calibration uses seed 9325, true viscosity 0.008 Pa s, and 200 measurements with independent Gaussian errors at 0.6% of the largest steady velocity. The parameter tolerance is 3%, and calibration reduced chi-square must be below 1.5. Hidden RMSE is divided by the reference RMS within each group, with a limit of 0.04. This is far above both numerical and fitted-parameter errors.

The checked-in fit gives viscosity 0.0080039437 Pa s, relative error 0.0493%, reduced chi-square 0.96129. Oracle hidden NRMSE is at most 0.000386; shortcut errors range from 0.79255 to 1.09694. All 256 additional noise realizations pass calibration and parameter checks, with maximum viscosity error 0.224%. At observed fitted-parameter extrema, the oracle remains below 0.001746 and the shortcut above 0.79016.

Independent exterior traction agrees with the closed impedance below 5.6e-16 relative error. The direct history-kernel Fourier integral agrees below 1.75e-13 and the potential-flow added mass below 3e-16. The formal high-frequency inertial limit is approached within 3.22e-5 in its diagnostic. The complex sign conventions are tested through both velocity components.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 scripts/validate_unsteady_sphere.py unsteady-sphere
```

Only add `--generate` to intentionally replace both calibration copies. Report: `jobs/unsteady-sphere-validation/summary.json`. Local isolated-copy pytest gives 7 oracle passes; the shortcut gives 4 passes and 3 hidden prediction failures. The parent coordinates Harbor controls and paired gpt-5.6-luna/high trials; scientific validity is separate from measured agent difficulty.
