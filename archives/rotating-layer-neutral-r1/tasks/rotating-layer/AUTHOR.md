# Rotating-layer revision 1

This task distinguishes velocity drag from direct relaxation of potential vorticity. The completed shortcut has the correct inviscid inertia–gravity frequency, the correct nonrotating damped wave, and the correct initial height and both velocity components. It also retains the initial balanced component. Its physical approximation is to let the entire linear potential-vorticity anomaly decay at the bottom-drag rate.

With height h*cos(kx), along-flow U*sin(kx), and transverse flow V*sin(kx), the apparatus gives

    h_dot = -H*k*U,
    U_dot = g*k*h + f*V - r*U,
    V_dot = -f*U - r*V.

Define w=h_dot and P=k*V-f*h/H. The same equations become

    w_dot = -r*w - (f*f+g*H*k*k)*h - H*f*P,
    P_dot = -r*(P+f*h/H).

The drag's curl damps relative vorticity k*V. The shortcut instead uses P_dot=-r*P. It retains the full forced height-wave equation, so it does not discard an initial mode or violate the specified initial-value interface. Its analytic prediction is independently checked against its own three-state generator. The two controls are exactly equivalent at f=0 and r=0. The nonrotating calibration identifies r through wave attenuation without testing the rotating circulation feedback.

The local f-plane approximation, resting centrifugal background, periodic geometry, linear shallow-layer limit, and prescribed drag force close the apparatus. There is no additional vertical mode, implicit mass sink, or freedom to choose a different friction law. All initial perturbations have zero spatial mean, and the full evolution conserves that mean layer volume. The oracle evolves h,w,P by a matrix exponential. The independent reference integrates height and both momentum equations with adaptive DOP853; a separate RK45 refinement provides a numerical cross-check.

The physical modal energy per density is (g*h²+H*(U²+V²))/4, with rate -H*r*(U²+V²)/2. The shortcut is linearly stable throughout the allowed parameter range, but does not inherit this energy inequality. In velocity variables it adds the spurious transverse acceleration r*f*h/(H*k). A supplied-domain counterexample has energy rate +5.6e-8 under the shortcut and -5.6e-8 under the correct model. This documented energy defect is a consequence of the wrong dissipative closure, not a numerical instability or coding error. The task does not claim that the shortcut is passive.

The private drag rate is 0.14 s^-1. Calibration has 144 measurements, fixed independent height uncertainty 5e-7 m, calibration seed 32021, and noise seed 42021. Fixed uncertainty carries no noiseless-response information. Both controls fit 0.1401201510 s^-1, with reduced chi-squared 1.06223. All 256 noise realizations pass calibration and parameter checks; maximum drag error is 0.355%.

The hidden metric is absolute height-amplitude RMSE in meters, with limit 1.5e-5. Nominal oracle errors are 2.75e-8–6.49e-8; shortcut errors are 4.64e-5–1.33e-4. Across fitted-parameter extrema, the oracle stays below 2.69e-7 and shortcut above 4.63e-5. This absolute threshold avoids division by near-zero or sign-changing wave amplitudes.

Oracle and independent momentum reference agree within 1.77e-15 m on hidden cases and 2.99e-15 m at parameter/input corners. Reference integrator refinement changes any state component by less than 1.27e-13. Both controls reproduce initial height exactly and initial height derivative to the finite-difference check's 1.06e-9 m/s error. The shortcut's closed expression agrees with its own independently assembled generator to 1.29e-18 m. Mechanical energy and drag-curl identities pass. The largest generator real eigenvalues in the allowed-domain scan are -0.00413 s^-1 for the oracle and -0.04 s^-1 for the shortcut; both are stable.

Local isolated verification gives oracle 7/7 passing tests in 0.61 seconds. The shortcut passes four interface/calibration/parameter checks and fails all three intended hidden groups in 0.66 seconds. See [the scientific report](../../results/rotating-layer-validation.json). Agent evaluation is pending; scientific separation does not establish a failure rate.

This is a distinct replacement for the weak variable-viscosity capillary candidate, whose full source, controls and reviewed 3/3 neutral-instruction passes are preserved in [its archive](../../archives/capillary-mixture-neutral-r2/README.md). This task uses the exact same neutral instruction as the current evaluation screen.

Reproduce with:

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_rotating_layer.py
```

Ordinary validation reads checked-in data; `--generate` intentionally rewrites both copies. Primary context for the potential-vorticity and wave decomposition is [the f-plane balance analysis by Mohebalhojeh and Dritschel](https://journals.ametsoc.org/view/journals/atsc/58/16/1520-0469_2001_058_2411_hobcft_2.0.co_2.xml). The bottom-drag correction used here follows directly by taking the curl of the specified force.

The completed neutral-instruction screen passed 2/3 unhinted Luna-high trials. Both passes restore the full velocity-drag dynamics. The failed trial retains independent exponential decay of potential vorticity; its source also states the correct velocity equations but incorrectly claims its closed form follows from them. A fixed-drag in-memory repair lowers every hidden error below 6.49e-8 m without changing calibration beyond 4.38e-18 m. This result exceeds the requested pass-rate target. All three trials, including the initial success, remain in [the reviewed evidence](../../results/neutrality-materials-trial-reviews.json).
