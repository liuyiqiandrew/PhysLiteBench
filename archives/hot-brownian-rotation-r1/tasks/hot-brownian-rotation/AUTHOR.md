# Hot Brownian rotation, revision 1

This staged task tests the contraction of a spatially nonisothermal liquid to a single effective thermal reservoir. The sphere is clamped; the readout is one component of the fluid torque spectrum. There is no particle-motion, heat-current, or detector-delay convention to infer. The source retains the complete frequency-dependent rotational hydrodynamic impedance and the exact heated zero-frequency noise. Its physical approximation is that the same effective temperature describes every frequency.

For the exp(-i omega t) convention, let k=sqrt(-i omega/eta), with positive real part. A unit auxiliary rotation generates

    u_phi(r,theta) = A(r) sin(theta),
    A(r) = a^3/r^2 * exp[-k(r-a)] * (1+k*r)/(1+k*a).

The response satisfies no slip at r=a and decays at infinity. Its torque impedance is

    Z = 8*pi*eta*a^3 * [1+(k*a)^2/(3*(1+k*a))].

The source computes this exact impedance, including viscous momentum diffusion and fluid inertia. Let h=A'-A/r. Angular integration of the viscous dissipation gives

    Re Z = (8*pi*eta/3) integral_a^infinity r^2 |h(r)|^2 dr.

The stated local stochastic stress transmits torque through the same response. Its two-sided spectrum is

    S = 2*(8*pi*eta/3) integral_a^infinity T(r)*r^2 |h(r)|^2 dr.

Equivalently, S=2*T_noise(omega)*Re Z, with the temperature averaged using that frequency's dissipation. At zero frequency h=-3*a^3/r^3, so T_noise(0)=T0+3*DeltaT/4. At high frequency dissipation is confined near the hot surface and T_noise approaches T0+DeltaT. This last limit is a check of the specified ideal incompressible model, not a claim about arbitrarily high real-fluid frequencies.

The supplied predictor instead uses S0=2*(T0+3*DeltaT/4)*Re Z at every frequency. It is a complete positive generalized-Langevin noise model at its calibrated effective temperature. All input controls enter that model. It is exact at every zero-frequency preparation, including nonzero heating, and at every frequency in a uniform-temperature liquid. The error is the physical reduction of many local baths to one frequency-independent temperature, not an incorrect impedance or omitted matrix cross term.

Calibration contains 288 independent measurements: 12 known heated preparations repeated 24 times. All have omega=0. Since S(0)=16*pi*eta*a^3*(T0+3*DeltaT/4), the response is strictly linear in the unknown viscosity and has a nonzero positive slope. Weighted least squares is globally identifiable throughout [.7,1.4]. The true viscosity is 1.07. Instrument uncertainty is the fixed absolute sigma=.02; it contains no response or unknown-parameter dependence. Seeds 118031 and 118037 generate calibration and the 256 noise checks. The exact neutral instruction permits replacing any prediction code while preserving the API.

The independent verifier does not use the analytic velocity field or its dissipation weight. It discretizes radial angular momentum into rotating spherical shells. The inertia of a shell is (8*pi/15)*(r_right^5-r_left^5). Between two shell angular velocities, the steady viscous conductance is 8*pi*eta/(r_left^-3-r_right^-3). The random transmitted torque has spectrum 2*g*T_edge, where T_edge is the local temperature averaged over that edge's radial shear resistance. For the prescribed T(r), this average is T0+(3*DeltaT*a/4)*(r_left^-4-r_right^-4)/(r_left^-3-r_right^-3).

The tridiagonal frequency-domain angular-momentum equations give each independent edge stress's transfer to the clamped sensor. Summing its positive transmitted noise power supplies the reference spectrum. Equivalently, a unit boundary rotation is the adjoint response used for this reciprocal scalar fluid. The discrete equality between edge dissipation and the real boundary impedance is checked. Richardson extrapolation of 384/768 shells removes leading grid error;768/1536 comparisons and doubled outer radius check convergence. A final edge extends to infinity with its exact static resistance. At nonzero frequency the dynamic grid reaches at least 16 viscous penetration depths, so this remaining quasistatic tail is negligible; domain refinement confirms this independently.

Validation is recorded in `staging/hot-brownian-rotation-r1/results/hot-brownian-rotation-r1-validation.json`:

- Both completed controls fit eta=1.070018528 with reduced chi-square .923405. Oracle hidden errors are below 1.29e-5; shortcut diagnostic errors are .10523,.12569,.11604. Both pass uniform-temperature and zero-frequency anchors.
- All 256 noise realizations pass calibration and parameter recovery. Every oracle passes and every shortcut fails its diagnostic groups. Maximum viscosity error is 4.40e-5 relative; maximum oracle hidden error 3.30e-5; minimum shortcut diagnostic error .105215.
- Oracle/reference agreement is within 4.57e-7 relative on all hidden inputs. Independent impedance error is below 5.92e-7; the discrete dissipation identity agrees within 6.24e-14. Public-domain corner reference error is 1.14e-7; doubled-domain change 1.10e-12.
- Uniform-temperature fluctuation-dissipation, exact static calibration, positive spectra, additive independent bath noise, and the low/high frequency limits pass. The original independent strain-versus-impedance prototype is preserved in `prototype/`.
- Local pytest gives oracle 9/9; the completed shortcut passes 6 checks and fails only the 3 diagnostic groups. Both verifiers complete in under one second. The .025 hidden tolerance exceeds numerical and fit errors by a wide margin.

Reproduce from the repository root:

```bash
uv run --no-project --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/hot-brownian-rotation-r1/scripts/validate_hot_brownian_rotation.py
```

Use `--generate` only for deliberate regeneration of both calibration copies. Root owns Docker controls and three fresh frozen unhinted trials. No model evaluations have been run at this stage, and the source's scientific validity does not establish difficulty.

The archive audit is `results/archive-duplicate-audit.json` within this stage. The previous unsteady-sphere task omitted deterministic fluid history; here that response is exact. Heated-filament and thermal-bodies r11 omitted correlations between separate modes; this task tests frequency-dependent scalar reservoir weighting. Hydrodynamic-fluctuations omitted equilibrium cross noise; uniform-temperature FDT is exact here. Gyroscopic-noise required time-antisymmetric cross spectra, absent here. Entropy-anomaly involved singular-mass calorimetry, also absent here. The broader local-bath physics is shared and is not presented as unrelated.

Primary reference: Falasco, Gnann, Rings and Kroy, [Effective temperatures of hot Brownian motion](https://arxiv.org/abs/1403.4040), Physical Review E 90, 032131 (2014), derives the spatial thermal-stress contraction and its frequency dependence. The task fixes its constant-property linear continuum model explicitly. The analytic torque response and independent shell/noise calculation above validate this particular clamped measurement without importing a fitted empirical formula.
