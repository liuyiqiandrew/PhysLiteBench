# Rotating-reservoir calorimetry, revision 1

This task distinguishes internal thermal heat from laboratory mechanical power transferred to a moving reservoir. The supplied predictor already solves the full finite-inertia stationary covariance, including rotation, anisotropic confinement and both thermal noises. Its error is the interpretation of the calorimeter signal. All calibration has a stationary reservoir, where the two signals coincide exactly.

The neutral instruction permits replacing any implementation. The public apparatus defines both ideal Langevin force laws, the motor-maintained reservoir motion, the thermostat readout, and the stationary preparation. It does not prescribe a heat formula or covariance solver. The force laws are the complete model; no fluid pressure, added mass, or unmodeled background force is assumed. Calibration sigma is fixed instrument noise rather than a function of the noiseless response. The hint is private unless explicitly added by a runner.

## Physics and calibration

Write b=0.9, gamma=drag, Omega=angular_speed, R=[[0,-1],[1,0]], and let C be the stationary covariance of (r,v). Define x2=tr(C_rr), v2=tr(C_vv), and L=<x*v_y-y*v_x>=tr(R*C_rv). The shortcut computes laboratory energy gain by reservoir B,

    q_lab = b*(v2-Omega*L)-2*b*T_b.

Its internal thermal heat instead uses the particle's velocity relative to the reservoir's local mean motion,

    q_B = b*(v2-2*Omega*L+Omega^2*x2)-2*b*T_b.

The reservoir torque on the particle is tau=-b*(L-Omega*x2). The motor power is P=Omega*tau, so q_B=q_lab+P. With q_A=gamma*(v2-2*T_a), stationarity gives q_A+q_B=P. The static trap can exchange angular momentum but does no work. The Stratonovich noise contribution is 2*b*T_b in both heat expressions because position has no instantaneous noise in the stated finite-inertia model.

At Omega=0 the exact signal is G(gamma)*(T_a-T_b), where G=2*b*gamma/(gamma+b). Its derivative 2*b^2/(gamma+b)^2 is strictly positive throughout the allowed range. Both models therefore share the same closed weighted least-squares fit, with gamma=b*G/(2*b-G). The source retains the full covariance even though this conductance identity simplifies calibration.

This is a frame-of-heat distinction, separate from the archived spring-reservoir boundary-energy and dispersive stored-energy tasks. A primary background source is Speck, Mehl and Seifert, [Role of External Flow and Frame Invariance in Stochastic Thermodynamics](https://doi.org/10.1103/PhysRevLett.100.178302), Physical Review Letters 100,178302 (2008), with [author manuscript](https://arxiv.org/abs/0712.0232). The present finite-inertia ideal force model and its heat balance are derived directly above; the source is background rather than an assumed formula for this apparatus.

## Independent checks

The oracle solves the four-dimensional continuous Lyapunov equation. The private reference independently integrates propagated stochastic impulses, C=integral exp(A*t)*Q*Q^T*exp(A^T*t) dt, and reconstructs heat from laboratory energy transfer plus torque work. A third frequency-response covariance integral agrees with both to 1.13e-13 over selected domain corners. Hidden oracle/reference agreement is 7.14e-15. A cold evaluation of all twelve private inputs takes 0.059 seconds, well below the verifier's 60-second limit.

A conservative Routh-Hurwitz bound gives a positive stability margin of 2.023775 throughout the public domain. Corner checks cover positive covariance, stationary first law, nonnegative entropy production q_A/T_a+q_B/T_b, and reversal of rotation. First-law residuals are below 7.33e-15. The formal one-bath, isotropic-trap limit recovers the rotating Gibbs covariance and zero thermal heat and torque; the stationary equal-temperature limit recovers ordinary Gibbs equilibrium. These author-only limit checks need not lie in the calibration domain.

## Validation

There are 288 calibration records, true drag 0.67, fixed sigma=0.001, calibration seed 71061 and noise seed 146822. Public and private data copies are identical. The fitted drag is 0.6698903768520476 and reduced chi-square is 1.059217. Nominal oracle hidden errors are at most 8.08e-5. Shortcut errors are 1.57049, 0.522332 and 1.59242 in the equal-temperature, temperature-bias and trap-stiffness groups. The standard hidden tolerance is 0.04 in group RMS relative error; parameter tolerance is 3% and reduced calibration chi-square must be below 1.5.

All 256 independent noise draws pass parameter and calibration checks; every oracle passes all hidden groups and every shortcut fails. Maximum fitted parameter error is 0.0514%, worst oracle hidden error is 0.000254, and minimum shortcut hidden error is 0.52226. Exact noiseless recovery is checked near both endpoints of the parameter interval. Calibration predictions of the two physical closures are identical.

Local isolated pytest reports oracle 8/8 and shortcut 5 passes plus the three intended hidden failures, taking 0.34 seconds per control. These controls establish physical and numerical separation; they do not establish Luna difficulty. Docker controls and model evaluations are managed separately by the root runner.

Reports: [science](../../results/rotating-reservoir-validation.json), [local controls](../../results/rotating-reservoir-local-controls.json), [source provenance](../../results/rotating-reservoir-source-provenance.json), [physics review](../../results/constraint-hardening-physics-review.json), and [independent prototype review](../../results/materials-rotating-bath-prototype-review.json).

Reproduce with `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_rotating_reservoir.py`. Use `--generate` only when intentionally regenerating both frozen calibration copies.
