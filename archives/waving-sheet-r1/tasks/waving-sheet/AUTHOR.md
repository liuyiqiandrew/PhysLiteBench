# Waving-sheet pumping, revision 1

The supplied model solves the complete first-order unsteady Stokes field and the second-order geometric boundary condition. Its approximation omits convective momentum transport in the mean fluid equation. Small displacement makes convection higher order than the oscillatory velocity, but the requested pumping begins at that same higher order. Correct harmonic calibration therefore does not establish the mean pumping law.

This stage has scientific validation and local completed controls. No model-agent outcome is claimed. The neutral instruction permits replacing all code, and the ordinary public fit stub starts with viscosity=None. The physical correction and private tests remain outside the agent image.

## Physical preparation and equations

The material sheet is anchored horizontally and extensible, with x=X and y=a*cos(kX-omega*t). It is not the inextensible sheet with an additional tangential material velocity. The liquid occupies only the upper half-space. No slip applies on the actual moving boundary. There is no imposed mean pressure gradient, far-field shear or counterflow. The stationary response is taken before its far-field mean, then the small-amplitude coefficient is extracted. This order avoids imposing a conflicting zero velocity at infinity before the pumping state develops.

Use the complex phase exp(i*(kx-omega*t)) and write the first-order streamfunction divided by a as

    psi = A*exp(-k*y)+B*exp(-s*y),
    s = sqrt(k²-i*omega/nu), Re(s)>0,
    A = omega*s/[k*(s-k)], B = -omega/(s-k).

It has psi(0)=omega/k and psi'(0)=0. The harmonic amplitudes are u=psi' and v=-i*k*psi. Both controls retain these full fields, including finite-frequency viscous penetration and phase. Expanding the exact moving-wall condition supplies the mean horizontal boundary value

    U(0)/a² = -Re(psi''(0))/2 = omega*Re(s)/2.

The completed unsteady-Stokes approximation extends this value as a constant mean velocity. That is the correct solution of its assumed linear fluid equation and moving boundary expansion. It is not a missing boundary Taylor term, a wrong decay root or an unfinished mean boundary solver.

For Navier–Stokes, period-averaged horizontal momentum gives

    nu*U''/a² = Re(v*conjugate(u'))/2,
    nu*U'/a² = Re(u*conjugate(v))/2,
    U(infinity)/a² = U(0)/a² + integral[Re(u*conjugate(v))/2]dy/nu.

The integration constant follows from zero distant shear. The oracle evaluates the last exponential integral analytically. At the far-field detector the oscillatory fields vanish, so an additional Eulerian-to-particle-following velocity conversion is not part of the readout. The measurement is a laboratory fluid velocity, not a swimmer speed.

Frequency is in inverse seconds, k and height in inverse millimeters and millimeters, and nu in square millimeters per second. Harmonic velocity divided by a is in inverse seconds. Pumping divided by a squared is in inverse millimeter per second. Both mean models give positive finite pumping for the positive traveling-wave controls; their low-frequency normalized limit is U/(a²*omega*k)=1/2.

## Calibration and identification

There are72 signed harmonic settings: three k values, three frequencies, two heights and four real/imaginary horizontal/vertical channels. Four repetitions give288 records. Instrument sigma is fixed at .002 inverse seconds, independent of the clean response and unknown parameter. Both physical controls coincide exactly on every calibration input at every nu.

The already measured fields provide a direct global identifiability argument:

    u(h)+i*v(h) = omega*exp(-s*h).

At fixed positive k, omega and height, its modulus determines Re(s), which strictly decreases with positive nu. No phase unwrapping is required. The validator checks this identity, monotonicity, and the full noiseless calibration objective on a401-point parameter grid.25 noiseless fits spanning [.7,1.4], including both endpoints, recover nu within4.86e-9 relatively. The checked noiseless objectives each have one minimum; no statement is made about arbitrary noisy objectives.

Private true nu is1.05, with calibration seed281021 and noise seed281027. The independent boundary-value reference generated the clean calibration, and public/private data are identical. Its bias relative to the analytic fields is at most6.55e-9 instrument sigma. Both controls fit nu=1.049973997, with reduced chi-square1.10161. There is no default truth in the public model or uncertainty side channel.

Three pumping groups vary frequency, wavelength and combined controls. Their positive reference signals are at least .4791 in the declared coefficient units. Nominal source errors are1.45746, .90302 and1.63704 normalized RMS, against the ordinary .04 gate. The separate harmonic anchor group passes both. Parameter and calibration gates remain3% and reduced-chi-square1.5. The verifier checks values, finiteness and shape, with no solver restrictions.

## Independent reference and validation

The private reference solves the fourth-order linear vorticity boundary ODE directly, splitting complex variables into real and imaginary components. It fixes psi and its derivative at the sheet and at a distant finite boundary. A second inhomogeneous boundary ODE uses the actual period-averaged convective acceleration of that computed field, the expanded wall velocity and zero distant mean shear. Its far-boundary velocity supplies the reference. It does not insert the analytic exponential coefficients or integrated pumping correction.

The default depth is20/k and BVP tolerance1e-8. A26/k,1e-9 refinement changes mean pumping by at most7.16e-10 relatively.24 public-domain cases (12 corners and12 random cases) agree with the oracle within1.86e-9 in pumping and8.16e-12 in harmonic readings on the stated normalization. The independently computed moving-boundary source value agrees with the completed shortcut within2.45e-15 relatively.

The leading oscillatory work done by the sheet equals integrated viscous dissipation within1.34e-15 relatively. This is a cycle-averaged first-harmonic energy check, not a claim of finite total kinetic energy for an infinite domain with uniform far-field mean flow. The independently computed mean momentum flux `nu*U'-Re(u*v*)/2` is spatially constant within4.55e-10 after normalization. Wall velocities, mean-flow reversal and dimensional scaling also pass. The source satisfies its own unsteady-Stokes mean equation; its failure in the physical momentum flux is the stated convective approximation.

All256 independent calibration-noise draws pass fit and parameter gates. Every oracle prediction passes, while every shortcut fails all three pumping groups. Maximum oracle hidden error is .0001528 and minimum shortcut diagnostic error .90253. The shortcut gap remains above .66366 throughout the fitted parameter range. The frozen prototype has12 domain cases; this package adds12 random cases and the noise, power, momentum and fit checks without changing that prototype.

Actual isolated local pytest gives oracle7/7 and shortcut4passes/3 intended mean-pumping failures, both in .92seconds. Scientific validation took6.33seconds. The standard600-second agent and60-second verifier limits remain. These controls do not predict whether a model will identify the missing physics.

A trial retaining unsteady Stokes plus the correct moving boundary is a physical-model failure. A trial recognizing the mean convective term but making a sign, phase, perturbation-order or numerical error must be classified separately. A fixed-parameter causal repair changes only mean momentum transport; the calibrated viscosity and exact harmonic field remain intact.

## Related work and provenance

The root-owned prototype and its independent source peer are preserved unchanged as five files. Its broad archive scan found no prior waving-sheet or convective-streaming task. Acoustic-cavity-pressure concerns fixed-mass compressible mean-density redistribution and wall force. Pressure-surface-waves concerns elastic follower traction. Unsteady-sphere and thermal-unsteady-sphere concern linear fluid impedance or thermal force spectra. They share a lower-order-field versus higher-order-observable theme, but not this moving-sheet incompressible momentum closure.

[E. O. Tuck, A note on a swimming problem, JFM31,305–308 (1968)](https://doi.org/10.1017/S0022112068000169), provides primary background on the convective contribution missing from Reynolds' calculation. Root read the publisher abstract; the full text was not obtained. The equations used here were derived independently and checked by the two boundary ODEs. [Sauzade, Elfring and Lauga, Taylor's swimming sheet: Analysis and improvement of the perturbation series](https://arxiv.org/abs/1302.4029), provides related small-amplitude sheet background rather than the finite-inertia correction.

Reproduce author checks from the repository root with:

    uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/waving-sheet-r1/scripts/validate_waving_sheet.py

Use `--generate` only for deliberate data replacement. Root owns Docker controls and all model evaluations.
