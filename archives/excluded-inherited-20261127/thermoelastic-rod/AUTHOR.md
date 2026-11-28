# Clamped viscous thermoelastic rod, revision 3

This revision adds a known graded Kelvin–Voigt axial stress. It preserves the
neutral instruction byte for byte and keeps the original parameter, calibration
and hidden-error thresholds. The complete preceding revision and its three
neutral passing trials are preserved in
`archives/thermoelastic-rod-neutral-v1/manifest.json`. Those trials correctly
repaired the elastic weighted-capacity coupling; they are not evidence about
this revision.

The local total stress is

    sigma(x) = E*(epsilon-alpha(x)*theta) + eta_m(x)*epsilon_dot,
    eta_m(x) = E*(12 s)*[1+0.97*cos(2*pi*x/L)].

The material remains positive: eta_m/E ranges from 0.36 to 23.64 seconds. Fast
mechanical force balance makes sigma spatially uniform; the clamps require
mean(epsilon)=0 and hence mean(epsilon_dot)=0. Temperature perturbations are
small, and the stated effective one-dimensional constitutive law is used after
transverse relaxation. The dashpot responds to total strain rate, as explicitly
specified, not to the derivative of the stress-free thermal strain.

Write R(x)=1/eta_m(x), r=R, and f=E*(alpha*theta-epsilon). Eliminating the
reaction stress gives

    sigma = -mean(R*f)/mean(R),
    epsilon_dot = R*f - R*mean(R*f)/mean(R).

The constrained mobility is M=R-r outer r/mean(r), with spatially normalized
inner products. The completed shortcut instead uses P*R*P, where P subtracts
the ordinary spatial mean. Both operators are symmetric positive, annihilate
constant forces, preserve the clamp exactly, and reach the same elastic
thermal equilibrium. The shortcut distributes the constraint force according
to an ordinary projection; it does not enforce uniform total stress when
viscosity varies. This is a coherent passive projected-rate approximation,
not a missing fit or an unstable discretization.

The linear reversible thermal equation follows from local entropy:

    T0*ds = c_e*dtheta + T0*E*alpha*depsilon,
    c_e*theta_dot + T0*E*alpha*epsilon_dot = k*theta_xx.

Viscous heating eta_m*epsilon_dot² is quadratic and consistently omitted at
this order. Fourier flux and the finite-body contact complete the thermal
boundary problem. The finite body has capacity 6 J/K. The contact heat flowing
into the rod is H*(T_body-T_surface), with the other surfaces insulated.
Initial strain is the zero-rate clamped equilibrium
alpha*theta-mean(alpha*theta); the public preparation specifies waiting for
all initial strain rates to vanish before releasing the thermal evolution.

The conserved linear internal-energy increment is

    U = V*[c_e*mean(theta)+T0*E*mean(alpha*epsilon)] + Cb*theta_b.

Both controls conserve this U. Their positive mobility laws also dissipate
quadratic availability

    A = V*[c_e*mean(theta²)/(2*T0)+E*mean(epsilon²)/2]
        + Cb*theta_b²/(2*T0).

The correct law additionally makes local total stress uniform. This distinction
prevents a simple clamp, energy or passivity check from revealing the shortcut.

Calibration is the insulated midpoint-odd first cosine temperature profile.
Expansion and mechanical viscosity are midpoint-even, so both controls remain
in the odd subspace and the weighted and ordinary projections both vanish.
They predict the same entire calibration curve for every allowed conductivity.
This curve includes the local viscous relaxation and identifies k. Hidden
midpoint-even profiles, with and without finite-body contact, require the
weighted reaction stress. Contact preparations remain inside the public 1 K
bound on the full initial temperature profile. No tolerance was tightened.

The oracle projects temperature onto 97 normalized cosine modes and strain
onto 96 nonconstant cosine modes. The latter impose zero mean strain directly.
It integrates products of the known material coefficients, uses the weighted
mobility, and propagates the symmetric generalized dissipation eigenproblem.
The completed shortcut changes only that mobility projection. Ninety-six
modes resolve the positive but strongly graded viscosity; a 128-mode check
measures residual spatial error.

The independent reference uses finite-volume entropy balances, cell strains,
and a bath temperature. It explicitly solves the common-stress algebraic
constraint at each linearized rate, rather than using the oracle modal
mobility. Helmert coordinates remove the conserved mean-strain degree of
freedom. A half-cell thermal resistance is in series with the contact. The
cell generator is diagonalized for time propagation; its state is entropy and
strain, rather than the oracle temperature/strain availability variables.
Independent cell refinement checks the reference.

Calibration seed 9313 gives k=144.95876 W/(m K), reduced chi²=0.92519 and fixed
record uncertainty 0.00137875 K. Noise seed 19313 supplies 256 calibration
realizations. Every fit passes; the largest conductivity error is 0.512%.
Oracle hidden RMSE is at most 0.000532 K for the frozen calibration and at most
0.000687 K at the observed fitted-parameter extrema. Shortcut hidden RMSE is
0.03305–0.04444 K, and stays above 0.03302 K at those extrema. This is a
256-fit Monte Carlo followed by parameter-extrema hidden checks, not an
exhaustive uncertainty bound.

The grading remains reduced calibration chi² <1.5, conductivity error <3%, and
hidden group RMSE <0.025 K. The finite-volume 128→256-cell change is at most
3.08e-5 K; oracle/reference discrepancy is at most 0.000970 K, and the
96→128-mode change is at most 0.000241 K. Closed linear energy is conserved to
7.2e-12 J and final equilibrium agrees to 5.5e-11 K. Availability decreases
under both positive mobilities. A low-mode state reconstructs total stress
uniformly to relative 4.65e-6 in the oracle; the shortcut has order-one stress
variation despite satisfying its global clamp and power identities.

Run `scripts/validate_thermoelastic_rod.py`. Without `--generate`, it reads the
frozen public and private calibration copies; regeneration explicitly writes
both together. Scientific and local-control reports are in
`jobs/thermoelastic-rod-r3-validation/`, copied to `results/` for sharing.
The baseline is `scripts/thermoelastic_rod_baseline.py`. Single-thread BLAS is
set for predictable eigensolver timing. The 600/60-second agent/verifier limits
are unchanged. Peer review is recorded under `thermoelastic-rod-r3` in
`results/materials-hardening-physics-review.json`. Harbor controls and agent
measurements remain separate from the scientific checks reported here.

The actual local verifier gives oracle 7 passed (0.73 s) and shortcut 4 passed,
3 intended hidden failures (0.65 s). No control fails calibration, parameter
recovery, API shape, or finiteness.
