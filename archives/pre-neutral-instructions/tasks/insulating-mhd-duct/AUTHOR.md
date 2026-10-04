# Insulating conducting duct, revision 1

This task tests cross-sectional electric-current closure and its feedback on
velocity. The scalar local-drag shortcut solves its own equations correctly.
Calibration uses B=0, where the potential and magnetic force vanish; pressure
startup readings identify the single viscosity parameter under both models.
Only those projections of the coupled system are observed in calibration.

With velocity u in +z and imposed field B in +y, the inductionless equations are

    Jx = sigma*(-phi_x-u*B), Jy = -sigma*phi_y,
    div(J) = 0, hence Laplacian(phi) = -B*u_x,
    rho*u_t = eta*Laplacian(u) + g + B*Jx.

All walls are no-slip and electrically insulating. Consequently the normal
potential derivative vanishes there, but the interior potential need not be
constant. There is no imposed axial electric field. The shortcut sets phi=0
and uses a scalar -sigma*B^2*u force. In a finite rectangular cross section,
this loses both charge conservation and the redistribution of momentum by the
induced electric field. The prescribed geometry is not a one-dimensional
Hartmann channel with an independently chosen external circuit.

The oracle expands velocity in normalized Dirichlet sine modes. Reflection
symmetry of a uniform pressure drive restricts both indices to positive odd
integers. For each x index m, potential uses the matching Neumann cosine in x
and even Neumann cosines q in y. The sine-to-cosine projection in y is

    A_n0 = 2*sqrt(2)/(pi*n),
    A_nq = 4*n/[pi*(n^2-q^2)]  for positive even q.

Eliminating potential gives the positive magnetic dissipation operator
I-kx^2*A*diag[1/(kx^2+ky_q^2)]*A^T. It is added to viscous dissipation; the
resulting symmetric modal system is propagated exactly in time after the
pressure step. The starter retains only the identity part of that operator.

The independent reference uses cell velocities, face currents, and a nodal
potential gauge. Charge balance and momentum are solved together in a sparse
Crank-Nicolson block system at every step. Face interpolation and divergence
are adjoints, preserving the mechanical/electrical dissipation identity.
All boundary currents are zero. Two spatial resolutions are combined by
Richardson extrapolation; a further refinement checks that result. At B=0,
an independently extended rectangular Poiseuille sine series supplies the
calibration reference. All reference code is self-contained inside tests.

Hidden fields 0.16, -0.32, and 0.6 T test mean and interior-point velocities
through the startup transient. Each group's score is norm(error)/norm(truth),
with a 0.04 limit. The pressure drive and reference norm are nonzero. All
outputs have units m/s; fitted viscosity has units Pa s. Constant temperature,
fully developed laminar flow, negligible magnetic Reynolds number, and absent
Hall/thermoelectric effects are explicit apparatus assumptions.

Calibration seed 9311 uses eta=0.01 Pa s and sigma=3.62830e-7 m/s. The fitted
eta is 0.009997797 and reduced chi-square 0.96236. Oracle hidden errors are
at most 0.000159; completed-shortcut errors range from 0.4339 to 0.8510.
Reference spatial refinement changes group predictions by at most 9.27e-5,
temporal refinement by 3.20e-5, and oracle/reference disagreement is at most
6.24e-5. Independent discrete charge balance and mechanical-versus-Joule power
identities hold to relative errors below 9e-16 and 2e-16 respectively.

Noise seed 19311 generates 256 calibration realizations; every fit passes.
Maximum relative viscosity error is 0.313%. Hidden checks at the observed
parameter extrema give oracle errors below 0.00232 and shortcut errors above
0.4332. This is a 256-fit Monte Carlo followed by extrema sensitivity checks,
not 256 full hidden evaluations or an exhaustive noise bound.

Run `scripts/validate_insulating_mhd_duct.py`; it writes
`jobs/insulating-mhd-duct-validation/summary.json`. Checked-in calibration is
unchanged unless `--generate` is explicitly requested. Both copies regenerate
together. The completed shortcut is `scripts/insulating_mhd_duct_baseline.py`.
A teammate reviews the current-closure signs, spectral projection, insulating
boundary treatment, and independent reference before Harbor. Agent evaluation is complete: 0/3 unhinted and 3/3 hinted. All unhinted failures retain the local-drag physical closure despite correct calibration.

## Reviewed evaluation

The frozen revision scores 0/3 unhinted and 3/3 hinted on GPT-5.6 Luna high, Codex CLI 0.154.0 and Harbor 0.21.0, with 600/60-second agent/verifier limits. All three ordinary failures pass calibration and parameter recovery and are reviewed physical-model failures. Fixed-parameter repairs recover the correct hidden predictions. See `results/candidates.json` and `results/hardening-status.json` for job IDs, per-trial reviews and matched-source checks. These are development-batch outcomes, not a future failure-rate guarantee.
