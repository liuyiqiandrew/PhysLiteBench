# Insulating conducting duct, revision 2

This revision tests whether two calibrated transverse magnetic responses can
be added independently. It preserves the neutral instruction verbatim. The
complete preceding task, controls, validator and three neutral trial reviews
are in `archives/insulating-mhd-duct-neutral-v1`, with a SHA256 manifest. That
revision passed 3/3 neutral trials; one passing final model also incurred an
agent timeout. Its earlier prompt-dependent results remain separate evidence.

The apparatus is unchanged apart from permitting an oblique uniform field
B=(Bx,By,0). With velocity u along z, the inductionless equations are

    J = sigma*(-grad(phi) + u*(-By,Bx)),  div(J)=0,
    rho*u_t = eta*Laplacian(u) + g + By*Jx - Bx*Jy.

No-slip insulating walls, no external current, no imposed axial field and
constant material properties determine the cross-sectional current problem.
The electrostatic potential is determined only up to an irrelevant constant.
The local velocity may be signed; the pressure gradient uses the signed z
axis convention. Current has units A/m² and velocity m/s.

In the simply connected rectangle, write J=sigma*(chi_y,-chi_x). The insulating
wall condition gives a constant boundary value of chi, set to zero. Taking
curl of Ohm's law yields

    -Laplacian(chi) = Bx*u_x + By*u_y.

The Lorentz force is sigma*(Bx*chi_x+By*chi_y). Eliminating chi gives a positive
magnetic dissipation matrix Q(B)=D_B^T K^(-1) D_B, where K is the Dirichlet
negative Laplacian and D_B=Bx*D_x+By*D_y. Thus

    Q(B) = Bx² Qx + By² Qy + Bx*By*(Dx^T K^-1 Dy + Dy^T K^-1 Dx).

The completed shortcut uses Bx² Qx+By² Qy. Each term is the correct insulating
current response for one applied axis. This is a coherent positive independent
channel approximation: each axis current satisfies charge balance and has a
nonnegative Joule dissipation. It omits the shared-current cross term. The
correct total current's mechanical/Joule identity is not claimed for that
approximation. Its own sum of channel dissipation is checked separately.

Calibration uses both nonzero field axes, (0,.3) T and (.3,0) T, with mean and
point velocity transients. It identifies viscosity while exactly eliminating
the mixed term. Both completed controls predict identical calibration values.
The shortcut uses every documented field component and already solves the
insulating-current boundary problem; it is not the preceding local-drag
shortcut. Hidden fields (.45,.45), (.45,-.45), and (.6,.2) T excite mixed-parity
velocity modes. Asymmetric interior point probes avoid reflection locations
where the mixed response can vanish. Mean readings also test the feedback
of those modes on the pressure-driven flow.

The oracle uses normalized Dirichlet sine velocity and current-streamfunction
modes. Uniform forcing and half-turn symmetry retain modes whose two indices
have the same parity. Derivative overlaps on a unit interval are

    <sin_p, d(sin_m)/dx> = 4*p*m/(p²-m²)

for opposite parity and zero otherwise; dimensional derivatives divide by W
or H. The shared off-axis matrix elements arise at the intersection of the
two derivative projections. Symmetric eigenmode propagation is exact in time
within the spatial truncation. Calibration axes use independent smaller
blocks, avoiding unnecessary full-matrix diagonalization during fitting.

The independent reference evolves cell velocities and solves the electrical
potential on the same cells using conservative face currents. With G the
face gradient and C=By*Cx-Bx*Cy the negative motional field operator,
J=-sigma*(G*phi+C*u), G^T*J=0. Crank-Nicolson momentum evolution and the
instantaneous current constraint are solved together. Boundary face currents
are exactly zero, wall viscous flux uses the half-cell distance, and one
potential is pinned only to fix gauge. Richardson extrapolation of two spatial
resolutions is checked against a further refinement. This formulation does
not use the oracle streamfunction or its mixed-mode coefficients.

The original grading thresholds remain unchanged: reduced calibration chi²
below 1.5, relative viscosity error below 3%, and norm(error)/norm(truth) below
4% for every hidden group. Calibration noise remains a single instrumental
scale equal to 0.4% of the largest noiseless calibration response, with seed
9311. This is not per-record response-shaped uncertainty. The model bounds
remain eta in [.006,.018] Pa s; the true value used for generation is .01.
Numerical errors and 256 noise fits are recorded in the revision-2 validation
report. Seeds, hidden extrema checks and convergence checks are reproducible;
these checks are not an exhaustive uncertainty bound.

Run `scripts/validate_insulating_mhd_duct.py`. It reads checked-in calibration
unless `--generate` explicitly updates both public and private copies. The
report is `jobs/insulating-mhd-duct-r2-validation/summary.json`; the completed
shortcut is `scripts/insulating_mhd_duct_baseline.py`. BLAS and OpenMP use one
thread in Docker to keep dense eigensolver runtime predictable. Dependencies,
600-second agent limit and 60-second verifier limit are unchanged. Harbor
controls and Luna evaluation for this revision are pending until separately
launched by the controller; scientific validation is not a difficulty claim.

## Completed scientific validation

Both controls fit eta=0.0100064345 Pa s with reduced chi²=0.96019. Oracle
hidden errors are 0.000365–0.000697; independent-axis errors are
0.09310–0.15271. All 256 calibration noise fits pass, with maximum viscosity
error 0.329%. At the observed parameter extrema, oracle hidden errors stay
below 0.002214 and shortcut errors exceed 0.09298. These are 256 noisy
calibration fits followed by hidden checks at the parameter extrema.

FV spatial refinement changes predictions by at most 9.28e-5 relative; time
refinement gives 3.05e-5. Oracle versus refined FV error is at most 3.26e-4,
and velocity-mode refinement changes predictions by at most 5.06e-4. These
errors are below the measurement uncertainty and far below the 4% threshold.
Charge balance and mechanical/Joule dissipation agree to about 1e-15. Both
magnetic damping matrices have positive eigenvalues in the checked finite
subspace, and the independent-channel power identity also passes.

The actual local verifier gives oracle 7 passed (14.3 s) and shortcut 4 passed,
3 intended hidden failures (13.4 s). Shareable reports are
`results/insulating-mhd-duct-r2-validation.json` and
`results/insulating-mhd-duct-r2-local-controls.json`.
