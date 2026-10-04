# Piezoelectric waves, revision 1

This task separates local fixed-electric-displacement elasticity from the
electrical constraint on an acoustic wave inside an insulating bulk. The
completed shortcut solves its three-dimensional acoustic eigenproblem
correctly and uses every tensor and propagation direction. Its constitutive
constraint is inappropriate for the stated wave.

Write B_kj=e_kij*n_i, d=n^T*epsilon*n, and g=B^T*n. For a displacement amplitude
u and scaled electric potential amplitude phi, the bulk equations are

    (Gamma_E - rho*v^2*I)*u + g*phi = 0
    g^T*u - d*phi = 0,
    Gamma_E = mu*I + (lambda+mu)*n*n^T.

Gauss' law constrains n dot D, while electroquasistatic curl E=0 makes E
parallel to n. Eliminating phi gives Gamma_E+g*g^T/d. The shortcut instead
uses Gamma_E+B^T*epsilon^-1*B, which enforces D=0 in all components. This is
the legitimate constant-D elastic tensor, but generally its electric field
has a transverse component that cannot belong to the specified plane wave.
The bulk correction is positive semidefinite. The local correction exceeds
it by another positive semidefinite matrix. Neither control relies on an
unstable stiffness or incorrect eigensolver.

The fastest axial branch has speed squared
(lambda+2*mu+e33^2/epsilon_z)/rho in both controls. The two basal transverse
branches have squared speeds mu/rho and
(mu+e15^2/epsilon_t)/rho in both controls. The other basal branch stays faster
throughout the allowed lambda range. Calibration includes all three branch
labels, with 48 axial records and 24 records for each basal transverse branch.
Only the axial speed depends on the unknown lambda, monotonically, and
identifies it. True lambda is 32 GPa; the calibration standard deviation is
6 m/s. Independent seeds are 12117 for the committed data and 22117 for noise
validation.

The hidden groups measure oblique lower and upper shear speeds, including
rotated crystal directions. The metric is group RMS prediction error divided
by group RMS physical speed, with a 0.03 limit. This is much larger than the
observed fit uncertainty and numerical error. The three shortcut errors are
0.3064, 0.2611 and 0.1678. The fitted oracle's maximum is 1.34e-5.

The verifier independently assembles stress and electric displacement from
unit displacement/potential amplitudes and solves a four-variable generalized
eigenproblem. It discards the electrostatic infinite eigenvalue and retains
the three finite acoustic eigenvalues. A potential scale balances SI units;
it does not change the physical eigenvalues. It does not reuse the oracle's
electric-field elimination. Parameter/direction corner agreement is
2.73e-12 m/s. Other checks cover the analytic axial limit, positive stiffness,
Gauss' law, electric curl, direction reversal and positivity of the difference
between the two constitutive corrections.

All 256 noisy calibration fits pass the chi-square and parameter criteria.
The maximum relative parameter error is 0.00374. Hidden checks at the observed
parameter extrema give oracle error at most 6.19e-5 and shortcut error at least
0.16779; these are sensitivity checks, not an exhaustive hidden Monte Carlo
claim. Local isolated pytest controls give oracle 7 passed and shortcut
4 passed/3 hidden failures. Agent difficulty has not been measured yet.

Run `python scripts/validate_piezoelectric_waves.py` to validate committed
data. Only `--generate` regenerates both public and private calibration copies.
The default report is `jobs/piezoelectric-waves-validation/summary.json`.
Harbor oracle and shortcut controls use the ordinary runner and
`--solution-model scripts/piezoelectric_waves_baseline.py`, respectively.
The hint is instruction-only. Docker copies only public environment files.

The bulk plane-wave distinction is described in
[Elastic-Wave Formulation for Electroelastic Waves in Unbounded Piezoelectric Crystals](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.7.1636).
The task uses the explicitly stated ideal material constants rather than
claiming to model a particular measured crystal.
