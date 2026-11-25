# Elastohydrodynamic film, revision 2

The coating has a known lateral modulus pattern, E(x)=E0 R(x), where R(x)=1+a cos(m k0 x+phase). The unknown E0 is shared by every experiment. Full plane-strain elasticity, a bonded bottom, zero surface shear and liquid normal traction define its surface compliance C. The coating is elastic and quasistatic; the liquid obeys the stated linear, leading thin-gap lubrication model. All fields are incremental about the preloaded reference described in the public apparatus. The rigid plate follows that reference so its gap is uniform before excess pressure is applied.

For downward surface displacement a=Cp, incompressibility and Poiseuille flow give

    C p_dot = (h^3/(12 eta)) p_xx = -K_flow p.

Initial pressure is prescribed after mechanical equilibration. Closing the preparation ports changes no initial field. Mean displacement, and hence liquid volume, is conserved; mean pressure generally changes in a patterned coating. Fixing mean pressure would impose an extra, incorrect constraint. Uniform pressure is a stationary state with a generally nonuniform displacement. Positive C and K_flow give decreasing elastic energy, d(p^T C p/2)/dt=-p^T K_flow p.

The completed shortcut retains the exact finite-thickness homogeneous compliance C_hom, including every Fourier mode and the uniform compression mode. It approximates the patterned response by

    C_shortcut = R^(-1/2) C_hom R^(-1/2).

This is a positive, reciprocal surface-compliance model. It uses the known contrast, phase and pattern spacing, conserves liquid volume and dissipates its own elastic energy. It is exact for every homogeneous calibration preparation. Its error is physical: local rescaling of surface compliance does not enforce strain compatibility and stress balance throughout the heterogeneous coating. Both controls solve their coupled fluid relaxation completely; the shortcut is not an incomplete ODE solver.

The oracle propagates the full first-order displacement/traction system through the coating. On a periodic x grid, with D the lateral derivative, unit-E0 Lame coefficients mu and lambda, L=lambda+2mu, and r=lambda/L, its blocks are

    A = [[0, -D], [-r*D, 0]],
    B = diag(1/(mu*R), 1/(L*R)),
    K = diag(-D*((L-lambda^2/L)*R)*D, 0),
    F = [[A, B], [K, -A.T]].

Bottom displacement is zero. Slice-wise compliance transfer through exp(F/64) avoids the growing solutions of a whole-layer transfer matrix. The final normal compliance block imposes zero top shear. The implementation uses 49 periodic points. It solves the generalized relaxation eigenproblem K_flow v=rate*C v; rates scale with E0 and initial amplitudes with 1/E0.

The verifier is independent. It minimizes the full bulk plane-strain elastic energy over periodic lateral fields and 16 polynomial displacement functions through depth, each vanishing at the bonded bottom. Its stiffness separately integrates volume, normal-strain and shear terms with the actual local modulus. Surface pressure and zero shear are natural boundary conditions. It forms the normal compliance from this bulk variational problem, then evolves pressure with a direct matrix exponential. It does not import the oracle or its traction-transfer algorithm. Its lateral grid has 33 points.

Calibration uses homogeneous coatings with pressure modes 1 and 2, several nonzero wavelengths and times, two depths and two gaps. Thus it already checks the full homogeneous finite-wave response. There are 144 observations with Gaussian sigma .04 nm. E0 is identifiable from the displacement amplitude and relaxation rate; the controls have exactly the same calibration predictions. The data seed is 15124, and 256 additional calibration-noise draws use seed 25124. Ordinary validation reads checked-in public/private copies. Only an intentional `--generate` call updates both copies.

The hidden groups test mixed output harmonics, pattern spacing, and rotation of the pattern relative to preparation and detector. Inputs remain inside the public ranges. Scores are group RMSE divided by the reference group RMS, with a 1e-6 nm floor and limit .04. The current fitted E0 is 1.09979173 MPa versus 1.1 MPa, and calibration chi-squared is 1.01975. Oracle hidden scores are .000232, .000249, .000279; shortcut scores are .3225, .4821, .07227. All 256 calibration fits pass; the largest parameter error is .0464%. At the observed fit extrema, oracle hidden error remains below .000684 and shortcut error above .0719. These are 256 noisy calibration fits plus hidden sensitivity at their parameter extrema, not 256 independent hidden-data simulations.

Independent-reference disagreement is at most 2.60e-6 nm; selected 33x16 to 41x20 reference refinement changes predictions by 6.53e-8 nm, and oracle 49 to 65 refinement by 1.29e-11 nm. Homogeneous calibration equivalence is within 6.65e-12 nm. Additional checks cover positive compliance, reciprocity, bottom clamping, top normal/shear traction, weak elastic residual, translated preparation and pattern, zero forcing, uniform-pressure stationarity, input/parameter corners, and both controls' volume conservation and energy dissipation. The traction check uses 28 depth functions and has error 7.24e-8 for a unit load. The pressure-mean check confirms that neither control imposes a spurious zero-mean constraint.

Local public/private tests give oracle 7 passed; shortcut 4 passed and exactly the 3 intended prediction groups failed. The report, including local verifier output, is `results/elastohydrodynamic-film-r2-validation.json`. The independent source review is the `elastohydrodynamic-film-r2` entry in `results/materials-hardening-physics-review.json`. Harbor results for this revision are pending; passing scientific controls do not establish agent difficulty.

Revision 1, its full source, six reviewed agent trials and manifests are preserved in `archives/elastohydrodynamic-film-r1`. All 3 plain and 3 hinted revision 1 agents repaired the local-column model. Revision 2 changes the apparatus to a patterned coating and supplies the correct homogeneous nonlocal response in its starter. It is evaluated separately.

Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_elastohydrodynamic_film.py --noise-trials 256` with the pinned NumPy 2.3.3/SciPy 1.16.3 environment. The revision 2 Docker image also pins BLAS/OpenMP to one thread. Add `--generate` only before an intentional new freeze.
