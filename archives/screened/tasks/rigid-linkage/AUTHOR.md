# Rigid linkage, revision 1

The task tests the equilibrium measure of an exactly constrained mechanical system. Calibration observes one free angle after physically clamping and equilibrating the other joint. Hidden experiments equilibrate both free angles. The unknown potential stiffness is identified by calibration; the unobserved kinetic coupling follows from the publicly specified rigid geometry.

For absolute rod angles q=(theta1,theta2), the inertia matrix is

    G(q) = [[r+1, cos(theta1-theta2)],
            [cos(theta1-theta2), 1]].

The canonical Hamiltonian is H=1/2 p^T G^-1 p+U. Integrating its canonical momenta gives

    integral exp(-p^T G^-1 p/2) dp = 2*pi*sqrt(det G),
    det G = r + sin(theta1-theta2)^2.

Thus the native angular measure of the rigid model is sqrt(det G) exp(-U) dtheta1 dtheta2. This is not an added correction to the true rigid model. A compensating Fixman potential would produce a different target ensemble. The public specification expressly chooses the canonical ensemble of the exact rigid Hamiltonian, excludes radial degrees of freedom, and distinguishes it from a stiff-spring limit. Relevant primary work on constrained mass-metric determinants is [Echenique et al.](https://arxiv.org/abs/q-bio/0512033).

The calibration clamp removes theta1's velocity and momentum before equilibration. It does not condition an already equilibrated two-angle distribution on theta1=0. The remaining kinetic coefficient G22=1 is constant, giving the ordinary von Mises result E cos(theta2)=I1(kappa f2)/I0(kappa f2). Varying f2 identifies kappa. The completed shortcut fits this exact calibration and correctly computes angular Boltzmann quadrature for its assumed flat-angle measure. Its only physical omission is the momentum-state density when both angles are free.

Hidden tests measure cos[2(theta1-theta2)] in unforced, aligned-field, and turned-field equilibrium preparations. Even with zero potential, absolute orientation remains uniform while relative orientation is biased by the kinetic measure. As r approaches zero the relative-angle moment approaches -1/3; for large r it approaches zero. These limits distinguish sqrt(det G) from its reciprocal or a flat measure. All hidden mass ratios, potentials, and observables lie within the public ranges.

The oracle uses 256-point periodic angular quadrature. The reference independently constructs the 4-by-2 Cartesian velocity Jacobian, pulls back the diagonal point-mass kinetic energy, and integrates the canonical Gaussian momenta through the Cholesky factor of G^-1. It then uses 192-point Gauss-Legendre angular quadrature. Direct nested momentum quadrature at four configurations independently verifies this Gaussian integral. The integration box [-12,12]^2 has a negligible omitted Gaussian tail for those cases; the inner quadrature resolves the conditional Gaussian peak. A first author diagnostic with unconstrained adaptive integration over infinite limits missed narrow displaced peaks; this numerical diagnostic was corrected before task evaluation. It did not affect the oracle, reference, calibration, or grading.

Calibration uses seed 9324, true stiffness 1.2, and 200 measurements with independent Gaussian errors at 0.6% of the largest calibration moment. The private parameter tolerance is 3%, and reduced chi-square must be below 1.5. Hidden groups use absolute RMSE with limit 0.025, much larger than the calibration and numerical uncertainties.

Checked-in data fit stiffness 1.19853519 (0.122% error) with reduced chi-square 0.98858. Oracle hidden error is at most 0.000104; the shortcut errors range from 0.21595 to 0.27848. All 256 additional noise realizations pass calibration and parameter checks; maximum stiffness error is 0.306%. At fitted-parameter extrema, oracle error stays below 0.000259 and shortcut error above 0.21594.

Independent Cartesian-momentum reference agreement is below 4.01e-12; 128 versus 192 Legendre nodes differ by less than 2.86e-9. Direct canonical-momentum quadrature has relative error below 9.37e-13. Rotational invariance is accurate below 4e-17, the clamped von Mises law below 4e-16, and normalization is exact to floating-point precision. The singular r-to-zero limit approaches -1/3 with 6.68e-5 error on the fixed periodic grid; this limit is outside the advertised r>=0.02 range.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 scripts/validate_rigid_linkage.py rigid-linkage
```

Only use `--generate` to intentionally replace both calibration copies. Report: `jobs/rigid-linkage-validation/summary.json`. Local isolated-copy pytest: oracle 7 passed; completed shortcut 4 passed and 3 hidden prediction groups failed. The materials reviewer approved the physical ensemble, source equations, clamped calibration, and independent reference. Harbor controls and the paired gpt-5.6-luna/high evaluation are coordinated by the parent; difficulty is not inferred from scientific validity.
