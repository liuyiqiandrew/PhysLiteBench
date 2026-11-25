# Elastohydrodynamic film, revision 1

The apparatus measures liquid-driven relaxation of a bonded elastic coating. Uniform pressure identifies its Young modulus but does not test lateral mechanical compatibility. The supplied predictor treats the coating as independent compressed columns, with compliance C0=d/(lambda+2mu). This is exact for all homogeneous calibration preparations. At finite wavenumber, lateral displacement and shear redistribute the load; the coating's normal compliance is a function of kd. The completed shortcut remains a positive, dissipative local-foundation model and solves its lubrication dynamics exactly.

For downward surface displacement a(x,t), linear liquid-volume conservation and Poiseuille flow give a_t=(h^3/(12 eta)) p_xx. Quasistatic elasticity gives a_k=C(k)p_k. Thus p_k and a_k decay at rate h^3 k^2/(12 eta C(k)). The observation is a_k(t)=p_k(0) C(k) exp[-h^3 k^2 t/(12 eta C(k))], in nanometres. The zero mode cannot relax because its mean liquid volume is conserved.

For Poisson ratio nu, q=kd and Young modulus E, the full finite-layer compliance is

    C(k)=2(1-nu^2)/(E k)
         * [(3-4nu)sinh(2q)-2q]
         / [(3-4nu)cosh(2q)+2q^2+5-12nu+8nu^2].

Its k=0 limit is d(1+nu)(1-2nu)/(E(1-nu)), matching C0. The half-space limit is 2(1-nu^2)/(E k). With nu=.48, true C/C0 at kd=.5,1,2,4 is approximately2.24,3.63,4.46,3.30. The calibration therefore cannot select the finite-wavelength law.

The public approximation is linear quasistatic elasticity coupled to leading thin-gap lubrication. Pressure dominates tangential liquid traction. Tangential elastic wall velocity gives a higher-order Couette flux in gap/depth and gap*wavenumber; it is consistently omitted from both controls. The coating thickness itself is not expanded relative to the pressure wavelength. Elastic and liquid inertia, nonlinear gap corrections and cavitation are excluded by the specified model. The maximum allowed deformation/gap ratio is about.022, based on a dense compliance scan at the extreme pressure, depth, modulus and gap; strains are also small. The prescribed pressure preparation already includes equilibrium elastic displacement. Closing the ports changes no initial field and conserves subsequent liquid volume.

The oracle evaluates the finite-layer Green response. The independent reference minimizes the full plane-strain bulk elastic energy over displacement fields U(z)sin(kx), W(z)cos(kx), with a basis that vanishes at the bonded bottom. Its stiffness contains the volumetric, normal-strain and shear terms separately; the top pressure supplies the boundary work, and zero shear traction is natural. It then integrates the gap-volume balance. A further check solves the independent four-component displacement/stress boundary-value transfer system with bottom U=W=0 and top shear0, normal stress-p. These routes agree at numerical precision. The reference uses no oracle compliance formula.

Current validation: energy-reference predictions agree within5.28e-11nm; increasing the elastic basis18 to26 changes predictions by2.39e-13nm. The independent stress transfer compliance agrees within5.78e-15 in dimensionless units. Uniform and half-space limits, input/parameter corners, positive compliance and elastic-energy/fluid-dissipation balance pass. A conservative periodic finite-volume flux calculation converges at second order (128/256 errors.002009/.0005022nm).

Calibration has240 observations with independent standard deviation.03nm; seed15123 generates the checked-in data. The parameter is uniquely identified because every calibration displacement is proportional to1/E. The256 additional noise realizations use seed25123. Ordinary validation reads checked-in public/private copies; `--generate` intentionally updates them together. All256 calibration fits pass; maximum modulus error is.0767%. This is a calibration-noise study plus hidden sensitivity at the observed parameter extrema, not256 independent hidden-data simulations.

Hidden metrics are group RMSE divided by reference group RMS (floor1e-6nm), with limit.04. Current oracle scores are.000513–.000544; shortcut scores are.818–.844. At the calibration-fit extrema the oracle stays below.000964 and the shortcut above.8176. Local public/private tests: oracle7 passed; shortcut4 passed and only the3 intended prediction groups failed. The peer source review is recorded in `results/materials-hardening-physics-review.json`; science and local controls are in `results/elastohydrodynamic-film-validation.json`. Agent evaluations are pending.

Primary context: [Chandler and Vella, Validity of Winkler's mattress model for thin elastomeric layers: beyond Poisson's ratio](https://doi.org/10.1098/rspa.2020.0551) examines when local-column response fails. The finite sinusoidal compliance also appears in [Joe et al., Effect of Surface Roughness on Adhesive Instabilities for the Elastic Layer](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2020.00031/full). The present preparation and transient observable are defined independently in the task.

Run `python scripts/validate_elastohydrodynamic_film.py --noise-trials 256` with the pinned NumPy2.3.3/SciPy1.16.3 environment. Add `--generate` only before an intentional new freeze.
