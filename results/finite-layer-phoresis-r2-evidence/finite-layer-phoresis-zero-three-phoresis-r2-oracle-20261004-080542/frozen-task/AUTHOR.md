# Navier-slip phoresis, revision 2

This is a substantive revision of the finite-layer-phoresis family. Revision 1 scored 1/3 passes, with two reviewed physical failures; its complete source, trials and fixed-viscosity diagnostics remain in `archives/finite-layer-phoresis-r1`. It is not counted as a new independent family. Revision 2 has no agent evaluations yet.

The sphere now has a specified Navier slip length. The supplied predictor solves the exact curved solute problem, retains the full no-slip force kernel, and uses the exact translating-sphere slip drag. Its approximation holds the phoretic thrust fixed when slip changes. Slip changes how the distributed interaction force is transmitted to the particle as well as its drag. Calibration on no-slip spheres validates all the curvature physics that separated the revision 1 controls.

## Mechanical derivation

Write the concentration perturbation as G f(r) cos(theta), with h=exp(U)f, sphere radius a and outer interaction radius R=a+width. Conservation of ideal-solute flux gives

    (r² exp(-U) h')' - 2 exp(-U) h = 0,
    h'(a)=0,  R h'(R)+2h(R)=3R.

The last condition matches the exterior solution r+C/r². Concentration does not depend on viscosity or slip in the specified zero-Peclet limit. The model solves this BVP in the rescaled layer coordinate; the verifier assembles its independent self-adjoint finite-element weak form.

Let beta=b/a and d=(1+2beta)/(1+3beta). Translation of an impermeable Navier-slip sphere has drag 6*pi*eta*a*d. The radial coefficient in its auxiliary translating flow is A*a/r-B*a³/r³, where A=3d/2 and B=1/[2(1+3beta)]. Combining the reciprocal identity with the equal-opposite reaction of the carried potential gives

    V/G = 2/(9 eta a d) * integral_a^R K_b(r) f(r) U'(r) dr,
    K_b(r) = r² - A*a*r + B*a³/r.

The correct kernel vanishes at r=a but has derivative 3a*beta/(1+3beta). At b=0 it becomes K_0=r²-3ar/2+a³/(2r). The completed shortcut uses K_0 with the correct d. It therefore uses every input and has exact calibration equivalence. Both models are finite and nonzero on the scored slip preparations; the control error is not an unfinished routine, missing input, or algebra error in a Navier drag formula.

The reference does not use this kernel or drag. It solves the fourth-order Stokes equation for a particle-frame streamfunction, with u_r=2psi*cos(theta)/r² and u_theta=-psi'*sin(theta)/r. Its inner conditions are

    psi(a)=0,
    (1+2b/a)*psi'(a)-b*psi''(a)=0.

The second relation follows from the stated tangential Newtonian traction, including its curvature term. Exterior matching keeps only r² and r^-1 solutions, excluding the Stokeslet and the growing r^4 solution. The velocity is -2[psi'(R)+psi(R)/R]/(3R). Direct integration of solution traction plus the potential reaction verifies force balance. The solution pressure already includes osmotic pressure; adding a second hard-wall osmotic term would double count it.

## Calibration, checks and evidence

There are 24 distinct no-slip sphere settings, each repeated six times: 144 readings with fixed instrument sigma=0.00003. Only the inverse viscosity is fitted. Its nonzero linear design norm proves identifiability over [.8,1.6], and noiseless endpoint recovery is checked. The true viscosity remains private. The ordinary starter has a fit stub and no default true value. Public and private data are identical; sigma does not encode the response.

The prediction gate remains 0.04 normalized RMS error. Fixed-viscosity physical repairs, if needed during trial review, must retain the fitted viscosity and replace only the force kernel. A solver that attempts the full slip mechanics but makes a sign, factor, boundary or indexing error must be distinguished from retaining the fixed-thrust approximation.

The 256-noise report gives oracle passes256/256 and shortcut rejection256/256. Nominal hidden errors are below5.3e-7 for the oracle and .3447–.3662 for the shortcut; the worst noise oracle error is3.35e-5. The minimum scored signal is .01395, so these are ordinary nonzero responses. Across98 domain/refinement cases the independent oracle/reference difference is5.14e-10 and reference refinement changes4.06e-10. Force-balance residuals are below6.4e-13. Geometric scaling, zero interaction, positive drag, nonnegative slip dissipation, b=0 equivalence and the planar thin-layer limit pass. The extra thin-layer check is an analytical limit outside the scored width range.

Local fresh-copy pytest controls finish in under one second each: oracle7/7; shortcut4pass plus3 intended finite-slip failures. Root will run the unchanged Docker controls and fresh agent trials. Successful scientific validation does not establish agent difficulty.

Reports are in `../../results/finite-layer-phoresis-r2-validation.json`, `../../results/finite-layer-phoresis-r2-local-controls.json` and `../../results/family-history.json`. The first validator attempt used bitwise equality for differently ordered but algebraically identical b=0 kernels; the observed2.22e-16 rounding difference is now bounded by1e-13. No scientific source, data or grading changed for that assertion correction.

Reproduce from the repository root:

    uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/finite-layer-phoresis-r2/scripts/validate_finite_layer_phoresis.py

Add `--generate` only to intentionally regenerate both calibration copies from the recorded seed. The stage uses the exact neutral instruction, pinned dependencies,600-second agent limit and60-second verifier limit.

## Primary background

Marbach, Yoshida and Bocquet, [Local and global force balance for diffusiophoretic transport](https://doi.org/10.1017/jfm.2020.137), JFM892,A6 (2020), develops local and global force balance for finite interaction layers and discusses relaxing no slip to partial slip. Its [author manuscript](https://sophie.marbach.fr/documents/JFM_2020_Marbach.pdf) supplies the mechanical background. The slip kernel and Navier streamfunction boundary used here were independently derived and cross-checked against each other; the paper is background support, not a copied verifier.
