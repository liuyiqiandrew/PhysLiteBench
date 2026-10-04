# Dielectric flow, revision 1 — unevaluated backup

This task is preserved as a backup. No Harbor control or solving-agent batch was launched. Scientific validation and local verifier controls do not establish that an agent will fail. The benchmark reached its requested ten zero-score tasks while this backup was being finished.

The externally maintained permittivity and conductivity fields define a steady Ohmic electrical problem. The potential correction is periodic, the mean electric field is prescribed, and div(sigma E)=0. This implies

    rho_free = div(epsilon E)
             = E dot [grad(epsilon) - epsilon*grad(sigma)/sigma].

The incompressible liquid's mechanical electric force, after absorbing electrostriction into pressure, is the divergence of the Maxwell stress:

    T = epsilon*(E E - |E|^2 I/2),
    div(T) = rho_free E - |E|^2 grad(epsilon)/2.

Calibration sets sigma=100 epsilon. Free charge is then exactly zero, but a crossed two-dimensional permittivity pattern can still generate dielectric-gradient forcing with nonzero curl. The completed shortcut solves the electrical problem correctly and includes this dielectric-gradient force for every input. It omits the free-charge force when conductivity and permittivity vary independently. Thus it already predicts nonzero calibrated circulation and uses all controls; it is not a zero-output or unfinished-physics baseline.

The pattern is fixed in laboratory coordinates by external material control. Charge convection and hydrodynamic feedback on current are excluded by the prescribed small electric-Reynolds-number limit. The reported sampled ratio of charge-convection to relaxation scales is at most .00533. This is a sampled consistency check, not a proof over every continuous public input. The uniform mechanical reaction enforces zero mean flow for both constitutive models. It is necessary because the approximate dielectric-gradient force can have nonzero mean, whereas the full periodic Maxwell-stress divergence cannot. The reaction performs no work at zero mean flow. All pressure fluctuations are periodic.

The oracle solves the scalar current problem on a 49-point periodic grid with Fourier derivatives and preconditioned conjugate gradients. It evaluates the free-charge and permittivity-gradient terms separately, then projects the force onto divergence-free Stokes flow. For fixed electrical controls, the velocity is exactly proportional to 1/viscosity. Weighted linear regression in reciprocal viscosity therefore identifies the single unknown whenever a measured coefficient is nonzero.

The independent reference uses conservative finite-volume face currents, a pinned potential gauge, and a sparse direct solve. It forms the Maxwell tensor from the resulting field, differentiates its normal and shear stresses onto velocity faces, and solves the staggered Stokes equations. Detector phases include their half-cell offset. Richardson extrapolation of 48 and 96 cells removes the leading spatial error. It does not import the oracle or its electric-force decomposition.

Calibration contains 144 measurements: 24 distinct preparations repeated six times with independent noise of standard deviation 2e-6. The data seed is 15151; the additional noise seed is 25151. The calibrated velocity-coefficient RMS is .00103247. The two completed models' calibration predictions agree within 3.09e-18, and free charge on proportional profiles is below 5.73e-13. The fitted viscosity is 1.09999247 versus 1.1, with reduced chi-squared 1.04731.

Hidden groups probe crossed profiles, conductivity/permittivity phase mismatch and oblique profiles. Scores are group RMSE divided by reference group RMS, with a 1e-6 floor in the verifier and limit .04. Oracle scores are at most 1.03e-5; shortcut scores are .986–1.096. All 256 noisy calibration fits pass, with largest viscosity error .0513%. At the observed fit extrema, oracle hidden error is below .000516 and shortcut error above .9857. This is a calibration-noise study followed by hidden sensitivity at the fitted extrema, not 256 independent hidden simulations.

Oracle/reference disagreement is at most 8.23e-8 in the reported velocity units; 96-to-192-cell refinement changes predictions by at most 4.07e-8. The extreme allowed-pattern check differs by 2.37e-8. Further checks cover current conservation, zero electric curl, incompressibility, zero mean velocity, Maxwell-stress equivalence, mechanical work and viscous dissipation, reversal of the applied field, translation, zero field and uniform material. Both force closures dissipate their mechanical power correctly; the approximation fails because its physical force is incomplete.

Local verifier controls are complete: oracle 7 passed; shortcut 4 passed and exactly the 3 intended prediction groups failed (each .56 seconds). The final source peer review passed and is copied to `archives/unevaluated/results/dielectric-flow-physics-review.json`. Its independent off-grid oracle/reference check differs by 2.35e-8, and its Maxwell-stress identity check by 3.76e-15.

The final report is `archives/unevaluated/results/dielectric-flow-validation.json`. The baseline and validator are under `archives/unevaluated/scripts`. Reproduce from the repository root with the pinned NumPy 2.3.3/SciPy 1.16.3 environment:

    OPENBLAS_NUM_THREADS=1 python archives/unevaluated/scripts/validate_dielectric_flow.py --noise-trials 256 --output /tmp/dielectric-flow-validation.json

Ordinary validation reads the checked-in equal public/private calibration copies. Only `--generate` intentionally overwrites both. Author notes, hints, solutions and private tests are excluded from the agent image by its explicit public-file Docker copies.

Primary physical context: [Electrohydrodynamic Flows in Nonhomogeneous Liquids](https://pubs.acs.org/doi/10.1021/ie0512237) studies flow driven by charge generated in conductivity gradients. The present constitutive model and calibration nullspace are specified independently.
