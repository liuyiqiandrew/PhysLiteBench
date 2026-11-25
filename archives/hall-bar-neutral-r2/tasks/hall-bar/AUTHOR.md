# Hall channel revision 2

The neutral-instruction version of the homogeneous Drude bar passed all three Luna high trials. Its full task, baseline and reviews are preserved in `archives/hall-bar-neutral-v1`. This revision changes the physical apparatus to a closed continuum moment fluid. The neutral instruction, mobility API, 600/60-second limits and 4% hidden-error cutoff are unchanged.

Let u(y) be longitudinal velocity, tau=0.4 the known second-moment relaxation time, and p the traceless departure of the kinetic momentum flux from equilibrium. Affine transport, rotation of both velocity factors by the Lorentz force, and relaxation give

    p_xx = 2 B tau p_xy,   p_yy = -p_xx,
    p_xy = -nu u',         nu = tau/(1+4 B^2 tau^2).

Longitudinal balance is `(nu u')' - u/mobility + Ex = 0`, with no-slip endpoints. The transverse electrochemical field is `B u + p_yy'`. Thus a voltage between contacts includes both the integral of B u and the difference of p_yy at the contacts. The latter is a nondissipative normal-stress contribution. It remains present for a uniform magnetic field and also includes gradients of B for a spatially varying field.

The starter and completed shortcut already solve the correct variable-field longitudinal boundary problem with the exact even magnetoviscosity. Their transverse voltage retains only the Lorentz contribution. At zero magnetic field, their current predictions are identical to the oracle for every mobility, so calibration cannot expose the missing transverse stress. This is a physical constitutive omission, not a solver or fitting defect.

The public apparatus defines a closed continuum model, including the walls, rather than claiming kinetic validity at arbitrary Knudsen number. The readout is an electrochemical voltage, so an unspecified isotropic pressure gradient cannot change the answer. Neither wall slip nor microscopic boundary layers are hidden parameters.

The private reference solves a first-order boundary-value problem for velocity, shear momentum flux, integrated current and integrated Lorentz force. The oracle instead solves a tridiagonal finite-difference momentum operator. Their off-grid agreement is within 1.7e-5 relative error. Separate checks establish tensor balance at the BVP accuracy, positive dissipation, steady power balance, magnetic-field reversal and BVP refinement.

Calibration contains 120 independent zero-field current measurements at fixed instrument sigma=1e-5. The larger sample keeps the unchanged reduced-chi-square gate stable; 256/256 calibration noise draws pass. All oracle hidden errors are below 0.000511. All shortcut groups remain above 0.4714, against the unchanged 0.04 threshold. The fixed seed is 12071 and noise seed is 44211. See `results/hall-bar-r2-validation.json`, `results/hall-bar-r2-local-controls.json`, and the `hall-bar-r2` entry in `results/constraint-hardening-physics-review.json`.

The previous task's agent scores are historical; new difficulty is determined only by the frozen neutral-r2 evaluations. Reproduce science with `python scripts/validate_hall_bar.py` in the pinned NumPy/SciPy environment. Use `--generate` only to intentionally regenerate both calibration copies.
