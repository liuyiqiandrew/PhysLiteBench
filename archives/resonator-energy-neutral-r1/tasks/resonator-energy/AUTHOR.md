# Resonator energy, revision 1

The supplied predictor solves the complete harmonic scattering problem for an elastic slab with local spring-mass attachments. It then treats the positive frequency-dependent effective inertia as a physical mass density in the ordinary nondispersive energy formula. The wave field, reflection, transmission, interface conditions and host strain energy are correct. The missing physics is the kinetic and spring energy held by internal motion, which is not determined by substituting the dynamic effective density into the host Hamiltonian.

The public apparatus is a unit-cross-section one-dimensional host between known semi-infinite leads, with attached resonator mass and spring stiffness per reference length. The detector measures cycle-averaged total mechanical energy inside the slab. The README gives no reduced energy correction, prescribed solver, or restriction on implementation. The standard neutral instruction is unchanged. Calibration uncertainty is constant instrument noise and does not encode the noiseless response.

## Derivation

With host density 1, stiffness K, resonator mass density m and attachment stiffness kappa=m*Omega^2, the equations are

    u_tt = K*u_xx + kappa*(w-u),
    m*w_tt = -kappa*(w-u).

For exp(-i*omega*t), w=Omega^2*u/(Omega^2-omega^2) and the host wave equation has

    rho_eff = 1 + m*Omega^2/(Omega^2-omega^2).

All hidden frequencies are below resonance; rho_eff and the shortcut energy are positive throughout the public domain. The scalar host scattering equation is exact for this defined continuum model, not a long-wave homogenization whose unmentioned corrections enter the verifier.

The physical cycle-averaged energy is the spatial integral of

    [omega^2*|u|^2 + K*|u_x|^2
     + m*omega^2*|w|^2 + kappa*|w-u|^2]/4.

After eliminating w, the coefficient of omega^2*|u|^2 is rho_eff+omega*d(rho_eff)/domega. The shortcut retains only rho_eff. Its energy error is positive and vanishes when attachments are absent or in the zero-frequency limit. No heat, dissipation, ambiguous effective stress, or external-drive energy is counted by this detector.

The readout region includes the host and local attachments but excludes both leads. The incident wave is fixed at the left face, no right-incident wave is present, and both displacement and traction are continuous. Specifying the harmonic scattering state removes arbitrary free resonator motion. Physical support and readout information are sufficient to derive the result without naming dispersion energy in the public text.

This mechanism concerns mechanical energy stored in eliminated internal degrees of freedom while all wave fields remain correct. It differs from retained hydrodynamic heating and entropy tasks, which concern reservoir heat or irreversible fluctuations, and from the archived quantum-residence task, whose observable differs from phase delay because of boundary interference.

## Independent reference

The oracle solves a two-by-two interface system and integrates squared standing-wave fields analytically. The verifier instead assembles linear finite elements for both host and resonator displacement fields, with independent radiation boundary terms from the leads. It solves the joint block system without inserting rho_eff, then adds host kinetic, host strain, resonator kinetic and relative-spring energies using consistent mass and stiffness matrices.

At 1024 cells, oracle/reference hidden disagreement is at most 6.06e-6 relative. Doubling the reference mesh changes hidden predictions by at most 4.54e-6. Full-domain corner error is 1.40e-5, far below the 0.04 prediction tolerance. Lossless reflection/transmission balance holds to 6.62e-14. Analytic spatial integrals agree with independent quadrature to 3.47e-18. An impedance-matched resonant case satisfies stored energy=incident energy flux times group delay to 2.23e-16 relative. Both controls obey quadratic amplitude scaling and share the correct quasistatic limit.

The coupled finite-element solve has an absolute residual up to 5.55e-10 in the algebraic resonator displacement relation at a corner. A validator bound of 1e-8 covers that numerical conditioning error; the first attempted bound 1e-10 was unnecessarily strict. This adjustment affected only the author-side numerical identity check. The task equations, calibration data, prediction reference and grading did not change.

Root's separate 24 off-grid traveling-wave/Hamiltonian quadratures agree with the oracle to 3.5e-18. Their finite-element refinement check shows the expected second-order convergence. [Root peer review](../../results/root-hardening-physics-review.json) records the independent source hashes.

Primary background: [Energy Localization through Locally Resonant Materials](https://pmc.ncbi.nlm.nih.gov/articles/PMC7372459/), Materials 13,3016 (2020), sections 3.1–3.2, distinguishes microscopic mechanical energy in a resonant medium from its effective wave parameters. The task uses the simpler explicitly specified spring-mass continuum and derives its energy directly from that Hamiltonian.

## Calibration and controls

There are 108 calibration records with no attachments, at varied slab lengths, frequencies and incident amplitudes. The slab stiffness differs from the known lead stiffness, so calibration includes reflected and standing waves. The true host stiffness is 1.13, its allowed interval is [0.7,1.6], fixed sigma=2e-7, calibration seed 38021 and noise seed 48021. Noiseless objective scans and fits at 0.701,1.13 and 1.599 verify a single minimum across the entire interval. Both completed controls agree exactly on every calibration input.

Nominal stiffness fit is 1.1300104635 and reduced calibration chi-square is 1.071553. The oracle hidden group errors are at most 3.37e-6; shortcut errors are 0.460,0.565 and 0.602. Across 256 independent noise realizations all parameter/calibration checks pass, all oracle hidden groups pass, and every shortcut fails. Worst oracle hidden error is 4.20e-5 and minimum shortcut error is 0.4602. Maximum parameter error is 4.71e-5 relative. Local isolated pytest gives oracle 7/7 and shortcut 4 pass/3 intended hidden failures.

Private prediction errors use group RMS divided by the true group RMS and require <0.04. Instrument noise, fitted-parameter uncertainty and reference discretization are all substantially smaller. Parameter recovery requires <3% and reduced calibration chi-square <1.5. These checks establish a sound physical comparison, not model difficulty. Root schedules Docker controls and the frozen Luna screen separately; no model outcome is claimed here.

Reports: [science](../../results/resonator-energy-validation.json), [local controls](../../results/resonator-energy-local-controls.json), [source provenance](../../results/resonator-energy-source-provenance.json).

Reproduce with `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_resonator_energy.py`. Use `--generate` only for intentional simultaneous regeneration of public and private calibration copies.
