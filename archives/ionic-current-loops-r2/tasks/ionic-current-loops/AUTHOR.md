# Ionic current loops, revision 2: closed-ring circulation

This is a scientific revision of the archived ionic-current-loops family, not a new family. It is staged and scientifically validated, with no model evaluations yet. The original revision remains under `archives/screened/tasks/ionic-current-loops`; its six reviewed outcomes and source hashes are preserved by reference in this stage's `results/predecessor-r1.json`.

Revision 1 passed 3/3 plain and 3/3 hinted trials before the uniform neutral instruction patch. Those trials solved the two-dimensional periodic scalar-potential equation. One plain trace explicitly used unused solver imports as a clue. The old hidden preparations have two independent cosine wavevectors and an inversion center, so they test local circulation but have zero mean winding current. Two passing traces described the one-dimensional zero-current reduction as generally exact. Those statements do not hold for an arbitrary phased profile on a closed ring. These historical passes are retained, not relabeled or replaced.

The new experiment is a closed thin ring with positive out-of-phase ternary profiles. Locally, any one-dimensional electric field is curl-free. A field can still fail to be the derivative of a single-valued potential around the full loop. With reduced diffusion ratios d_i, define

    g = sum(z_i*d_i*c_i), sigma = sum(d_i*c_i),
    J = -g' + sigma*E.

Quasineutrality requires J to be spatially constant. Electrostatics gives the global loop condition, and therefore

    J = -integral(g'/sigma)/integral(1/sigma),
    E = (g'+J)/sigma.

The common physical diffusivity multiplies all species fluxes and J; it cancels from E. The completed source instead imposes J=0 pointwise. It uses the exact local conductivity, full ionic gradients and conservative species fluxes; the error is a local open-circuit closure applied to the closed ring. It contains no unused solver imports. Both controls conserve species amounts and local charge, and both dissipate the ideal mixture free energy. The latter facts cannot establish global electrostatic integrability.

Calibration uses reflection-symmetric profiles. Here g'/sigma is odd, J=0 exactly, and both controls are identical. Fourier concentration rates are linear in the one unknown D, so weighted least squares is globally identifiable whenever the response vector is nonzero. The checked information is 4.88e8; the true D is 1.13 in [.5,2]. There are 288 independent records, 12 settings repeated 24 times, with fixed sigma=.001. Seeds 117821 and 117829 generate data and the 256-noise validation respectively. No uncertainty depends on the true response or unknown parameter.

The readout remains an instantaneous concentration Fourier rate, as in the predecessor. It does not directly ask for the hidden circulation or a voltage constraint. Revision 2 changes the topology-sensitive preparation and diffusion ratios. Early prototypes retained ratios (1,.08,3) and gave only .05–2.23% whole-profile slope gaps over the explored amplitudes, so they were not used for a task claim. Moderate ratios (1,4,1) with positive profiles and ordinary second-harmonic phase readouts give useful nonzero signals and larger gaps. The original prototype/report and this development history are preserved. All ratios, profile limits and detector settings are disclosed publicly. Tolerance .04 is unchanged from revision 1; it was not tightened to manufacture failures.

The oracle uses a spectral concentration derivative and the exact global loop integral. The independent reference never inserts the global-current formula: it solves charge continuity for a periodic scalar potential on a conservative face grid with one gauge value fixed, then forms face fluxes and concentration rates. Richardson extrapolation of 512 and 1024 cells supplies verifier values. This independent boundary-value construction tests the global constraint and field sign.

Validation in `staging/ionic-current-loops-r2/results/ionic-current-loops-r2-validation.json` reports:

- Both controls fit D=1.12992076 with reduced chi-square 1.08460. Physical hidden errors are about 7.01e-5; shortcut errors are .42075,.09122,.42075. The aligned-profile anchor group passes both.
- All 256 noise realizations pass calibration, parameter recovery and oracle prediction. All 256 reject the shortcut; minimum diagnostic error .09107 at tolerance .04. Worst oracle error .000138.
- Spectral/independent-reference error is 6.16e-9. Refining the reference changes predictions by at most 5.77e-9; refining the oracle changes them by 2.11e-15. Strong public-domain corners also pass.
- Integrated mass-rate error 2.72e-16, local charge-rate error 2.08e-13, global mean-field error 4.80e-17, spatial charge-current variation 3.11e-15. The free-energy dissipation identity agrees within 1.78e-15. Minimum hidden concentration is .16. Independent potential-solve loop current agrees within 4.40e-6 before extrapolation; reflection reverses it correctly.
- The completed shortcut's spurious mean field reaches .1246 despite zero local charge current. Its calibration and aligned preparations have the correct zero-current limit.
- Local oracle 9/9 passes; shortcut 6 pass/3 intended prediction failures. Each local verifier completes in under one second. Docker controls and fresh model trials are left to the root runner.

Reproduce existing-data validation from the repository root:

```bash
uv run --no-project --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/ionic-current-loops-r2/scripts/validate_ionic_current_loops.py
```

Use `--generate` only for deliberate data regeneration. The task has the exact neutral full-replacement instruction and standard 600s/60s limits. Public inputs specify the closed ring, quasineutrality, independent preparation and measurement; they give no loop-current formula, solver mandate or corrective hint. Only public files enter the image. The grader checks numeric outputs, shapes and the inferred shared parameter. Difficulty must be established by three fresh frozen trials, not inferred from the prototype.

Primary context: [Warren, Non-Faradaic electric currents in the Nernst–Planck equations (2020)](https://arxiv.org/abs/1912.06016) explains why electroneutral transport can admit solenoidal ionic currents without electrodes. The present one-dimensional ring's winding sector follows independently from charge continuity and electrostatic integrability; it is not claimed to be that paper's crossed-gradient example.
