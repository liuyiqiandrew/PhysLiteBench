# Hall channel revision 3

The full neutral-r2 task, controls, validator, scientific reports, three reviewed passing Luna-high trials and their terminal evidence are preserved in `archives/hall-bar-neutral-r2/manifest.json`. Revision 3 keeps that continuum apparatus, API, neutral instruction, runtime limits, hidden experiments and grading. It strengthens the starter and calibration without changing a prediction cutoff.

For the stated kinetic momentum-flux model, with tau=0.4,

    p_xy = -nu u',   nu = tau/(1+4 B² tau²),
    p_yy = 2 tau B nu u',
    (nu u')' = u/mobility - Ex,
    Ecy = B u + p_yy'.

For a spatially uniform magnetic field, longitudinal balance eliminates the velocity curvature and gives the complete Hall response

    Ecy = B [u + 2 tau (u/mobility - Ex)].

The starter now supplies this full homogeneous response and the correct conservative variable-viscosity velocity solver. It evaluates that local homogeneous relation at the actual B(y). This is a physical constitutive approximation: it treats a response derived in a homogeneous medium as a local force relation. It is not an incorrectly implemented product derivative. Both magnetoviscosity and the uniform-field transverse stress are already present.

For a graded field, taking the divergence of the local stress adds

    2 tau B'(y) nu(y) u'(y).

The oracle retains the stress difference between voltage contacts. The independent reference solves the first-order boundary problem for velocity, kinetic shear flux, integrated current and integrated Lorentz force. Uniform-field predictions in both controls use the same exact hyperbolic-function solution, making their calibration equivalence exact for every allowed mobility. General profiles retain the finite-difference oracle and independent BVP reference.

Calibration now contains 120 zero-field current records and 64 uniform nonzero-field current/voltage records. All use fixed instrument sigma=1e-5, independent of mobility and signal, with the unchanged calibration seed 12071. Calibration generation is deliberate and updates both identical public/private copies. The README changes only its description of those measurements. The apparatus, neutral instruction, public test, private grading functions, hidden experiments, task configuration and Docker image are unchanged. The hint now points to the divergence of local momentum flux in the actual field profile.

`results/hall-bar-r3-validation.json` records all 256 independent noise draws. Both controls fit mobility 0.649940748 and reduced chi-square 1.03848 on the frozen data. Across noise draws, every calibration/parameter check passes and the oracle's largest hidden error is 0.000188. The shortcut's uniform-field group passes; its gradient and modulated groups remain above 0.27177 and 0.40319, respectively, against the unchanged 0.04 gate. Group-specific ranges are reported rather than claiming that every hidden group fails.

Independent BVP agreement, tensor balance, positive longitudinal dissipation, power balance, field reversal and BVP refinement pass. A separate integral check verifies that the omitted term above accounts for the shortcut discrepancy to finite-difference accuracy. The teammate review is `hall-bar-r3` in `results/materials-hardening-physics-review.json`. The private harness and complete source hashes are in `results/hall-bar-r3-local-controls.json` and `results/hall-bar-r3-source-provenance.json`.

No agent outcome is claimed for this revision before its frozen evaluations complete. Run `python scripts/validate_hall_bar.py` in the pinned NumPy/SciPy environment. Use `--generate` only to intentionally regenerate calibration.
