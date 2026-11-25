# Thermoelastic rod r4

Status: approved by independent final source review; frozen and ready for root evaluation planning. This isolated stage is not canonical and has not been evaluated by a model.

The revision preserves the r3 passers' correct uniform-total-stress mechanics and adds a thermally coupled Maxwell internal strain. The completed source keeps that internal relaxation but uses a locally relaxed caloric state. Its own heat equation is mathematically complete; its physical-energy and thermodynamic-reciprocity defects are documented privately. Both controls exactly share nonzero thermoelastic calibration at chi=0, including active mechanical relaxation, initial preparation and final equilibrium.

All 256 noise draws passed for the oracle and failed all three diagnostic groups for the completed source. The worst oracle error is .000145794 K and the smallest source error is .034632 K, with the unchanged .025 K gate. Across the full sampled conductivity range, the smallest diagnostic gap is .0297255 K. Weak and zero-coupling cases remain in the preserved prototype and domain validation.

The independent entropy-coordinate reference, per-graded-case refinement, 57 domain systems, global sampled identification checks, energy/dissipation identities and input audit pass. Fresh actual local tests give oracle 7/7 and source 4 passed/3 intended prediction failures in about one second each. The reference shares the finite-volume discretization family; independent constitutive elimination and separate bounded BDF checks provide the additional validation. No claim is made that the source preserves the material's physical energy or that the new task is empirically difficult.

The 18 task/control files, seven reports/logs and eleven immutable prototype files are hashed in `results/thermoelastic-rod-r4-source-provenance.json`. Existing r3 and older outcomes are preserved; canonical files were reverified unchanged. Root owns any subsequent freeze, Docker controls and three-trial evaluation.
