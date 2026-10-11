# Counterion cloud r2: source and physics review

**Approved for frozen evaluation. No blocking issue found.** Reviewed the 17 task files, completed control, validator and final validation JSON hashed below. No task files, data or grading were changed, and this reviewer ran no model evaluations.

The public apparatus specifies both valences, separately conserved species counts, the charge-fraction convention, common potential, canonical PB equilibrium, dielectric assumptions, pressure normalization and finite-cell-then-dilution limit. These determine the response. Pure-species endpoints are explicit. Radius independence is a consequence of the stated limit. The ordinary Docker image contains only public apparatus, starter, calibration and interface tests; the diagnosis and private answers remain separate.

For `xi=lB*u*tau0` and `g=1-f+f/3`, the oracle correctly minimizes the leading free-energy coefficient `(v1+3*v3)^2-2*(v1+v3)` under the two population bounds. Its four branches select trivalent condensation first, then exhausted trivalents with free monovalents, then partially condensed monovalents. The pressure `(vfree-xeff²/2)/(xi*g)` includes outer-cloud electrostatics. The formula recovers both pure-valence limits and joins continuously with continuous first derivatives. AUTHOR correctly states that both valences cannot be partially condensed simultaneously.

The completed mean-valence control preserves total charge and total ion number, applies the complete single-species condensation law, and is exact at both pure endpoints. It is a valid implementation of its assumed closure, rather than an unfinished solver or incorrect pure-species formula. This is a substantive selective-condensation revision of r1.

All calibration has `3*xi<=.7875` throughout the permitted parameter interval. Both models therefore share `1-xi/(2*g)` and the same unique weighted linear fit. Calibration copies match; 240 records represent 48 settings with five independent replicates, seed 619001. The fitted density is 1.2995900587947886 nm^-1, with reduced chi-square .9732847344. The three discriminating group errors are 48.7299%, 41.2099% and 24.8652% for the control; the maximum fitted oracle error is .031544%. Across 256 noise draws, the weakest control error remains 24.7923% and the largest oracle error is .149421%, comfortably separated from the 4% gate.

The reference uses bounded convex minimization without the oracle's explicit branches. AUTHOR accurately calls this an independent implementation of the limiting law, and identifies the species-conserving finite-cell PB BVP as the independent physical check. The saved 288 domain cases, 48 transition cases and 41 noiseless recoveries agree within `4.45e-16`. Nine BVP cases at log radii 80, 160 and 320 cover all four regimes; the largest final finite-size error is `3.703e-5`. Separate species quadrature and the finite-cell contact identity agree within `1.032e-10`. The finite-cell helper is used for positive interior mixtures; public pure-species predictions are handled by the limiting reference and oracle, with pure-species hidden anchors.

I additionally implemented the original PB potential equation with two unknown density normalizations and an independent trivalent-number integral. Six checks at `(xi,f)=(.9945,.85),(.9945,.4),(1.43,.2)` and log radii 3 and 12 matched the packaged finite-cell solver within `3.952e-11`; reconstructed species counts agreed within `1.666e-13`. At 63 coupling/composition settings, including exact and near absent-species endpoints, the convex optimizer's projected-gradient residual was below `2.23e-15`. These checks found no sign, normalization, constraint or branch issue.

Read metrics, pytest output and job summaries from `counterion-cloud-expansion-r2-{oracle,shortcut}-20261011-1100`: oracle **7 passed** in .67 s; control **4 passed, 3 prediction failures** in .68 s, exclusively `selective_condensation`, `trivalent_exhaustion` and `two_condensed_species`. Both jobs have zero exceptions. The final validator uses the accurate field name `oracle_reference_max_error`; it does not mislabel that comparison as refinement. Scientific controls are established; agent difficulty remains unevaluated by this review.

Reviewed source SHA256 values:

```text
45ea4e3547ab48f3fccdc39dcb53cc7b7330e627fae80c44c41d2284592dc960  tasks/counterion-cloud/AUTHOR.md
375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70  tasks/counterion-cloud/environment/Dockerfile
05e85df58bb8d0d993c835517a9641b1773bd8767e78fc3bba09b4897ddf1c8d  tasks/counterion-cloud/environment/README.md
90a0d7ab927f0b893d8f9178fb35eba8d41bb83d30d3636afa96f8a75806aeb2  tasks/counterion-cloud/environment/data/calibration.json
79b5c32dde02f96acfcdf42d598d9febec14577e5d0e1328a3cbfed4ad81b403  tasks/counterion-cloud/environment/model.py
9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9  tasks/counterion-cloud/environment/test_public.py
46cb7109de5d091dec78262368bf42906b90a71fbe61c3989bf03850d6ad6281  tasks/counterion-cloud/hint.md
d82a3bf816cec6fec0e1ca4bf1d7ba35ca090149583ef4f2932c0f357c2bd587  tasks/counterion-cloud/instruction.md
d8997b052b2c64a82ba30e75c93a9d0eed07c8f4a16e77f8b105b1086b0f4ac9  tasks/counterion-cloud/solution/model.py
bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644  tasks/counterion-cloud/solution/solve.sh
cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af  tasks/counterion-cloud/task.toml
90a0d7ab927f0b893d8f9178fb35eba8d41bb83d30d3636afa96f8a75806aeb2  tasks/counterion-cloud/tests/data/calibration.json
f79a764b3f9063a230f4db5d6ef14655f3faf7791eac5d04f90db8c012c698e4  tasks/counterion-cloud/tests/data/hidden_truth.json
e7e73504277a03622823d3a03a52315f7383d161d0b94df65dcbb2a1f0c00a89  tasks/counterion-cloud/tests/metadata.json
72b28d1d7b6d05ecdf8eac0bc57c9593d87bf07a4a5e5b0e365de453c19fc750  tasks/counterion-cloud/tests/reference.py
74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c  tasks/counterion-cloud/tests/test.sh
0a814cab3866bf0ec8545df05fa2eeb362f751fb2d29a6d9303530de7bf4a4cf  tasks/counterion-cloud/tests/test_hidden.py
4e374aac5c4b33a9880306c8b8075953ef165d238834cdc52b0fd512753ce38f  scripts/counterion_cloud_baseline.py
c810ec007053b80753389e4cb56367345f64373c4ae4af202b56ed323b7bd8a8  scripts/validate_counterion_cloud.py
0a3a2386b90cc098e4d2b328d5d3af461a971543b58e39488692285caee9f629  results/expansion-counterion-cloud-r2-validation.json
```
