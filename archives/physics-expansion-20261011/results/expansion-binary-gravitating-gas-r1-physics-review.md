# Binary gravitating gas r1: independent source review

Reviewed 2026-10-11. **Approve the scientific package for a frozen evaluation batch. No blocking issue found.** No task files were edited, no model trials were run, and no work was delegated. This review does not establish model difficulty.

## Physics and public scope

The README fixes heavy **mass fraction**, the two particle masses, separate conserved species numbers, common imposed temperature, and the light-mass convention for theta. Both species can move freely in their shared gravitational potential. The rigid wall, lack of additional forces, regular dilute branch, local spherical free-energy criterion, and exclusions of finite-particle collapse and collisionless stability are explicit. The public apparatus determines the intended equilibrium and stability problem without a hidden composition constraint.

Both equilibrium integrations correctly solve the two Boltzmann density laws. At each total-central-density radius alpha, they solve for central heavy fraction to enforce the **global** heavy fraction. The species mass integrals have the correct radial Jacobians; eta=alpha psi′ gives theta=GM/(R eta). The oracle scans from the dilute branch before maximizing eta locally, rather than selecting a later peak. The reference independently scans the smallest free-energy-curvature eigenvalue for its first crossing. Its integration is in logarithmic radius and does not use the oracle's turning-point condition.

The Hessian has separate positive entropy-gradient blocks, including the heavy-species factor 1/chi, and a shared gravitational block acting on the sum of the two mass perturbations. Its normalization equals 2 delta²F/theta, so positive eigenvalues mean stability. Each basis `r³(1−r²)P_j(2r²−1)` enforces its own fixed species mass and regular center, without an extra wall-density constraint. The absent heavy species is removed rather than divided by zero.

The mean-particle-mass control is a valid monodisperse auxiliary gas with the same total mass and total particle count. Its mean mass is correctly `m/(1−q+q/chi)`. It is exact in both calibration families, but misses spatial mass segregation in mixtures. The separate gravity-family lineage is documented; this is different from the prior finite-reservoir constraint.

## Independent checks

I explicitly tested the equal-mass case with **both** Hessian blocks retained. For q=0.1 and 0.5, at alpha=5, the canonical threshold, and alpha=12, the full generalized spectrum equals the single-species spectrum plus relative-composition eigenvalues **1**, within **4.73e-13**. Opposite species mass perturbations have zero total gravitational perturbation and curvature 1. They remain stable while the ordinary total-density mode crosses. The reference's one-species reduction at chi=1 is therefore justified, not an omitted instability.

Additional near-limit tests covered `(chi,q)=(1.0001,.3)` and `(5,q)` for q=1e-6, .003, .006, and .01. The first-crossing Hessian and oracle temperature coefficients agreed within **1.14e-11**. Their first critical alpha values ranged from 8.993 to 11.021, safely inside both continuation intervals. These checks supplement the validator's equal/absent species limits and trace-heavy cases.

The saved validator checks 64-to-96-mode refinement with 256-to-384-point quadrature, signs on both sides of the first loss, positive earlier curvatures, separate mass normalization, central heavy enrichment, and 28 random/boundary domain cases. Hidden temperature refinement changes values by at most **1.333e-15**; maximum oracle/reference domain discrepancy is **1.068e-10 absolute**. The seven reported mass-normalization errors are below **3.63e-12**. The first-mode tests and independent formulations support the selected physical branch.

## Calibration and grading

Public/private calibration copies are byte-identical: 240 readings at 30 settings, sigma=0.001. Every calibration mean is a positive known coefficient times G, identifying it globally. Both controls have the same complete calibration law. My independent fit reproduced **G=1.0698048317893067**. The recorded reduced chi-square is **1.0299328887**.

| Group | Oracle error | Shortcut error |
|---|---:|---:|
| Shared anchors | 0.0182400% | 0.0182400% |
| Moderate masses | 0.0182400% | 12.3966% |
| Unequal masses | 0.0182400% | 26.7225% |
| Heavy component | 0.0182400% | 48.3512% |

Across 256 noise draws, worst oracle error is 0.03750% and the weakest discriminating shortcut error is 12.3492%, well separated from the 4% gate. The 41 noiseless fits recover G to floating-point precision. Hidden cases obey the public domain, predictions are nonzero, and validator/verifier use the same relative-RMS normalization. The 3% parameter and 1.5 chi-square gates have ample margins.

I inspected metrics and pytest output from `binary-gravitating-gas-expansion-r1-{oracle,shortcut}-20261011-0910`. The oracle passed **7/7** in 8.80 s. The shortcut passed interface, calibration, parameter, and shared-anchor checks; only the **three mixture prediction groups** failed, in 0.69 s. Docker metrics agree with the saved scientific report.

Only public environment files enter the agent image. Private truth, parameter, reference, author notes, and optional hint remain separate. The installer changes only the model; the verifier records metrics and numeric reward. No public/private leak or undisclosed physical assumption was found. Later trial review must distinguish retained mean-mass physics from composition-normalization, branch, numerical, or implementation mistakes.

## Reviewed SHA-256 hashes

All 17 task files, completed control, validator, and saved validation report were reviewed; generated caches are excluded.

| File | SHA-256 |
|---|---|
| `tasks/binary-gravitating-gas/AUTHOR.md` | `3266dd62906c62f867dd0c2f8f99c98b9638222f8bc1dcf35fe3ef512d476a23` |
| `tasks/binary-gravitating-gas/environment/Dockerfile` | `375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70` |
| `tasks/binary-gravitating-gas/environment/README.md` | `85f27b782fa485dfe015c30773fa36b0b2ae7af12a0bed2ac19fb1aa7c59e868` |
| `tasks/binary-gravitating-gas/environment/data/calibration.json` | `b12cd47686548a78db2bf102c65d2fa985bcc334c3b6a62b20f6795ca6badc03` |
| `tasks/binary-gravitating-gas/environment/model.py` | `030d32db19db3dbdd0acbcd574f29c66993857dda1e421904222f2c7350b7c30` |
| `tasks/binary-gravitating-gas/environment/test_public.py` | `9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9` |
| `tasks/binary-gravitating-gas/hint.md` | `84d85ee409c2ace7dd63e1ab0aba839103fa4a457028c3df4d65024fb98e16ec` |
| `tasks/binary-gravitating-gas/instruction.md` | `075451b4d4a59eb61d3e91c423081b2feab4dd0f5b0e4644a9b37b396b3e3f06` |
| `tasks/binary-gravitating-gas/solution/model.py` | `9525099ff92ab1eb1dc36da58290f2be8b3e297b09cd374b0d61407a34389351` |
| `tasks/binary-gravitating-gas/solution/solve.sh` | `bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644` |
| `tasks/binary-gravitating-gas/task.toml` | `cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af` |
| `tasks/binary-gravitating-gas/tests/data/calibration.json` | `b12cd47686548a78db2bf102c65d2fa985bcc334c3b6a62b20f6795ca6badc03` |
| `tasks/binary-gravitating-gas/tests/data/hidden_truth.json` | `b5a281cc468b047d4d1b5f3fc351fd014117c58b8e1bfa2d93bed45e001b0ef0` |
| `tasks/binary-gravitating-gas/tests/metadata.json` | `b9a20f300e1f93cb9193564b3af8439692e8525b3f207e92d4ba1644b7e7b661` |
| `tasks/binary-gravitating-gas/tests/reference.py` | `3ce84c3803fdfb2a8230fa093476f3c8043495e5210a69de336f569da1e30caf` |
| `tasks/binary-gravitating-gas/tests/test.sh` | `74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c` |
| `tasks/binary-gravitating-gas/tests/test_hidden.py` | `5e3541bd053695b4228c19fe85c836e79fa998cba4e549851f22e33b954c821f` |
| `scripts/binary_gravitating_gas_baseline.py` | `5550a216695a5175ee651b28973da53c698694e69c0ff84b824868fcf1edf68a` |
| `scripts/validate_binary_gravitating_gas.py` | `7f5692c31bde2be2d1cf335541165bf1f5f157570ce998879def427297d00c82` |
| `results/expansion-binary-gravitating-gas-r1-validation.json` | `ac85f24d959f042ad27fb8176f104ebfefbdb80a37043d09ba5acffb22c69646` |
