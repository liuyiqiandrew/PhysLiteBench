# Finite-bath gravitating gas r1: independent source review

Reviewed 2026-10-11. **Approve the scientific package for a frozen evaluation batch. No blocking issue found.** I changed no task or validation files, ran no model trials, and delegated no work. This approval does not establish model difficulty.

## Physics and public scope

The README fully specifies the collisional Boltzmann mean-field description, regular spherical profiles, rigid pressure-supporting wall, reservoir energy and heat capacity, and local entropy criterion. It distinguishes isolated composites, zero capacity, and the imposed-temperature limit. It excludes finite-N collapse times, global collapse, and Vlasov stability. Contrast labels a preparation rather than constraining perturbations. The first instability is selected by continuation from dilute profiles; the public instruction correctly distinguishes it from the branch's temperature minimum. Units, domain, API, and unknown G are complete.

The solution correctly uses theta=GM/(Rv), gas energy (GM²/R)(u−3/2)/v including wall pressure, and composite energy with the additional bM theta. The finite-bath condition `4u²+2uv+(2b−11)u+3−2b=0` and the canonical condition `u=1` select the first losses throughout the allowed domain. Ordinary-profile root brackets cover contrasts 1.5–25. The completed control correctly implements the infinite-reservoir approximation for every capacity; it retains exact hydrostatic profiles and common calibration.

## Independent reference and extra checks

The reference integrates equilibrium in logarithmic radius and determines stability from the entropy Hessian, without using the turning-point polynomial. At G=M=R=1 the implemented matrix is minus twice

\[
Q=-\frac12\int\frac{q'^2}{4\pi\rho r^2}dr
+\frac1{2\theta}\int\frac{q^2}{r^2}dr
-\frac{[\int M(r)q/r^2\,dr]^2}{(3+2b)\theta^2}.
\]

Consequently, positive curvature eigenvalues mean stability. The rank-one coefficient, sign, and infinite-bath limit are correct. The positive entropy-gradient matrix is a valid generalized metric. The basis `q=r³(1−r²)P_j(2r²−1)` enforces regular central density and zero total mass perturbation without imposing an extra wall-density constraint. Density, enclosed mass, derivatives, and quadrature factors are correct. Checks before the threshold and across its first crossing support selection of the first unstable mode.

My additional checks were:

- **160 modes/512 quadrature points**, at b=0, 0.7, 1.5, and infinite: maximum critical-temperature-coefficient change from 96/256 was **7.183e-12**; maximum difference from the oracle was **1.164e-10**.
- **Direct fixed-energy entropy variations**, computing gravitational self-energy from the spherical shell integral rather than the turning-point formula: three regular perturbation checks agreed with the Hessian quadratic coefficients within **2.858e-9 absolute**.
- **Direct mass and wall-energy integrals**: discrepancies below **4.862e-10**.

These checks support the reference's physical independence and mode convergence. Small differences between the two equilibrium integrations are negligible for grading.

## Calibration, grading, and controls

Public/private calibration copies are byte-identical: 240 independent readings at 30 settings, sigma=0.001. Every mean is a positive known coefficient times G, so G is globally identified. Both controls have exactly the same calibration response for every G. I independently reproduced the fit **G=1.079964207650113**, reduced chi-square **0.8705600421**, every frozen prediction within 1e-12, and all reported group errors.

| Group | Oracle error | Shortcut error |
|---|---:|---:|
| Shared anchors | 0.0033141% | 0.0033141% |
| Small reservoir | 0.0033141% | 18.2099% |
| Medium reservoir | 0.0033141% | 13.0782% |
| Larger reservoir | 0.0033141% | 7.9855% |

The saved report matches the validator: 256 noise draws, 41 noiseless fits, and 60 domain cases. Worst oracle noise error is 0.02962%; weakest discriminating shortcut error is 7.9557%, above the 4% gate. Maximum domain oracle/reference discrepancy is 3.452e-10; hidden reference refinement changes values by at most 1.183e-11. Hidden capacities stop at 1.5; the weakly separated b=2 endpoint remains in domain validation. Prediction normalization is consistent, outputs are nonzero, and all cases are public-domain preparations.

I inspected verifier metrics and pytest output from `finite-bath-gravitating-gas-expansion-r1-{oracle,shortcut}-20261011-0940`. The oracle passed **7/7** in 0.79 s. The shortcut passed interface, calibration, parameter, and shared-anchor checks, failing only the **three finite-bath prediction groups** in 0.70 s. Docker metrics reproduce local results.

Only public environment files enter the image. The private truth, parameter, Hessian reference, author notes, and optional hint remain separate. The installer changes only the model; numeric rewards and diagnostic metrics are recorded. No leak or undisclosed physical assumption was found.

Later qualification requires trial-source and trajectory review. Retaining fixed-temperature stability is the intended physical error; algebra, wall-energy, root-selection, and numerical mistakes need separate classification.

## Reviewed SHA-256 hashes

All 17 task files, the completed control, validator, and saved validation report were reviewed. Generated Python caches are excluded.

| File | SHA-256 |
|---|---|
| `tasks/finite-bath-gravitating-gas/AUTHOR.md` | `4be99db41e5eacdca4d23687cd67173132a9beb2f9fac115741c6d446af23551` |
| `tasks/finite-bath-gravitating-gas/environment/Dockerfile` | `375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70` |
| `tasks/finite-bath-gravitating-gas/environment/README.md` | `ab1ad7eb993e56f7cb65bbfd1b1b147978eb4bf2b7634c0f785d6275658a5b50` |
| `tasks/finite-bath-gravitating-gas/environment/data/calibration.json` | `292ba58970bd96f21800a43cc7b600f4c1310ff47e1f58f2c9962cc02c759a27` |
| `tasks/finite-bath-gravitating-gas/environment/model.py` | `4c7f161e38577e5b55d3075bffbe54c33e9480e2ac561b852c0717d81b5f3328` |
| `tasks/finite-bath-gravitating-gas/environment/test_public.py` | `9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9` |
| `tasks/finite-bath-gravitating-gas/hint.md` | `e35fcb144b95b8e49d8c9352b33389fe48442210be796df7eab9e15f4dc1724b` |
| `tasks/finite-bath-gravitating-gas/instruction.md` | `2a7df2934c89b1925e193d07402bebf6e074967a8e91afaa747552ea22e4c5db` |
| `tasks/finite-bath-gravitating-gas/solution/model.py` | `c21bd142983637b80a2f464fc8396ae793dc674a8abddba915f66515a59fd85f` |
| `tasks/finite-bath-gravitating-gas/solution/solve.sh` | `bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644` |
| `tasks/finite-bath-gravitating-gas/task.toml` | `cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af` |
| `tasks/finite-bath-gravitating-gas/tests/data/calibration.json` | `292ba58970bd96f21800a43cc7b600f4c1310ff47e1f58f2c9962cc02c759a27` |
| `tasks/finite-bath-gravitating-gas/tests/data/hidden_truth.json` | `1f127bae8d0e47a5af6ad9e8f7d2acfc48a79a61f415c039eab3c8614424a142` |
| `tasks/finite-bath-gravitating-gas/tests/metadata.json` | `b9a20f300e1f93cb9193564b3af8439692e8525b3f207e92d4ba1644b7e7b661` |
| `tasks/finite-bath-gravitating-gas/tests/reference.py` | `66cda774e946b5b8284a7ba8fcb8afc7d0ad57939edae9c5005fcd9a9a08ba12` |
| `tasks/finite-bath-gravitating-gas/tests/test.sh` | `74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c` |
| `tasks/finite-bath-gravitating-gas/tests/test_hidden.py` | `5e3541bd053695b4228c19fe85c836e79fa998cba4e549851f22e33b954c821f` |
| `scripts/finite_bath_gravitating_gas_baseline.py` | `75ebbfd652546c630440eecd8cef1e6d448e7ccb4e248f31a2dbc3910ee23345` |
| `scripts/validate_finite_bath_gravitating_gas.py` | `34a5a4c0f530265ca6d90ea25c05308a43152efec87896bba3d57f20980e7588` |
| `results/expansion-finite-bath-gravitating-gas-r1-validation.json` | `7d4f4415efc1e318e48ff5d5bf0722cff13682d4222e9e9a8b9d131875fe99ed` |
