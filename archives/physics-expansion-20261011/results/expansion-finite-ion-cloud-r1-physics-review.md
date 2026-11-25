# Finite-ion cloud r1: source and physics review

**Approved for frozen evaluation. No blocking issue found.** Reviewed the 17 task files, completed control, validator and saved validation JSON hashed below. This reviewer changed no task files, data or grading and ran no model evaluations.

The public Hamiltonian unambiguously specifies a finite canonical two-dimensional logarithmic gas, with fixed N, neutral total charge, hard walls and no images. It fixes the area derivative, temperature/coupling convention, finite-N order of limits and accessible-area normalization. The oracle correctly uses the leading sector exponents `A_k=2*k-Gamma*k*(k+1)/2` and returns `max(A_k)/(2*N)`. The finite maximum handles ties without a bound-count convention. It gives the exact one-ion result, continuous pressure at all transitions and zero pressure for Gamma>=2. Disk-radius independence follows only after the stated limit.

The completed control correctly implements the independently localized Hartree trial distribution, including the binomial variance in `E[K(K+1)]`. It preserves finite-N pair counting, the exact N=1 response, the all-free branch and first condensation threshold. Its missing anticorrelations are a physical approximation. It is not the earlier rejected continuous relaxation of an integer sector maximum; AUTHOR makes that distinction accurately.

Calibration has Gamma<=.42 throughout the allowed parameter interval, below the earliest threshold .5. Both models therefore share the exact response `1-Gamma*(N+1)/4` and one uniquely identified weighted linear fit. Public/private calibration copies match: 240 records, 24 settings, ten independent replicates, sigma .001 and seed 620001. Both controls fit g=1.119813059546279 with reduced chi-square 1.1299595931. The shortcut's three discriminating errors are 30.9521%, 68.9039% and 85.8012%; the maximum oracle error is .131811%. Across 256 noise draws, the weakest shortcut error remains 30.8203% and the largest oracle error is .641882%, safely separated by the 4% gate. The discriminating truth is positive, including a minimum .012 in the final group; fully condensed one-ion anchors are shared.

The reference evaluates the ordered logarithmic-radius partition integral with a triangular matrix exponential and extrapolates its logarithmic derivative. Its radial slopes, cumulative-sum generator, exponential shift and derivative are consistent with the ordered integral. Angular distance bounds justify equality of the leading exponent with the full Gibbs partition function. The saved factor-two refinement, 160 domain checks, exact/near-transition checks and elementary one-ion integral support the frozen limiting truth. Near ties the recorded maximum extrapolation error is `4.607e-7`, decreasing to `3.871e-8` on refinement. Forty-one noiseless parameter recoveries agree within `2.23e-16`.

The full-angular Sobol check retains the pair-distance factors and uses the correct uniform/inner/outer mixture importance density. Its omitted angular-volume and factorial constants cancel between log radii. Four independent scrambles at six settings provide a physical Gibbs check, not grading truth. The finite logarithmic partition ratio at a tie correctly differs from the infinite-radius pressure. The independent angular factors need not produce exactly the same finite-radius ratio as the ordered-radial model; the validation allows their finite-size difference as well as sampling error.

Additional reviewer checks: sixteen direct nested-radial ODE integrations agree with the matrix-exponential log partition within `5.479e-12`. Exact N=2 angular integration using the averaged pair factor `hyp2f1(-Gamma/2,-Gamma/2;1;exp(-2*gap))` at six couplings and log radii 40, 80 and 160 independently recovers all-free, partially bound, fully bound and tie limits. At Gamma=1.8 the pressure derivatives approach .05 as `.05001745`, `.050000006`, `.0500000000000004`; at Gamma=2 the residual is exactly `1/(4*log(R/a))`, tending to zero.

One harmless convention: `radial()` returns `(2*N)^-1*d_L log Z`. The actual finite-annulus coefficient also contains `1-exp(-2*L)`. Accordingly, the finite one-ion test verifies the logarithmic derivative, rather than that finite-annulus coefficient. This has no effect on the public dilution limit, the large-L extrapolation or frozen truth.

Read metrics, pytest output and job summaries from `finite-ion-cloud-expansion-r1-{oracle,shortcut}-20261011-1215`: oracle **7 passed** in 1.07 s; shortcut **4 passed, 3 prediction failures** in .93 s, exclusively `few_free_ions`, `strong_binding` and `last_free_ion`. Both jobs report zero exceptions. The Dockerfile copies only public files; author derivation, hint, solution and private reference/truth stay outside the ordinary image. Scientific controls are established; this review does not claim agent difficulty.

Reviewed source SHA256 values:

```text
790be11be4d5ec10d5dee00fd5c794a2dcd66b91258d0ec490c45ec1efb5b2ba  tasks/finite-ion-cloud/AUTHOR.md
375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70  tasks/finite-ion-cloud/environment/Dockerfile
49212c270009af5fccfbcb4252e431a58961cc1314b6185aeef23204d9b3580c  tasks/finite-ion-cloud/environment/README.md
e1378c1d80f3e7d551bbda080bab347cca88f036e35487dac12cf1518a26f3ca  tasks/finite-ion-cloud/environment/data/calibration.json
c79c487763edc3f7c7432ccef213e0088bdbd45bffb7c32df92a77cbffa44e13  tasks/finite-ion-cloud/environment/model.py
9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9  tasks/finite-ion-cloud/environment/test_public.py
654db335a5992e7ec4c10ac79700a696596b94a12da675554ae1a9b3b85abdad  tasks/finite-ion-cloud/hint.md
68f5e266ffeab93b7c926bbe35519f04c01295ffdf737f21da75adf858b8960b  tasks/finite-ion-cloud/instruction.md
72a81a938da2b45bcb009c241652d10de0112787cbe0442c4e47dd258e63669c  tasks/finite-ion-cloud/solution/model.py
bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644  tasks/finite-ion-cloud/solution/solve.sh
cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af  tasks/finite-ion-cloud/task.toml
e1378c1d80f3e7d551bbda080bab347cca88f036e35487dac12cf1518a26f3ca  tasks/finite-ion-cloud/tests/data/calibration.json
a7d8f72148074f04f0be752f6e336b3e13858c5f9752447ce9d3865c760c8ca9  tasks/finite-ion-cloud/tests/data/hidden_truth.json
b9a20f300e1f93cb9193564b3af8439692e8525b3f207e92d4ba1644b7e7b661  tasks/finite-ion-cloud/tests/metadata.json
c7a910b4f99799fb38644840021209582faf568fe3f06dee9928be65352e46e9  tasks/finite-ion-cloud/tests/reference.py
74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c  tasks/finite-ion-cloud/tests/test.sh
663c0feb7fe5898f6ffbcdc6ffe12b85b795fc72b728ca970c7f9fd1a03b949d  tasks/finite-ion-cloud/tests/test_hidden.py
3b760f67e4fee051bf2cdc3f038e383325b924e85272944362f5cfc150ae59dd  scripts/finite_ion_cloud_baseline.py
7ab72bdbce67ff1b392d43bbcafa6717d02755c0b4c8fd75d1467688846143e1  scripts/validate_finite_ion_cloud.py
c41f1b9fe905c5aa5e0019763cea821eb70964221697bdea40d503c61f58d5a3  results/expansion-finite-ion-cloud-r1-validation.json
```
