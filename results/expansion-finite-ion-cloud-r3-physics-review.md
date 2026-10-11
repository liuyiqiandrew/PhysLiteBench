# Finite ion cloud, revision 3: source and physics review

**Approve for the frozen evaluation batch. No blocking issue.** Reviewed all 17 task files, both scripts, the completed validation report, and both fresh Docker verifier outputs. This is a scientific source approval, not evidence of agent difficulty. No task files were changed or model trials run by this reviewer.

The public specification fully determines the finite canonical N=1,2,3 logarithmic gas: fixed neutralizing disk, both hard walls, genuine two-dimensional pair interactions, homogeneous dielectric without images, temperature convention, common unknown coupling, and finite-radius pressure derivative. The accessible-area normalization and fixed quantities in that derivative are correct. Absolute-radius independence at fixed ratio follows from scale invariance. Neither a dilution limit nor an additional three-body interaction is implied.

The three-ion oracle correctly uses the linked relative-angle convolution. The real Fourier coefficients have the right zero-mode, doubled positive-mode, and single Nyquist weights for its even angular grid. The common-radius exponent is 6-6*Gamma, the two gap exponents are 4-3*Gamma and 2-Gamma, and pressure is `(1-exp(-2L))*Z'/(6Z)`. The triangular radial Jacobian and the continuous common-radius limit at Gamma=1 are correct. The one-ion and two-ion responses are the previously checked exact laws, with stable removable limits.

The completed control consistently implements the conditional-angle Bethe partition approximation: it retains each exact pair factor and integrates the correlated radii, but omits nonzero loop harmonics. It is positive and exact for all one-ion/two-ion calibration preparations. AUTHOR correctly distinguishes this closure from a realizable three-position Gibbs trial distribution or a variational upper bound. Retaining this closure despite the shared particle angles is the intended physical error. The unfinished public fit is not itself such evidence.

The independent reference evaluates positive microscopic weights over two linked angles, both gaps and the common radius. Its second angle is the third particle's orientation; the second pair uses the angle difference, correctly enforcing the loop. It uses neither Fourier convolution nor analytic common-radius reduction for three ions. Refinement from 40/160 to 64/256 radial/angular nodes changes hidden predictions by at most 8.12e-10. The 96-case domain disagreement is 1.371e-5; the largest reference refinement is 1.597e-5, oracle refinement 7.169e-6, and refined disagreement 7.113e-6, all at weak-coupling preparations with order-one pressure. These reported discretization errors are far below the 4% grading tolerance. The eighteen Gamma=2 determinant checks and the ideal-gas limit validate normalization independently.

I added a separate Gamma=4 calculation: expand the squared three-variable Vandermonde polynomial and integrate its squared modulus using elementary original-radius moments. Its 19 nonzero monomials give the equal-radius angular norm 90. At six radius ratios from 1.25 to 12, the resulting pressure agrees with the packaged oracle within 5.00e-16. Four explicit discrete linked-angle sums, including weak coupling and unequal gaps, independently confirm the Fourier contraction and Nyquist weighting within 5.33e-15. No logarithmic-gap reduction or Fourier formula is used in the Gamma=4 check.

Calibration contains 240 records at 48 one-ion/two-ion preparations, sigma 0.001, seed 622001; public and private copies are byte-identical. The one-ion subset strictly identifies the shared coupling. Both completed fits give g=1.11993343604, reduced chi-square 1.0687472081. Forty-one noiseless fits recover the full parameter interval within 3.83e-8. The oracle's largest diagnostic group error is 0.03342%; closure errors are 10.9818%, 19.0037%, and 29.5393%. Across 256 noise draws, the worst oracle error is 0.44612% and the weakest diagnostic closure error is 10.7926%. Stronger one-ion/two-ion anchors pass both controls. Thus neither parameter uncertainty nor numerical error explains the hidden separation.

The image copies only public files and calibration. AUTHOR, hint, reference, solution, true coupling and hidden preparations remain private. Frozen truth is checked against fresh and refined references by the validator; private tests grade output shape, calibration, parameter recovery, and group-relative RMS predictions. The completed controls are comfortably within the unchanged time limits.

Fresh Docker evidence inspected:

- `finite-ion-cloud-expansion-r3-oracle-20261011-1027/finite-ion-cloud__LNu9wMX`: 7/7 passed in 1.01 s, reward 1, zero infrastructure exceptions.
- `finite-ion-cloud-expansion-r3-shortcut-20261011-1027/finite-ion-cloud__C4W4Kth`: four passed; only `three_ion_loop`, `strong_loop`, and `wide_cell` failed in 0.79 s, reward 0, zero infrastructure exceptions. Calibration, coupling and shared anchors pass.

SHA-256 of reviewed sources and validation evidence (20 files; final AUTHOR included):

```text
c7f633f12e2218bc72ae33088a3c3204bf27d6b7a2435c6d31ae6d818ecfbf50  tasks/finite-ion-cloud/AUTHOR.md
375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70  tasks/finite-ion-cloud/environment/Dockerfile
089f3c7acc33ba824ef933656fafd52b3e011e3a5c48d525d47390cfa4d9c09d  tasks/finite-ion-cloud/environment/README.md
773d87c3087a9a467f82a3e75a619eec2ad35515b7a0d16f5d093ec24509fa12  tasks/finite-ion-cloud/environment/data/calibration.json
23b294599c3ed5516834c61252eac1735a9c82c20155e45008f9b2425b8a6f91  tasks/finite-ion-cloud/environment/model.py
9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9  tasks/finite-ion-cloud/environment/test_public.py
7ed1473866c5b1f4b9d6fda075ad19b5903fefd13229ecb02a40214bd859cbaa  tasks/finite-ion-cloud/hint.md
d45ee4a47bc9a6eac698f1025bbad6e995b7d73b56e5e3bbf7055599642aa94d  tasks/finite-ion-cloud/instruction.md
671eb6a8ec11c8831ab1b14dc29994843932bfcc32dcc21a5ce48921bcb906e3  tasks/finite-ion-cloud/solution/model.py
bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644  tasks/finite-ion-cloud/solution/solve.sh
cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af  tasks/finite-ion-cloud/task.toml
773d87c3087a9a467f82a3e75a619eec2ad35515b7a0d16f5d093ec24509fa12  tasks/finite-ion-cloud/tests/data/calibration.json
287783c6a487c5bb0e02c7596fc25992f5f7be29bd9cacf4c6e28d8cf53281a4  tasks/finite-ion-cloud/tests/data/hidden_truth.json
5add34437cfa5d8a8d97fd166dab00229aaaf77b7d80ff9345c0abf2a86b7494  tasks/finite-ion-cloud/tests/metadata.json
6f0e2515a4cd118533f6b224307f98019282f591bae0811991f6b7ac0f074d55  tasks/finite-ion-cloud/tests/reference.py
74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c  tasks/finite-ion-cloud/tests/test.sh
663c0feb7fe5898f6ffbcdc6ffe12b85b795fc72b728ca970c7f9fd1a03b949d  tasks/finite-ion-cloud/tests/test_hidden.py
0d6ee4717c2ee27044a2e61434e521cf37d57215e4a6a81dba55bc98e7547eb7  scripts/finite_ion_cloud_baseline.py
a6cefd58fdc3014d3a6db0dec94d7fffcea220e0ad0eb29cd956811d25ddaabb  scripts/validate_finite_ion_cloud.py
e2a7020251e62aec782b95cba037b1c01fa458884463a06d2388d4a2ff5c91ab  results/expansion-finite-ion-cloud-r3-validation.json
```
