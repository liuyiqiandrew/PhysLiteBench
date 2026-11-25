# Counterion cloud r1: source and physics review

**Approved for a frozen evaluation. No blocking physics, specification, numerical or separation issue found.** Reviewed the 17 task files, completed shortcut, validator and saved validation report hashed below. No task files or data were changed; no model evaluations were run by this reviewer.

The public apparatus fixes the positive rod, negative monovalent ions, canonical neutral finite cell, homogeneous dielectric, PB approximation, accessible-volume normalization and order of limits. These determine the response without a missing boundary condition. `rod_radius` correctly drops out only after the specified dilution limit. The Dockerfile copies only public files; author diagnosis, hint, solution, reference and private truth remain outside the ordinary image.

Writing `xi=lB*u*tau0`, the oracle implements `1-xi/2` below 1 and `1/(2xi)` above 1, with matching value and slope at the threshold. The completed control correctly implements its all-ions-dilute-away pressure closure: it omits the inner contact contribution in `phi_R=(1-a²/R²)[1-xi/2+pi*a²*n(a)/(u*tau0)]`. AUTHOR correctly distinguishes this closure from an admissible supercritical PB profile. Its predictions remain positive throughout the public domain.

Calibration has `xi<=.7` for every permitted fitted parameter. Both models therefore share the exact response and the same uniquely identified weighted linear fit. The 240 records, independent Gaussian replicates, public/private calibration equality, 41 noiseless recoveries and 256 noise draws are consistent. The fitted density is 1.2994863872479447 nm^-1, with reduced chi-square 1.0294801131. Frozen hidden couplings are 1.36864–1.625 and have ample separation: shortcut relative RMS errors are 17.1535%, 25.1009% and 33.3508%, versus oracle 0.039525%. Across noise draws the weakest shortcut error remains 17.0451% and the largest oracle error is 0.094390%, well inside the 4% gate.

The private reference independently solves the finite-cell first integral, including its hyperbolic/trigonometric branch crossover at `xi=L/(L+1)`, and extrapolates four finite-cell pressures. It does not substitute either limiting oracle branch. The BVP check separately imposes the two electrostatic boundary conditions and checks neutrality. The saved factor-two refinement, 80 domain cases and threshold checks support the frozen truth. I additionally solved the original PB potential equation with unknown wall density at nine settings, `xi=.084,.8,1,1.4,1.75` and log radii 3 or 12. Maximum pressure disagreement was `2.765e-13`; contact-identity and neutrality residuals were below `2.975e-13`.

My denser 2,195-point scan around `xi=1` found a maximum L800 extrapolation error of `1.2932e-8` at `xi=1.00045`, falling to `2.5275e-9` with L1600. Thus the report's tested sub-`1e-8` errors are not a uniform bound over all threshold inputs. This causes no grading concern: the reference retains much better than the validator's `1e-6` numerical allowance and the 4% prediction gate; hidden groups are away from this transition region.

Read the verifier metrics and pytest output from `counterion-cloud-expansion-r1-{oracle,shortcut}-20261011-1020`: oracle **7 passed** in 0.74 s; shortcut **4 passed, 3 failed** in 0.76 s, exclusively `strong_charge`, `long_bjerrum_length` and `largest_coupling`. Both fit and recover the parameter correctly. No infrastructure exception appears in those outputs. Scientific controls are established; agent difficulty and causal classification remain to be measured.

Reviewed source SHA256 values:

```text
b6eeae6ef40b6114996e1b8c3e45594cc9d1c1927b99d9b8c8fcab0afd2c443c  tasks/counterion-cloud/AUTHOR.md
375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70  tasks/counterion-cloud/environment/Dockerfile
7d6c014afb5e840c17e84f4826475a2e9bf30ea6c831467ccb30e515e7f0b511  tasks/counterion-cloud/environment/README.md
dddb166055207fe74ef751d01ddb749f24130a500ecc2aa7219b8dd5a271933d  tasks/counterion-cloud/environment/data/calibration.json
aed1937f7114b106684328bbef8571f903b41fa8508e37333b16a8c1fb7a62c8  tasks/counterion-cloud/environment/model.py
9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9  tasks/counterion-cloud/environment/test_public.py
305dee952fc80dc0cfdacdf4a2491142cd6287b759c51ffd2a910962b44263ca  tasks/counterion-cloud/hint.md
d82a3bf816cec6fec0e1ca4bf1d7ba35ca090149583ef4f2932c0f357c2bd587  tasks/counterion-cloud/instruction.md
c9604ac39bd5309f7af90e36a8921802eb690d23c435757d0c18878de76c74f0  tasks/counterion-cloud/solution/model.py
bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644  tasks/counterion-cloud/solution/solve.sh
cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af  tasks/counterion-cloud/task.toml
dddb166055207fe74ef751d01ddb749f24130a500ecc2aa7219b8dd5a271933d  tasks/counterion-cloud/tests/data/calibration.json
bee58a7261abb9fbf6bee8d16a9e673387c81608b32c810a45cff553b98f7bbf  tasks/counterion-cloud/tests/data/hidden_truth.json
b9a20f300e1f93cb9193564b3af8439692e8525b3f207e92d4ba1644b7e7b661  tasks/counterion-cloud/tests/metadata.json
1154f10591734fed2b60fded5a19386e1764216fbfeb9a0107343a075c0e6a5f  tasks/counterion-cloud/tests/reference.py
74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c  tasks/counterion-cloud/tests/test.sh
0a814cab3866bf0ec8545df05fa2eeb362f751fb2d29a6d9303530de7bf4a4cf  tasks/counterion-cloud/tests/test_hidden.py
f50cb0707ff6c3107f50452961ce9e5de49367907e3693287c2f6e4c0bfb2810  scripts/counterion_cloud_baseline.py
08faf274104f26f2b8f33f48fcf00a66211eaaa868435ac831bfba8b4f7f05ef  scripts/validate_counterion_cloud.py
203adf7ac652c5392abf5e86952758bef898ad46f09ed16fa4916af490963ee9  results/expansion-counterion-cloud-r1-validation.json
```
