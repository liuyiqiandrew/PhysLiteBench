# Swirling heat tails revision 1: independent physics review

Reviewed 2026-10-11. **Approved for frozen revision-1 evaluation. No blocking physics, source, reference, calibration, or tolerance issue found.** This review establishes scientific consistency, not target-model difficulty. The author separately reports the expected Docker controls: oracle 7/7 and shortcut 4 passes/3 diagnostic prediction failures. I made no task-source, data, or grading edits and ran no model evaluations.

## Apparatus and physical law

The public specification determines the experiment. It gives the overdamped diffusion, force orientation, fixed temperature, common unknown mobility, stationary initial preparation, unconditioned endpoints, Stratonovich heat into the bath, and the two limits defining a local large-deviation rate. It explicitly excludes a separate inertial kinetic-energy readout. The drive ratio is positive and bounded away from zero, so the normalization by each preparation's own mean heat is well defined. Stiffness and mobility change the physical time scale; drive ratio changes the dimensionless rate function.

The stationary position density is the Gaussian with covariance `I/stiffness`; the circulating probability current does not change that density. Dimensionless time is `mobility*stiffness*t`, and dimensionless mean heat production is `2*drive_ratio**2`. Consequently the implemented rate is `mobility*stiffness` times the dimensionless rate at the supplied relative heat. Using the preparation's own stationary mean for normalization is essential and is stated publicly.

For `r=drive_ratio` and the plus-sign counting field a, the bulk work generating rate is `g(a)=1-sqrt(1-4*r*r*a*(1+a))`. Heat differs from work by the unbounded potential-energy change. The stable tilted Gaussian solution and its overlap with the stationary initial Gaussian restrict the interior heat domain to `-1<a<a_plus`, with the oracle's piecewise upper boundary. Maximizing `2*r*r*z*a-g(a)` gives exactly the implemented clipped saddle. The return to physical time includes the correct stiffness factor.

For z below -1 the heat rate is `-2*mobility*stiffness*r*r*z`. At weak drive, `r*r<1/3`, the positive tail changes at `z=(3-r*r)/(1-3*r*r)`; the rate and its first derivative are continuous there. The oracle is stable at both switches, at the drive threshold, and at the mean where the rate vanishes. Tiny floating-point signs at exact zero are immaterial to the stated tolerances.

The use of a boundary saddle in this rate formula means the limiting value from the interior counting-field domain. It does not claim continuity of the actual exponential-moment growth at the exact domain endpoints. At a=-1, for example, the finite-time heat moment grows as `exp(2*mobility*stiffness*t)`, unlike the interior limit of g. The task asks for finite rate-function values, so this distinction creates no public ambiguity.

## Independent reference and numerical checks

The private reference constructs an independent time-discrete Gaussian path integral. Each OU transition has its exact mean rotation/contraction and exact covariance. The midpoint force integral preserves the potential-energy difference and approximates the rotational Stratonovich integral. Initial and final Gaussian precision blocks retain the two energy endpoints. The Schur-pivot positivity conditions determine the admissible counting-field interval; the reference does not use the oracle's heat-domain formula or its clipped saddle.

The largest positive pivot root gives the discrete bulk determinant growth. The admissible interval contains zero; Gaussian-moment convexity justifies the reference's two bisections. The chosen outer bracket is well outside the heat domain throughout the disclosed drive interval. Optimization compares the interior solution and both nearby boundary candidates. Three time steps remove the leading discretization terms, and the validator repeats the calculation at half those steps. The exact transition law alone would not make the polygonal heat exact; that refinement is necessary and is present.

The stable discriminant correctly has coefficient 8 in its linear counting-field term. My initial concept note incorrectly suggested 4. The author's calibration check caught that factor-of-two error before any task data were generated. I corrected the concept review and recorded the mistake there. Its earlier numerical checks used the original subtractive expression and were unaffected. Current task reference bytes contain the correct coefficient.

The frozen validator report has 146 domain cases, including the drive threshold, domain endpoints, z=-1, and z=1. Maximum oracle/reference absolute difference is `1.077e-7`. Hidden reference refinement errors are below `5.049e-9` in group relative norm; refined oracle/reference errors are below `7.234e-10`. Eight long finite-chain calculations extrapolate to the discrete bulk generating rates, and outside-domain finite-chain moments diverge as expected.

I also performed independent read-only checks in NumPy 2.5.2 and SciPy 1.18.0:

- Constructed full stationary OU path covariance matrices and the complete heat quadratic form directly, then evaluated the Gaussian moments from their eigenvalues. This avoids the source's scalar Schur-pivot algebra. The 106 finite cases agree within `3.553e-13` absolute, and all 14 divergent cases match.
- Compared 162 additional corner and crossover cases against reference steps beginning at `.0015`. Maximum absolute discrepancy is `3.230e-9`, including the largest mobility, both stiffness bounds, both sides of z=-1, weak-drive positive switches, and both sides of `r*r=1/3`.
- Checked positive-switch slopes numerically; maximum deviation from the continuous analytic slope is `2.353e-8`.

No warnings occurred. These checks supplement the author's pinned-image validation; they do not claim an exhaustive floating-point proof for every real input.

## Calibration, shortcut, and grading

There are 256 independent Gaussian readings at 32 settings, each with sigma `.0004`. Their z values lie in the common central region and avoid the zero rate at z=1. The completed shortcut computes the exact work rate function of the same process, so its agreement throughout calibration is physical and exact. Its error is using that rate for heat tails after discarding energy endpoints. It remains a nonzero completed approximation throughout the hidden tests.

Mobility is an identifiable linear scale: the unit-mobility calibration basis has norm `.8911323`. Both fitted controls give mobility `1.0799612406033` and reduced chi-square `.95008163`. Public and private calibration records match, and every frozen hidden input list matches the reference definition. The unfinished public fit is allowed to fail before implementation; the independently completed baseline supplies the required shortcut control.

Across 256 noise draws the fitted mobility range is `[1.078871748712,1.082034531173]`, and the maximum reduced chi-square is `1.367866`. Maximum oracle prediction error is `.001883831`, well below the 4% gate. The weakest diagnostic shortcut group remains above `.1184850` across those draws; the others exceed `.2445874` and `.3693022`. Central anchors correctly pass both controls. All hidden truth norms are nonzero, ranging from `2.5687954` to `51.4775744`. The verifier uses group norms, not relative errors divided by an individual near-zero rate.

The verifier checks interface shape, finiteness, fit quality, parameter recovery, and every frozen group. It records metrics and a numeric reward. Ordinary validation recomputes references without replacing frozen data; replacement requires `--generate`. The Dockerfile includes only public environment files, leaving the oracle, hidden preparations and truth, parameter value, reference, and author material outside the ordinary agent image. The task preserves the standard time limits and artifact collection.

## Distinction and trial interpretation

The archived thermal-bodies r12 task tests finite-window heat variance from stored-energy/current correlations. This task tests a finite long-time heat rate function whose linear tails retain rare endpoint energies of order observation time. The archived entropy-anomaly r4 author note explicitly separates its first two long-time cumulants from endpoint-sensitive full exponential-moment domains. This is a related physical theme with a distinct asymptotic observable and failure mechanism; the finite operational readout, exact calibration, large hidden margin, and independent reference address the previously recorded concerns about such an extension.

A later submission retaining the work rate because bounded-in-time typical energy differences are assumed irrelevant to all long-time fluctuations has made the intended physical mistake. A submission deriving the heat-domain restriction but failing its algebra, optimizer, interface, or code requires a separate diagnosis. No target-model outcome is inferred here.

## Reviewed SHA-256 hashes

| File | SHA-256 |
|---|---|
| `tasks/swirling-heat-tails/AUTHOR.md` | `ff2a8625711decbcff837099fc3411f81170fe9e52fa6c6f1a51e3c483dcdaa3` |
| `tasks/swirling-heat-tails/environment/Dockerfile` | `375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70` |
| `tasks/swirling-heat-tails/environment/README.md` | `06181e512422fed88b24d51b5720a83eb714b66e31ca25170c404365dd454eeb` |
| `tasks/swirling-heat-tails/environment/data/calibration.json` | `a93bbc40e3e42f0c3f3e94e901f93e68c659ccc29cf745b697c2918d86eb391c` |
| `tasks/swirling-heat-tails/environment/model.py` | `fc2ef2104bb4a90be240cf6c6e6dee4fbef62f857a6f5c37aab6ed0e92474aa6` |
| `tasks/swirling-heat-tails/environment/test_public.py` | `9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9` |
| `tasks/swirling-heat-tails/instruction.md` | `5f7a09478c61cf57a68f4768ea91836178bb26d37a8c22830df77879e6982833` |
| `tasks/swirling-heat-tails/solution/model.py` | `5cc9d0e2f75c052ea6b214085901b8a53fddfbd5cae86a9e4066c254aca1ea13` |
| `tasks/swirling-heat-tails/solution/solve.sh` | `bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644` |
| `tasks/swirling-heat-tails/task.toml` | `cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af` |
| `tasks/swirling-heat-tails/tests/data/calibration.json` | `a93bbc40e3e42f0c3f3e94e901f93e68c659ccc29cf745b697c2918d86eb391c` |
| `tasks/swirling-heat-tails/tests/data/hidden_truth.json` | `63468fffeef2c47425c52c0fdfb5232eb9d5ca7b86e3d5ceddb8e532da371b3f` |
| `tasks/swirling-heat-tails/tests/metadata.json` | `b9a20f300e1f93cb9193564b3af8439692e8525b3f207e92d4ba1644b7e7b661` |
| `tasks/swirling-heat-tails/tests/reference.py` | `0d7388262a76212a21d02bfd3e2e823239afb206c311acb684a17ed12ba1e580` |
| `tasks/swirling-heat-tails/tests/test.sh` | `74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c` |
| `tasks/swirling-heat-tails/tests/test_hidden.py` | `17400c4cbe5cdff85077429fb3551d1c9e64b23cf472353331f416e8c9681ca4` |
| `scripts/swirling_heat_tails_baseline.py` | `4a69d031c85c9a2047f0a669b083c4ac677db68112cb57ed2e2acbc11361833e` |
| `scripts/validate_swirling_heat_tails.py` | `b2c662b8af2a70df13415c83fe8c90d6fee3c2bdd4e61d2d2821b2154a19a0ab` |
| `results/expansion-swirling-heat-tails-r1-validation.json` | `3a6d91bcef38f10e1482d22b306d2b2cc8251c7544e4112c8d71d753153e7ded` |
