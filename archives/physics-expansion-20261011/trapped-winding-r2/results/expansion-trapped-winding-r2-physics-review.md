# Trapped winding revision 2: independent source review

Reviewed 2026-10-11. **Verdict: approve the reviewed scientific package. No blocking physics, specification, calibration, numerical-reference, or grading issue was found.** Docker controls are being run separately by the author; this review does not establish model difficulty. I changed no task, control, validator, or data files and ran no model evaluations.

## Physical law and public scope

The public specification selects a stationary planar OU process with strictly positive stiffness, temperature, and mobility, no finite central core, and continuous unwrapped winding. It explicitly takes winding and radial occupation from the same particle trajectory, averages the joint observable before taking its logarithm, and measures the strict long-time rate. The physical radial probe is independent of the unknown mobility. Units, preparation, parameter ranges, noise, and API are complete. The absolute value in the public logarithm creates no ambiguity: the actual joint expectation is positive.

Writing \(a=\mu k\), \(D=\mu T\), and \(\nu=|\lambda|\), conditioning on the radial trajectory gives

\[
\mathbb E\exp\left[-\int_0^t(D\lambda^2/R_u^2+sR_u^2)\,du\right].
\]

The radial backward generator is \(D(\partial_{rr}+r^{-1}\partial_r)-ar\partial_r-D\nu^2/r^2-sr^2\). Its regular positive ground mode is \(r^\nu\exp[-(\omega-a)r^2/(4D)]\), with \(\omega=\sqrt{a^2+4Ds}\), giving

\[
\Gamma=(1+\nu)\omega-a.
\]

The ground-state transform is a conservative radial OU generator with radial dimension \(2\nu+2\) and relaxation \(\omega\). Stationary endpoint overlaps are positive and finite. This checks the selected origin branch and asymptotic rate, rather than inferring a fixed-frequency result from a weak winding limit. The solution implements this law as `probe*shifted + 4*diffusion*radial/(shifted+relaxation)`, avoiding cancellation in the small radial-probe limit. Zero probes, either marginal, and negative angular probes are handled correctly.

The completed control takes winding and occupation from independent stationary copies of the same OU process. This is a valid auxiliary joint law with rate \(a\nu+\omega-a\), not a malformed characteristic function. Both complete marginals are exact for every allowed mobility. The omitted contribution \(\nu(\omega-a)\) comes from sharing the radial trajectory. Revision 2 therefore repairs the previous winding-marginal approximation and tests a different physical assumption within the same task family. The public same-particle statement selects the intended model without supplying its solution.

## Reference and independent Cartesian check

I checked the stationary two-radius kernel, including its prefactor, Bessel argument, endpoint powers, and stationary weights. It agrees with the derivation in the [concept review](expansion-trapped-winding-r2-concept-review.md). Generalized Laguerre order \(\nu/2\) correctly absorbs both endpoint factors; the divided Bessel power remains well behaved at the finite reference times. The \(s=0\) kernel reduces to the original winding kernel, while \(\lambda=0\) reduces to the stationary radial-occupation transform. The reference uses finite-time integration and log slopes, independently of the oracle's ground-mode substitution. The hypergeometric check is accurately described as another representation of that same kernel.

The validator's Cartesian check provides a separate route from the stated stochastic dynamics. For one Cartesian coordinate, the exact sampled covariance is

\[
\Sigma_{ij}=(T/k)e^{-a|t_i-t_j|}.
\]

Let \(Q=2s\,\mathrm{diag}(w_i)\) contain the trapezoidal occupation weights. The two Cartesian coordinates together contribute \(\det(I+Q^{1/2}\Sigma Q^{1/2})^{-1}\). The code's `-2*sum(log(diag(cholesky)))` is exactly its logarithm. The covariance update is the Gaussian identity

\[
\Sigma_s=\Sigma-\Sigma Q^{1/2}(I+Q^{1/2}\Sigma Q^{1/2})^{-1}Q^{1/2}\Sigma.
\]

The resulting endpoint correlation gives the integer angular harmonic of the tilted isotropic Gaussian endpoint pair. The hypergeometric endpoint formula and its normalization are correct. For integer probes, the endpoint phase and unwrapped winding have identical harmonics; the validator does not extend this shortcut to fractional probes. This check uses the Cartesian Gaussian path law, not the radial oscillator kernel.

I extended its four checks from 513 to 1,025 time points without changing repository files. The maximum refined Richardson log-characteristic error was **1.567e-12**, down from the report's 2.478e-11 at 257/513 points. Errors decreased by approximately four before extrapolation, as expected for trapezoidal occupation. I also rebuilt the weighted Gaussian law from the exact OU transition's tridiagonal precision matrix and used the elementary second-harmonic expression where applicable. Its 513-point results agreed with the covariance implementation within **1.179e-10**; subtracting large precision log determinants makes that alternative less well conditioned. The agreement and refinement support the Cartesian check's algebra and continuum limit.

## Calibration, frozen truth, and tolerances

Both public and private calibration files contain the same 252 records: 42 marginal settings repeated six times, each with additive standard deviation 0.001 inverse seconds. Their byte hashes match. Angular-axis settings alone identify mobility globally; the radial marginal supplies a second nontrivial family and also increases with mobility. Both controls use the same weighted fit, retaining the mobility dependence in both \(a\) and \(D\).

I independently refit the checked-in records and recomputed every frozen hidden value from the private kernel. Both controls recover mobility **1.0899862089646062**, with reduced chi-square **0.9548807450**. All frozen predictions agree with a fresh reference evaluation within 1e-12. The reported and independently reproduced group errors are:

| Group | Oracle relative error | Shortcut relative error |
|---|---:|---:|
| Shared anchors | 0.0011164% | 0.0011164% |
| Joint positive | 0.0006595% | 42.0339% |
| Joint negative | 0.0006856% | 39.7227% |
| Joint strong radial | 0.0006042% | 47.0999% |

The saved validation report and source agree on 256 noise draws, 41 noiseless recovery cases, and 144 random/boundary domain cases. Worst noise-draw oracle error is 0.01566%; the smallest joint shortcut error remains 39.7146%. Maximum domain oracle/reference discrepancy is 1.022e-9 absolute, the finite-time kernel/series discrepancy is 2.887e-15 in log characteristic, and hidden-reference refinement changes values by at most 5.907e-10. These errors are far below the 4% prediction threshold. The 3% mobility gate and 1.5 calibration chi-square gate are also insensitive to the recorded noise and optimizer precision.

The hidden test uses group RMS error normalized by the group's reference RMS, equivalent to the validator's norm ratio. The shared-anchor group contains nonzero values, so its zero-probe anchors do not cause division by zero. Prediction groups lie within the disclosed domain and add no preparation, cutoff, or measurement assumption.

## Separation and review limits

The Dockerfile copies only the public README, starter, public test, and calibration data from `environment/`. The private truth, independent reference, author explanation, parameter value, and hint remain outside the agent image. The unfinished starter exposes the allowed approximation but not a fitted answer. The completed shortcut is a separate control. The solution installer changes only `/app/model.py`; verifier output includes a numeric reward and diagnostic group errors. Existing 600/60-second limits are retained.

Approval applies to the hashes below. It does not predict a 0/3 result. Any eventual failure still needs source and trajectory review: retaining independent marginal records is the intended physical mistake; an eigenfunction, branch, quadrature, or implementation error requires its own classification. Revision 1 must remain preserved and is not a separate new task contribution.

## Reviewed SHA-256 hashes

All 17 task files, the completed control, validator, and saved validation report were reviewed. Generated Python caches are excluded.

| File | SHA-256 |
|---|---|
| `tasks/trapped-winding/AUTHOR.md` | `04318fe48e6eab8f4072aac0b0d379f7d13453f4dbe559c52f1379e0af3f7400` |
| `tasks/trapped-winding/environment/Dockerfile` | `375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70` |
| `tasks/trapped-winding/environment/README.md` | `0a6606c86234293a26eacf8b5015924f95b47cbbb5e9e5328b3409fcfdd26d7e` |
| `tasks/trapped-winding/environment/data/calibration.json` | `02171aca58dab31ccd935e1c6de99b5da8a33780e510a5bc258272b0cf34c4e6` |
| `tasks/trapped-winding/environment/model.py` | `35da803cbb496a8cdd5f49d0865204c92173407f2d89e1a64ed4e13a4c45d9a2` |
| `tasks/trapped-winding/environment/test_public.py` | `9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9` |
| `tasks/trapped-winding/hint.md` | `43296d5cc59bc9aa7c3cdd21a9f973f8800dfe3581a734dcc8a533e0a252594f` |
| `tasks/trapped-winding/instruction.md` | `fce2fee917b55aadf6706cf39831589fc2ded9765ba254b7b6fe8f8b9db92a28` |
| `tasks/trapped-winding/solution/model.py` | `3e204df3d96f8ed8989faced3b22b657e0a13404148b2d89a251a8554071fdd4` |
| `tasks/trapped-winding/solution/solve.sh` | `bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644` |
| `tasks/trapped-winding/task.toml` | `cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af` |
| `tasks/trapped-winding/tests/data/calibration.json` | `02171aca58dab31ccd935e1c6de99b5da8a33780e510a5bc258272b0cf34c4e6` |
| `tasks/trapped-winding/tests/data/hidden_truth.json` | `a0e83060dfe3d6928056d9b8d6e55738f37f0e10ad956aa520d9c6740b9d81ca` |
| `tasks/trapped-winding/tests/metadata.json` | `e7e73504277a03622823d3a03a52315f7383d161d0b94df65dcbb2a1f0c00a89` |
| `tasks/trapped-winding/tests/reference.py` | `cc471887c7c02d031c4205388d193ca63a667f3823a9e22368773c8ea9fdbe0b` |
| `tasks/trapped-winding/tests/test.sh` | `74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c` |
| `tasks/trapped-winding/tests/test_hidden.py` | `17400c4cbe5cdff85077429fb3551d1c9e64b23cf472353331f416e8c9681ca4` |
| `scripts/trapped_winding_baseline.py` | `8800fa4c90cf1f0604a2f156e50148a647c6e591b8822a0d648ae48ff54eba2d` |
| `scripts/validate_trapped_winding.py` | `38d580df6143486adde198951988f1a3c1addf8398e341905c1977d141589cd2` |
| `results/expansion-trapped-winding-r2-validation.json` | `faa144390744511b62ca28a567f8c079a3b0d02c384ddf1ee70565a476f94b61` |
