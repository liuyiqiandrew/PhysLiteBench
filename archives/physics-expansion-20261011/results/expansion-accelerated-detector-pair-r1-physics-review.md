# Accelerated detector pair revision 1: independent physics review

Reviewed 2026-10-11. **Approved for frozen revision-1 evaluation. No blocking source, physics, calibration, reference, or tolerance issue found.** This review does not establish target-model difficulty. Docker controls are checked separately by the task author. I made no task-source, data, or grading changes.

## Public apparatus and transition convention

The public description fixes the canonically normalized massless scalar vacuum, transverse worldlines, common proper time, identical real coupling signs, free detector gaps, initial detector states, initially uncorrelated field, and resolved final state. It also excludes switching transients, depletion, recoil, additional field radiation, direct detector interactions, and higher-order frequency shifts from the retained weak-coupling rate. The stationary Golden-Rule convention is sufficient to resolve the apparent long-time/before-depletion issue. The observable is a proper-time transition coefficient at leading order, not an infinite-time population derivative at fixed coupling.

Both symmetric and antisymmetric readouts count only gg. An accelerated Bell state can also transition to ee, but that channel is correctly excluded. The two lowering amplitudes into gg are `(1/sqrt(2),1/sqrt(2))` for the symmetric state and `(1/sqrt(2),-1/sqrt(2))` for the antisymmetric state, up to an irrelevant common phase. Hence the resolved rates are precisely

```
mu**2 * (G_local + G_cross),
mu**2 * (G_local - G_cross).
```

There is no extra factor of two. At vanishing cross correlation either Bell state has one excitation and decays at the single-detector downward rate; at coalescence the symmetric/antisymmetric channels become bright/dark. The oracle implements these signs and factors correctly.

## Spectrum and complete physical shortcut

Pulling back the public Wightman function gives the stated sinh-squared interval. With Fourier sign `exp(+i*omega*s)`, the oracle's local downward spectrum and normalized cross factor have the correct sign, Planck factor, length dependence, and zero-acceleration limit. `expm1` avoids local-spectrum cancellation at high acceleration. Throughout the stated finite-separation domain, the subtraction in the antisymmetric rate is numerically safe. The smallest corner rate checked below remains positive.

The complete shortcut is the actual correlation spectrum of stationary inertial detectors in a homogeneous thermal scalar bath at `T=a/(2*pi)`. Its spatial factor is sinc(omega*L), including the correct signed collective interference for both Bell states. It therefore reproduces all local-detector calibration for every acceleration, and both inertial-pair calibration preparations at every stated separation. The error is substituting inertial thermal spatial correlations for the vacuum correlations sampled along two accelerated worldlines. The local Unruh spectrum alone does not determine this cross spectrum. This is a physical approximation with finite, nonzero predictions, not an unfinished or numerically broken control.

The analytic normalization and cross spectrum agree with [Hu and Yu, Entanglement dynamics for uniformly accelerated two-level atoms](https://arxiv.org/html/1501.03321), equations 13–18 and the collective-state population equations. The public finite-worldline apparatus independently determines the same response.

## Independent reference and numerical accuracy

The reference directly Fourier-integrates the pulled-back scalar correlator. It contains neither the closed cross factor nor the local Planck response. For the downward contour shift `z=s-i*eta`, the multiplying factor `exp(omega*eta)` is correct. The chosen depth lies strictly inside the lower analytic strip; the real cosine term minus the imaginary sine term correctly reconstructs the full line integral. The inertial branch uses the correct rational correlator. The guard at `a*u>700` only removes an exponentially negligible tail. QUADPACK's infinite oscillatory integral, tolerance refinement, and contour-depth changes address tail and integration error rather than relying on a single finite integration window.

The frozen validator report checks 88 random and boundary cases, including zero and small acceleration. Its maximum absolute oracle/reference discrepancy is `7.4663e-15` per unit coupling. Refined hidden references change by at most `4.3827e-13` in group relative norm. Positive and negative Fourier frequencies are integrated separately for moderate accelerations; the reported KMS residual is `2.2332e-16` absolute. Positivity, dimensional scaling, and the Bell-rate sum identity are also checked. These tests do not manufacture upward rates by invoking KMS.

I added read-only numerical checks in an independent installed environment (NumPy 2.5.2, SciPy 1.18.0):

- Forty symmetric/antisymmetric corner cases with accelerations `0,1e-8,1e-4,.1,10`, gaps `.5,2`, and separations `.1,3`, using contour depth factor `.41` and tolerance `1e-13`. Maximum absolute oracle/reference discrepancy was `2.054e-15`; maximum relative discrepancy was `1.224e-11`, on a small positive rate. The smallest checked unit-coupling rate was `3.3153e-5`.
- Forty independent field-commutator checks. Subtracting the directly integrated negative-frequency spectrum from the positive-frequency spectrum agrees with the light-cone result `sin(omega*d)/(2*pi*L*sqrt(1+(a*L/2)**2))`, where `d=2*asinh(a*L/2)/a`, within `8.882e-16`. At zero separation the result is `omega/(2*pi)`. This checks the vacuum normalization separately from the thermal factor.
- Explicit two-detector lowering matrices reproduce the signed amplitudes above. No integration warnings occurred in these additional checks.

The reference construction is independent of the closed-form oracle. Like any finite numerical sampling, the domain checks are not a uniform proof over every real input, but the tested limits and analytic formulas leave no identified numerical gap relevant to grading.

## Calibration, uncertainty, and file separation

The 224 readings cover 28 settings. The unknown squared coupling multiplies known positive coefficients; weighted least squares therefore has a unique minimum. Both completed controls recover `0.00011000005861722155`, with reduced chi-square `0.9805965364`. The private and public calibration files are byte-identical, and all frozen hidden input lists match their reference definitions.

All 256 noise realizations pass the calibration and parameter gates. Fitted squared couplings range from `0.0001099863339074` to `0.0001100130570753`; maximum reduced chi-square is `1.270966`. The largest oracle hidden relative error is `0.000124238`, far below the 4% gate. The weakest shortcut diagnostic group remains above `0.406419`. The diagnostic groups' minimum rates are above `1.5001e-5`; small antisymmetric rates elsewhere in the allowed domain do not create a hidden near-zero normalization problem. The shared-anchor group correctly passes both controls.

The verifier checks the advertised NumPy output shape and finiteness, uses its own calibration copy and frozen truth, and records both diagnostic metrics and numeric reward. The ordinary validator verifies frozen predictions without regenerating them; `--generate` is the explicit data replacement path. The Dockerfile copies only public environment files. The true coupling, reference, solutions, hidden preparations, and author explanation remain outside the agent image. The standard 600-second agent and 60-second verifier limits are unchanged.

The archived circular-detector task shares the scalar-field setting but concerns temporal correlations on one rotating worldline. This task keeps local response exact and probes spatial cross correlations, so the physical omission is distinct. Any later failure review should distinguish choosing the inertial thermal spatial spectrum from deriving the correct pullback and then making a sign, factor, or quadrature error.

## Reviewed SHA-256 hashes

| File | SHA-256 |
|---|---|
| `tasks/accelerated-detector-pair/AUTHOR.md` | `22150de45e44c2a2e009ecd302fc1d89c7364e823ee8246efe66601adf00439a` |
| `tasks/accelerated-detector-pair/instruction.md` | `6a5ef18a542d5fd42298ccab151355ecf224974f3b465d9a6299768988739b62` |
| `tasks/accelerated-detector-pair/task.toml` | `cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af` |
| `tasks/accelerated-detector-pair/environment/README.md` | `49ebac71e6951adae67d07893b3af228994ab0b6b35a415441fe7b9a991316ff` |
| `tasks/accelerated-detector-pair/environment/model.py` | `5ad51197b99d09eb2b76e8557b59490a1c5a7948993040baeffdf0d0b60322f1` |
| `tasks/accelerated-detector-pair/environment/Dockerfile` | `375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70` |
| `tasks/accelerated-detector-pair/environment/test_public.py` | `9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9` |
| `tasks/accelerated-detector-pair/environment/data/calibration.json` | `fccfc63cbda8dd08300672b3d2b266d839a07407c2ea9a76857641dc6e01f101` |
| `tasks/accelerated-detector-pair/solution/model.py` | `0ca9274454fd17fc00e8a1ed2d75db5a3a9b2e034aeac4b5737bc590fa19beaa` |
| `tasks/accelerated-detector-pair/solution/solve.sh` | `bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644` |
| `tasks/accelerated-detector-pair/tests/reference.py` | `a19f21f97cac03a2ca0bb0f1d0ba3e280d11d2470454fef1318a9fac181ba97a` |
| `tasks/accelerated-detector-pair/tests/test_hidden.py` | `e1f3ab3f12457852ecfd3b27b9a66729f5798c105c48595f5733c40bb7db3483` |
| `tasks/accelerated-detector-pair/tests/test.sh` | `74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c` |
| `tasks/accelerated-detector-pair/tests/metadata.json` | `b9a20f300e1f93cb9193564b3af8439692e8525b3f207e92d4ba1644b7e7b661` |
| `tasks/accelerated-detector-pair/tests/data/calibration.json` | `fccfc63cbda8dd08300672b3d2b266d839a07407c2ea9a76857641dc6e01f101` |
| `tasks/accelerated-detector-pair/tests/data/hidden_truth.json` | `b9cd19e69ae5735ced6ff01ac3f27e4e65f832dc79ff32e563ac2309ee720ac1` |
| `scripts/accelerated_detector_pair_baseline.py` | `db3c6e3cf5d0a2873104da5de7a334aeded9e021c5b953dc469de47ae6e05971` |
| `scripts/validate_accelerated_detector_pair.py` | `0c0a87fd4ea92472d975ab1b280393f916a31acc6c30568ce3368c252a4833e9` |
| `results/expansion-accelerated-detector-pair-r1-validation.json` | `9956341b77ca66ede0eb7a4c2ac36d8bd2004dfce48d8d47034a813120ff986d` |
