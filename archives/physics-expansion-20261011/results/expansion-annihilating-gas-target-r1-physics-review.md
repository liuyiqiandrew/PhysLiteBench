# Annihilating-gas-target revision 1: independent source review

Verdict: approve the reviewed source for evaluation. I found no blocking physics, specification, numerical, or grading issue. I reviewed every task file, the completed shortcut, validator, and saved validation report. No task source or data was modified, and no model trial was run for this review.

## Preparation and physical law

The public README specifies an independently inserted stationary tag, a Poisson background with independent symmetric velocity marks, mutual pair annihilation including stationary background particles, finite observation time, and fresh configurations on every repetition. Units, parameter ranges, and the tag's status at `p=0` are explicit. The continuous positions exclude triple collisions almost surely. No boundary condition or long-time approximation enters the hidden cases.

The first-hit generating function, its analytic branch, and the independence needed to combine the collision-selected rank with its renewal distance match the primary result checked in [the concept review](expansion-annihilating-gas-target-concept-review.md). A surviving incoming particle arrives at its initial distance divided by `v`, so the Poisson/gamma argument is `z=rho*v*t`. Independent no-hit events on the two sides give the square. The Bessel normalization at `p=0` is correct; there is no missing factor of two. At `p=1` and at time zero, the readout is exactly one.

The oracle's odd-coefficient recurrence correctly implements the quartic after writing `f(x)=x*g(x^2)`. Its subtraction is physically justified and is not concealed by clipping. Degree 256 has a uniform survival truncation bound `2*gammainc(257,93.6)=1.3207242e-43`. This bounds the omitted probability mass, while separate numerical checks address arithmetic error.

## Reference and microscopic validation

The contour calculation continues the root from zero along complex rays and extracts coefficients by Fourier integration. It is a different numerical representation of the same renewal law. The author correctly distinguishes it from the independent microscopic event queue. The queue maintains live neighbors, rejects stale events, removes both colliding particles, and records the tag's first contact. The initial interval `[0,v*t]` is sufficient by finite propagation speed; deleting particles outside it does not introduce a physical boundary for the finite-time arrival event.

The saved validator checks 29,325 small-system velocity/gap-permutation configurations and 400,000 independent microscopic histories across eight preparations. Enumeration discrepancies are below `3.13e-16`; Monte Carlo discrepancies are below 1.84 stated standard errors. These statistical checks are accurately labeled and are not used as grading truth. The saved 120-case domain check, coefficient signs, probability bounds, and two contour resolutions support the implementation across the full stated range.

My additional independent checks used the current source:

- An all-pairs chronological collision calculation agrees exactly with the neighbor heap on 1,200 random histories. The simpler calculation uses neither a heap nor linked neighbors.
- On those same histories, retaining or removing initial particles beyond the causal interval gives the same first-hit result in all 1,200 comparisons.
- Seventy-digit decimal coefficient calculations at nine mixtures, including both endpoints and points immediately around `p=1/4`, differ from the double-precision oracle by at most `5.56e-17`.
- Eighty-one additional endpoint, near-critical, tiny-time, and maximum-time cases agree with a 4096-point contour at radius `.975` within `4.86e-16`; survival stays within `[0,1]` and does not increase with observation time.
- Stored hidden inputs and values match the reference. Independently refined hidden predictions differ by at most `3.34e-16`.

## Calibration, shortcut, and grading

The public and private calibration copies match and contain the expected 256 readings: 32 positive-time settings repeated eight times, all with `p=0` and standard deviation `.001`. The strictly decreasing Bessel response identifies speed uniquely. The validator's 51 noiseless fits include the allowed endpoints; their maximum `3.79e-8` absolute fitting error is negligible against the 3% parameter gate. Both controls use the same bounded known-noise least-squares fit and recover `v=1.0800641205162462`, with reduced chi-square `0.93881`.

The shortcut implements the complete two-speed annihilating gas after deleting stationary background particles. It therefore has valid, nonzero predictions and is exact on the whole calibration family. It misses the altered background collision histories. The source does not claim that stationary particles protect the tag in every realization.

I reproduced all saved control group errors. The three mixture errors of the shortcut are 24.06%, 55.07%, and 66.15%; both controls pass the shared anchors. Across 256 noisy fits, the smallest diagnostic shortcut error remains 24.01%, and the largest oracle error is 0.05959%, against a 4% group gate. All group norms are nonzero. Fitting noise, reference precision, and coefficient truncation are far too small to explain the diagnostic separation.

The Dockerfile copies only public environment files and calibration. Author notes, private truth/reference, and the solution stay outside the ordinary image. The instruction permits replacing the response, while the tests retain the documented API, shared parameter check, calibration check, and four prediction groups.

Approval does not predetermine outcome classification. The exact renewal result is specialized. A future solver that models the collisions correctly but fails through insufficient sampling, an event-queue defect, or unsuccessful coefficient mathematics must be classified separately. A physical-failure claim requires evidence of an incorrect collision assumption in the retained code and emitted trajectory, as AUTHOR.md states.

## Reviewed SHA-256 hashes

| File | SHA-256 |
| --- | --- |
| `tasks/annihilating-gas-target/AUTHOR.md` | `86a956ae836d217d333a846ca21700bd26194d88aaf7835f19adbc7194b34599` |
| `tasks/annihilating-gas-target/environment/Dockerfile` | `375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70` |
| `tasks/annihilating-gas-target/environment/README.md` | `4c644f5d3c1c622061baf472f100fa2f21522d30505748c57e59b4be4d35565f` |
| `tasks/annihilating-gas-target/environment/data/calibration.json` | `c6abb9fb83a94d702f6dd64200baca2fab5b759d150f31031ff9c9866bb9b3b0` |
| `tasks/annihilating-gas-target/environment/model.py` | `a047ebf786647ea5470038b1df3e82141d4ee64f6ee27d4b4fff3c835dbfcbf0` |
| `tasks/annihilating-gas-target/environment/test_public.py` | `9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9` |
| `tasks/annihilating-gas-target/instruction.md` | `78f6931174679f4e88d394fc1a0b927065a40259fa38cec499791a0bab412888` |
| `tasks/annihilating-gas-target/solution/model.py` | `0b95b7923d8232b5193c0e883ce9143ef205a59a8eed0b318f8178c911b42dad` |
| `tasks/annihilating-gas-target/solution/solve.sh` | `bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644` |
| `tasks/annihilating-gas-target/task.toml` | `cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af` |
| `tasks/annihilating-gas-target/tests/data/calibration.json` | `c6abb9fb83a94d702f6dd64200baca2fab5b759d150f31031ff9c9866bb9b3b0` |
| `tasks/annihilating-gas-target/tests/data/hidden_truth.json` | `ab2d2bd895ff1b46e04a3d78ea2d9cdf708103331f0089ea4ad83c3360f0e0a5` |
| `tasks/annihilating-gas-target/tests/metadata.json` | `b9a20f300e1f93cb9193564b3af8439692e8525b3f207e92d4ba1644b7e7b661` |
| `tasks/annihilating-gas-target/tests/reference.py` | `991ecefeed86a4ab26d7576554566b3d4e2874ff7560a8f2538cb2c731bfe0e2` |
| `tasks/annihilating-gas-target/tests/test.sh` | `74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c` |
| `tasks/annihilating-gas-target/tests/test_hidden.py` | `15421d378c8ca34ddecf0397abe2d3cfeaa31bb4e367d5e1a5022641e6aa958a` |
| `scripts/annihilating_gas_target_baseline.py` | `e5b8584e29ed5198617988462cca82835b92ea6f87d67cdf1117e10bf3b2c3bd` |
| `scripts/validate_annihilating_gas_target.py` | `81a670f5c8726ab27a62bbd648765f85110fe887018f573e26e81c6e97cc5391` |
| `results/expansion-annihilating-gas-target-r1-validation.json` | `925c62e54cf705793c032b6edccfc003427f078c98ffd780e8a4fb75d466999c` |
