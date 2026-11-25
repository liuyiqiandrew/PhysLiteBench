# Driven mobile-trap survival revision 2: independent source review

Verdict: approve the reviewed revision for evaluation. I found no blocking physics, specification, calibration, or grading issue. This review covers all task files, the completed shortcut, validator, and saved validation report. No task source or data was changed and no model evaluation was run for this review.

## Apparatus and physical rate

The public preparation specifies independent Brownian drivers, a fresh Poisson trap cloud and fresh paths each realization, first contact killing, an infinite line before the time limit, and an ensemble survival probability averaged before its logarithm. Force signs, units, fixed long-time parameters, the deterministic target at zero temperature, and independently controlled trap diffusion are explicit. There is no common-bath ambiguity at zero target temperature. The public API and the oracle cover zero force, zero temperature, negative force, and the crossover within the stated domain.

With `u=mu*abs(F)` and `D=mu*T`, the conditional Poisson survival is exactly `exp(-rho*E_B range(x-B))`. Convexity gives the endpoint upper survival bound. The moving tube, drift-change density, and absorbing-interval cosine bound give the matching lower survival bound. Minimizing `(w-u)^2/(4D)+rho*abs(w)` yields the oracle's two branches, including the factors of four and the threshold `abs(F)=2*rho*T`. The optimum is `max(u-2*rho*D,0)`. For every fixed finite positive trap diffusivity, its expected-range correction is sublinear in time and does not enter the exponential rate.

At `T=0`, deterministic motion gives `K=rho*mu*abs(F)` directly. At `F=0`, `K=0`, while finite-time survival still decays. Reflection is implemented with `abs(force)`. Both branches match continuously with matching first derivatives. The rates have inverse-time units and are nonnegative. The oracle avoids subtracting nearly equal large terms in the low-force regime.

## Reference and independent checks

The private reference numerically minimizes the Gaussian velocity cost plus the Poisson empty-interval cost. Explicit endpoint candidates make the optimizer safe at zero drift and zero temperature. The finite-time endpoint moment and moving-tube calculation verify the limiting contraction through different physical bounds. The tube width correctly minimizes its linear clearance/endpoint cost plus its inverse-square confinement cost.

The saved validator reports 153 domain/crossover cases, contraction disagreement below `4.45e-16`, and 24 independent Brownian hitting-tail range quadratures with discrepancy below `7.11e-15`. I additionally checked:

- 270 direct Gaussian endpoint quadratures against the log-normal-moment implementation: maximum rate discrepancy `8.882e-15`, with no integration warnings.
- 135 drifted absorbing-interval survival probabilities, obtained by integrating the Dirichlet heat-kernel series: all exceed the implemented Girsanov/confinement lower bound; the minimum logarithmic margin is `0.2351`.
- 480 extra domain cases, including both force signs, zero force, both mobility endpoints, temperatures down to `1e-12`, and points immediately around the crossover: all oracle rates are finite and nonnegative, with maximum absolute contraction discrepancy `1.301e-12`.

The finite-time bounds are bounds on `-log(S(t))/t`, not a finite-time bracket around its limit. The source and author notes correctly preserve this distinction. The documented `1e-6` endpoint convergence check was inappropriate for the zero-force, zero-temperature case: its exact positive correction is `4*rho*sqrt(D_B/pi/t)`, reaching `8.7404e-5` at the validator's final time. The corrected finite-time tolerance is justified by this known correction and is unrelated to the 4% task grading gate. No physical formula needs changing.

## Calibration, shortcut, and grading

Both calibration copies match. There are 256 records at 32 nonzero-force, zero-temperature settings, with eight readings per setting and disclosed independent Gaussian noise. The positive linear basis identifies mobility uniquely. Both completed controls fit `mu=1.08005725292891` with reduced chi-square `1.02049`. The shortcut is a valid deterministic-mean-path auxiliary process, not an unfinished or numerically broken solver. It omits the rare drift reduction selected by survival conditioning.

I reproduced every saved control group error and checked frozen hidden inputs and values against the reference. All group norms are nonzero, including cold anchors containing exact zeros. The oracle's nominal relative errors are about `5.3012e-5`; the shortcut's three thermal errors are 347.55%, 14.20%, and 58.87%. Across the validator's 256 noise draws, the weakest thermal shortcut error remains 14.18%, while the worst oracle error is 0.01311%. These margins comfortably separate physical failure from fitting noise or reference precision.

The Dockerfile copies only public environment files and calibration data. Private truth, reference, author explanation, and correct solution remain outside the ordinary image. The neutral instruction permits changing the response. The public and private API checks, 3% parameter gate, reduced chi-square gate, and four 4% prediction groups match the author documentation. Revision 1 remains archived with its successful 3/3 outcome; this driven apparatus and exponential readout constitute a substantive revision.

## Reviewed SHA-256 hashes

| File | SHA-256 |
| --- | --- |
| `tasks/mobile-trap-survival/AUTHOR.md` | `d576aa1ef80465c8b83409d36d48f6b496f415dd2ef6f2b22a4068b9c8efc5bb` |
| `tasks/mobile-trap-survival/environment/Dockerfile` | `375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70` |
| `tasks/mobile-trap-survival/environment/README.md` | `3ace3bd290edfd67a6f3d34124daf182556ed8eef732912edab2cf67044479b5` |
| `tasks/mobile-trap-survival/environment/data/calibration.json` | `a95cc8ba8b758946f919022492ac71a7965324e008da6c9c152fcb7b39f0322e` |
| `tasks/mobile-trap-survival/environment/model.py` | `0673f918f82543c3991864f50f4a95b6e49bbd34933b8fb95fdae9a2de9a2ba2` |
| `tasks/mobile-trap-survival/environment/test_public.py` | `9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9` |
| `tasks/mobile-trap-survival/instruction.md` | `e35a9a24fe35be1fe6160d38b01c3fe4ba4b5c5e80acb5cae419ee27e53fc343` |
| `tasks/mobile-trap-survival/solution/model.py` | `dfa95344e79a9d41c2d15733a0ab85bcda812fafe9543788996ace06ff3b8c65` |
| `tasks/mobile-trap-survival/solution/solve.sh` | `bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644` |
| `tasks/mobile-trap-survival/task.toml` | `cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af` |
| `tasks/mobile-trap-survival/tests/data/calibration.json` | `a95cc8ba8b758946f919022492ac71a7965324e008da6c9c152fcb7b39f0322e` |
| `tasks/mobile-trap-survival/tests/data/hidden_truth.json` | `d9607575a53f413bef717abfce760e5ab760153c066446e7c71a37918835c484` |
| `tasks/mobile-trap-survival/tests/metadata.json` | `e7e73504277a03622823d3a03a52315f7383d161d0b94df65dcbb2a1f0c00a89` |
| `tasks/mobile-trap-survival/tests/reference.py` | `3e9daf522c12c64dd3c0b2a00af9f7e46f2d618201eaf27b4acfd15055a4df52` |
| `tasks/mobile-trap-survival/tests/test.sh` | `74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c` |
| `tasks/mobile-trap-survival/tests/test_hidden.py` | `17400c4cbe5cdff85077429fb3551d1c9e64b23cf472353331f416e8c9681ca4` |
| `scripts/mobile_trap_survival_baseline.py` | `fa9395734fce582b306859fea7be37fba10adaa8f86f6d3839844cf671572af3` |
| `scripts/validate_mobile_trap_survival.py` | `2ace43e8575ce70e3a31136cbcaa69b141178ca0e547db06584fec8bd1125f42` |
| `results/expansion-mobile-trap-survival-r2-validation.json` | `c695fc27fbc6d6a71c8829cc0bc25e9d80b2408b257f37cf955e5065b90e5878` |
