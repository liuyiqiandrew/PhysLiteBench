# Mobile-trap survival revision 1: independent physics review

Reviewed 2026-10-11. **Approved for frozen revision-1 evaluation. No blocking physics, source, reference, calibration, or tolerance issue found.** This establishes scientific consistency, not target-model difficulty. The author separately reports Docker controls of oracle 7/7 and shortcut 4 passes/3 diagnostic prediction failures. I made no task-source, data, or grading edits and ran no model evaluations.

## Public apparatus and physical result

The public README specifies independent Brownian drivers with the `sqrt(2*D)` convention, a point target initially at zero, independent Poisson trap positions, freely crossing noninteracting traps, and killing at first contact. The infinite line precedes the long-time limit. Both initial positions and all paths are averaged before the logarithm. Trap diffusivity is positive, and target diffusivity is held fixed and finite while time grows. The readout is explicitly a leading asymptotic coefficient, not a finite-time survival estimate. These conditions determine the law without a preparation, boundary, or annealed-versus-conditioned ambiguity.

For a fixed continuous target path x, each marked Brownian trap path hits precisely when its initial point lies in the interval traced by x-B. Poisson thinning gives `S[x]=exp(-rho*E Range(x-B))`. Range is convex and even; Brownian symmetry makes its expectation an even convex functional of x, minimized at zero. Therefore `S[x]<=S[0]` for every fixed target path, and averaging the target preserves the inequality. This directly establishes the needed continuum Pascal bound rather than assuming independent target-trap relative paths.

The stationary-target first-passage integral is correctly doubled for traps on both sides. With the stated Brownian convention, it gives `-log S[0]=4*rho*sqrt(D_B*t/pi)`. For a lower survival bound, the empty interval, exterior no-entry, and target-confinement events are independent and jointly sufficient. The ground-state martingale gives confinement probability at least `exp(-pi**2*D_A*t/(4*ell**2))`; using this bound avoids the incorrect treatment of the first term in an alternating series as a lower bound.

Minimizing the empty-interval/confinement exponent gives the stated optimal half-width and a log-survival-coefficient gap `3*(pi**2*rho**2*D_A/4)**(1/3)*t**(-1/6)`. It vanishes for every fixed allowed preparation. D_A=0 is treated exactly. Thus the oracle's `C=4*rho*sqrt(D_B/pi)` is correct, including its factor of four and units `s**(-1/2)`. Target diffusivity can disappear from this leading measurement even though it affects finite-time survival.

AUTHOR accurately notes that the original Blythe–Bray paper did not prove its Pascal claim. The direct expected-range proof supplied here resolves that issue for the exact one-dimensional continuum apparatus.

## Independent reference and numerical certificate

The private reference computes the stationary-target coefficient by quadrature of the one-trap complementary-error-function hitting law, rather than copying the oracle's closed coefficient. It independently minimizes the confinement and initial-void cost in logarithmic scaled half-width. The common limiting value is certified by the opposing survival bounds. This is an analytical bounding reference with numerical integrals, not a simulation of the many-particle survival probability or an unsupported finite-time extrapolation.

The transformation `ell=u*t**(1/3)` removes time from the width optimization. The reference evaluates bounds on the normalized logarithm directly and does not exponentiate probabilities at times as large as `1e36`. The positive optimization cost and explicit stationary-target branch keep the bounds ordered. Quadrature tolerances are refined by three orders of magnitude.

The frozen validator covers 104 domain and corner cases. Maximum oracle/reference absolute discrepancy is `4.441e-16`; maximum width-optimization gap discrepancy is `1.777e-15`. Its five times verify the slow `t**(-1/6)` squeeze, with maximum final relative gap `8.688e-6`. The bound itself, not the last finite time alone, establishes the asymptotic result. Eighty Dirichlet spectral confinement probabilities independently satisfy the ground-state lower bound and probability ceiling.

Additional read-only reviewer checks in NumPy 2.5.2 and SciPy 1.18.0 were:

- Sixteen independent Gaussian first-moment integrals for the Brownian expected range; agreement with the no-entry quadrature within `4.441e-16` absolute.
- 288 additional bound checks, including target diffusivities zero, `1e-18`, and `1e-9`, both trap-diffusivity endpoints, and times through `1e36`; gap discrepancy below `4.441e-16`.
- A separate absorbing diffusion operator on 127, 255, and 511 interior grid points. Its extrapolated confinement probabilities at seven dimensionless times agreed with the continuum series within `4.168e-10`; grid refinement changed them by at most `6.285e-9`.

No warnings occurred. These supplement the pinned-environment validator without replacing or modifying its report.

## Calibration, shortcut, and grading

The shortcut is a coherent auxiliary process: an immobile target among independent traps of diffusivity `D_A+D_B`. This preserves the exact relative diffusivity for one target-trap pair. Its physical mistake for multiple traps is removing their common target motion: two true relative coordinates have increment covariance `2*D_A*dt`. It also misses the atypically confined target paths selected by long survival. Its coefficient is positive and finite throughout the domain.

All 256 calibration readings use stationary targets at 32 densities, each repeated eight times, with fixed sigma `.001`. Both controls coincide exactly for every allowed D_B. Their positive response amplitude is linear in `sqrt(D_B)`, so the weighted fit uniquely identifies the positive diffusivity. Both recover `1.1001703533125362`, with reduced chi-square `1.12105077`. The source's unconstrained amplitude-squared fit is appropriate for these positive, well-resolved readings; all tested noise realizations remain within the disclosed diffusivity interval.

Across 256 noise draws, fitted diffusivity ranges from `1.099726219449` to `1.100258331435`, and maximum reduced chi-square is `1.303008`. Maximum oracle relative prediction error is `.000124454`. The minimum diagnostic shortcut errors remain `.2656223`, `.4660203`, and `.9461447`, well beyond the 4% gate. Stationary anchors correctly pass both controls. All truth-group norms are nonzero, between `2.984558` and `4.518874`.

I verified the public/private calibration match, calibration-input ordering, and each frozen hidden input list. The verifier checks output shape, finiteness, calibration quality, parameter recovery, and group-normalized prediction error, then writes metrics and numeric reward. It does not restrict the solver. The Dockerfile copies only public environment files; hidden cases, truth, parameter, reference, oracle, and author notes remain private. The starter fit is intentionally unfinished and has a separate completed shortcut. Data replacement requires `--generate`; ordinary validation checks existing data. Standard time limits and artifact collection are preserved.

The mechanism is distinct from archived fixed-trap escape-rate memory, finite-cross-section survivor-conditioned transport, and switching-surface mean capture. A later model retaining independent relative traps after fitting D_B has made the intended physical error. A correct shared-path argument followed by a factor, numerical, or implementation error requires separate classification. No trial outcome is inferred from this review.

## Reviewed SHA-256 hashes

| File | SHA-256 |
|---|---|
| `tasks/mobile-trap-survival/AUTHOR.md` | `0d0ae583b9105d81dae1dc2114aae0d2007d1044688bb232bdc973a61a7e3705` |
| `tasks/mobile-trap-survival/environment/Dockerfile` | `375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70` |
| `tasks/mobile-trap-survival/environment/README.md` | `b162f0faf359b57ee04a1c098c2e1e9d6a89a42ee7d561f07a2e901c562d48fa` |
| `tasks/mobile-trap-survival/environment/data/calibration.json` | `eb08e9732e498ad1a61127f130f9e34f9ba1e80db084e844866d4b19f9a4cec9` |
| `tasks/mobile-trap-survival/environment/model.py` | `ce4f5c72e011bb474e6e6f785a7aa144da59b316af5cbc3e3d97ae94612c1d39` |
| `tasks/mobile-trap-survival/environment/test_public.py` | `9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9` |
| `tasks/mobile-trap-survival/instruction.md` | `182ae0e1ecacb969b11a01dc9467e0b3f77a35563d3ff756eae2c67a5168bc3b` |
| `tasks/mobile-trap-survival/solution/model.py` | `49529d9997442703727141eeb1d969e99288b2896580b5f3c9ab399178c42fff` |
| `tasks/mobile-trap-survival/solution/solve.sh` | `bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644` |
| `tasks/mobile-trap-survival/task.toml` | `cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af` |
| `tasks/mobile-trap-survival/tests/data/calibration.json` | `eb08e9732e498ad1a61127f130f9e34f9ba1e80db084e844866d4b19f9a4cec9` |
| `tasks/mobile-trap-survival/tests/data/hidden_truth.json` | `1b40546aca25bd12a54fa3c46da9681c7f901b6cd8caa5ce0bcdf80c976b8d97` |
| `tasks/mobile-trap-survival/tests/metadata.json` | `b9a20f300e1f93cb9193564b3af8439692e8525b3f207e92d4ba1644b7e7b661` |
| `tasks/mobile-trap-survival/tests/reference.py` | `8990db6986f9af39ee101a8fbf28b1c36a0238018a4ddb0aecc7ec8b8bca2a07` |
| `tasks/mobile-trap-survival/tests/test.sh` | `74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c` |
| `tasks/mobile-trap-survival/tests/test_hidden.py` | `085274e8ae509492b936c67261f80cfa7884f04eec51373124d9ae71a7aa858c` |
| `scripts/mobile_trap_survival_baseline.py` | `d74532dd6487fdb3d19c3278dea001be0c5c4b24fbd7cd7db94211ef7e43a533` |
| `scripts/validate_mobile_trap_survival.py` | `0d336fca5a9e99be1f05253f839a5a684191aea8c3b97853aeb55bf14c9adf39` |
| `results/expansion-mobile-trap-survival-r1-validation.json` | `118f143452bc8f3a80fc210854434210f8467ccbb4af327bc58c174a32f6bba5` |
