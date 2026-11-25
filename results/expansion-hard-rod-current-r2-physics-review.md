# Hard-rod current revision 2: independent physics review

Reviewed 2026-10-11. **Approved for frozen revision-2 evaluation. No blocking physics, source, reference, calibration, or tolerance issue found.** This is a scientific review, not a prediction of target-model difficulty. The author runs Docker controls separately. I made no task-source, data, or grading edits and ran no model evaluations.

## Apparatus and physical derivation

The public specification states equal masses, unequal lengths fixed to physical rods, velocity exchange, no overtaking, independent symmetric velocity marks, and the stationary ring preparation with equal fixed population counts. The thermodynamic limit precedes time. Each repetition redraws the order, positions, and velocities; the exponential moment averages that whole ensemble before its logarithm. These details distinguish the apparatus from a typical fixed-order calculation and from variable-length mathematical models that exchange lengths along with velocities.

Center compression `x_n=y_n+A_n`, with `A_(n+1)-A_n=(a_n+a_(n+1))/2`, maps the mechanics to sorted free trajectories. The fixed-total-length ring limit gives Poisson compressed positions of density `c=rho/(1-rho*length)`, independent of local fair binary length marks. Packing is at most .72 and the minimum individual length is positive, so there is no geometric singularity in the disclosed domain.

For positive current j, a long-rod fraction f in the approximately j*t net crossed labels gives mean crossed length `b=length*(1-contrast+2*contrast*f)`. The free crossing intensities are `c*(v-b*j)/2` and `c*(v+b*j)/2`. Their flow entropy is supplemented by `j*D(f||1/2)`. The factor j counts the crossed block, not all rods in a ballistic interval. Minimizing over composition yields the oracle's `logcosh` term and its stated implicit field equation.

The implicit equation is strictly increasing. Its derivative is `1+c*b_star*cosh(s)`, where `b_star=length*(1-contrast*tanh(c*length*contrast*sinh(s)))` is positive. Its current saddle stays strictly inside `b_star*j<v`, validating the positive Poisson rates. The bracketing interval covers the unique root; absolute field correctly implements reflection. The stable `2*sinh(s/2)**2` and `logaddexp` forms avoid unnecessary cancellation and overflow in the stated field range.

The exact mean is zero and the variance growth is `rho*v*(1-rho*length)`. The leading contrast correction to the SCGF is cubic in absolute field, as documented in AUTHOR. Thus the limiting function is even but generally not three-times differentiable at zero. The validator correctly uses the order-step error of its finite variance difference instead of imposing spurious fourth-order smoothness. The oracle has the correct monodisperse, zero-bias, and reflection limits.

## Independent references

The main reference uses nested current/composition optimization of explicit Poisson relative entropies and binary composition entropy. Its composition bounds enforce the free crossing cone. It does not call the oracle root or import its reduced field equation. The reference variance coefficient uses the independent compressed-density expression, which simplifies to the same physical result.

The supplementary finite-time reference uses the exact center rank event `J>=k` iff `Q_t(-A_(1-k))>=k`. The `k-1` index and both endpoint half-lengths are correct. A block of n compression steps contains n-1 full independent lengths and two half endpoint lengths; the implemented two binomial factors give exactly that distribution. The Skellam noncentral-chi-square identity has the correct tail orientation. The finite current sum covers every possible positive crossing count.

The moment computed is `E exp(field*max(J,0))` at positive field. Its difference from `E exp(field*J)` is bounded by one, and the positive-field limiting SCGF is strictly positive, so both have the same growth rate. This does not assert that their finite-time distributions or ordinary moments agree.

The free-origin construction requires the stated Poisson free-gap/Palm correction before comparison to a stationary physical origin. Bounded endpoint lengths, local mark reweighting, and bounded-window Poisson counts preserve the speed-time rate. AUTHOR addresses this explicitly rather than treating arbitrary random order-one differences as harmless for exponential moments. Taking the thermodynamic limit first also removes the global half-population constraint from each finite crossed block.

The initial long-time numerical issue is correctly identified and preserved in AUTHOR. A noncentral CDF can underflow for a tail whose exponential current weight makes it important. The corrected `rare_skellam_tail` exponentially tilts two Poisson variables to means differing by k. It then sums their near-mode probabilities with the exact inverse-tilt factor. Its cost and sign conventions are correct. The pruning test includes the composition probability and current exponential weight in its Chernoff bound; it does not discard tails based on unweighted probability alone. Its finite offset and Poisson windows are far inside the required error budget for the exercised microscopic domain.

All four recorded microscopic preparations now agree with the contraction after the t=64,128 extrapolation within `8.882e-16`. The report also retains t=16,32. These are selected supplementary convergence checks, not a claim of an exactly linear inverse-time correction at arbitrary small fields. The numerical repair changed neither oracle nor task data or grading and preceded all model trials.

## Validation and extra reviewer checks

The frozen scientific report contains 120 random/corner cases with maximum oracle/reference discrepancy `6.662e-15`. Hidden tolerance refinement changes group predictions by at most `2.141e-16` relatively, and oracle/reference errors are below `1.668e-15`. The finite-difference variance error is `4.900e-6`, consistent with the physical cubic-absolute-field correction.

I additionally ran read-only checks in NumPy 2.5.2 and SciPy 1.18.0:

- 98 random, small-contrast, and small-field cases, including contrasts `1e-8` and fields `1e-5`; maximum oracle/nested-reference absolute discrepancy `7.106e-15`.
- A separate simultaneous two-variable current/composition optimization on 32 cases, using the closed flow cost and a bounded population search; maximum discrepancy `5.179e-12`. An initial reviewer-only clipped-simplex search stuck at zero current for two small fields. The population search resolved this optimizer defect without changing any task file.
- Eight rare Skellam tails checked against an independently summed 80-digit Decimal Poisson/Bessel series. Log probabilities range from -17.7 to -1362.5; maximum absolute log error is `2.274e-13`, including probabilities far below double-precision underflow.

The additional contraction and calibration checks produced no warnings. These supplement, rather than replace, the pinned-environment validator.

## Calibration, shortcut, grading, and separation

There are 240 independent Gaussian readings at 48 settings with fixed sigma .001. Half the settings are monodisperse finite-field SCGFs and half are mixture variances. Both controls are exact throughout this calibration family for every allowed speed. Nonzero unit-speed response coefficients make the common speed identifiable by weighted linear regression. Both fit `1.1199586271326543`, with reduced chi-square `1.09450018`.

The completed shortcut solves the full monodisperse implicit current law at the mean length. It therefore repairs revision 1's frozen-current-feedback approximation before omitting the new crossed-block composition fluctuation. It is a valid mechanical auxiliary gas, finite and nonzero on the diagnostic SCGFs. The public fit remains intentionally unfinished, and the separate completed control implements it correctly.

Across 256 noise draws, the fitted speed lies in `[1.119453727883,1.120363533491]`, and maximum reduced chi-square is `1.305281`. Maximum oracle prediction error is `.000487743`; the weakest shortcut diagnostic group remains above `.3834232`. Nominal shortcut errors are `.5231353`, `.5031032`, and `.3836364`, well beyond the 4% gate. Shared anchors correctly pass both controls. Every truth-group norm is nonzero, ranging from `3.328143` to `9.337175`; grading never divides by an individual zero mean or zero-bias SCGF.

I checked equality of public/private calibration, calibration-input ordering, and every frozen hidden input list. The verifier checks array shape, finiteness, calibration, parameter recovery, and each prediction group, and writes metrics plus numeric reward. The Dockerfile includes only public environment files. Author notes, true speed, hidden cases and truth, reference, and oracle remain outside the ordinary agent image. Data replacement requires explicit `--generate`; ordinary validation reads and checks the frozen data. Standard time limits and artifact capture are retained.

Revision 1's 1/3 outcome is preserved in the archive and described honestly in AUTHOR. This revision changes the physical preparation and tested approximation rather than resampling the earlier mechanism. A later submission retaining typical crossed composition has made the intended physical error; a correct composition derivation followed by an optimization or coding mistake requires a separate classification.

## Reviewed SHA-256 hashes

| File | SHA-256 |
|---|---|
| `tasks/hard-rod-current/AUTHOR.md` | `320cde22bc362e5399a5eb200ef171886d67f3da819218f73216615e281d4550` |
| `tasks/hard-rod-current/environment/Dockerfile` | `375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70` |
| `tasks/hard-rod-current/environment/README.md` | `8be0f90e10ae20ebbd108011c396de7c9e5074b6af92d682a9d9d470e42694c0` |
| `tasks/hard-rod-current/environment/data/calibration.json` | `7e2391e1835b4d13254683f40b28722d3055101ea35e9cd2294ac21cd2ba9c1a` |
| `tasks/hard-rod-current/environment/model.py` | `6e30faf4748ba885e1cc2a978d36d37064b40e7cc4f71e65b2aedaadf084e4b4` |
| `tasks/hard-rod-current/environment/test_public.py` | `9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9` |
| `tasks/hard-rod-current/instruction.md` | `23107674ba10df1627ec8f35865c78bef2eab82477351ab18d507ab9327cbe40` |
| `tasks/hard-rod-current/solution/model.py` | `f1d4c0f0a53303bd69dfd372b6807974d1d1210bd6bd87229f4ff7b3cabe461d` |
| `tasks/hard-rod-current/solution/solve.sh` | `bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644` |
| `tasks/hard-rod-current/task.toml` | `cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af` |
| `tasks/hard-rod-current/tests/data/calibration.json` | `7e2391e1835b4d13254683f40b28722d3055101ea35e9cd2294ac21cd2ba9c1a` |
| `tasks/hard-rod-current/tests/data/hidden_truth.json` | `0a4893fc4dd13493ab0bc25737a4f5e4c56d8b43a78de449bb2833d8a3cea3f3` |
| `tasks/hard-rod-current/tests/metadata.json` | `e7e73504277a03622823d3a03a52315f7383d161d0b94df65dcbb2a1f0c00a89` |
| `tasks/hard-rod-current/tests/reference.py` | `76e4605d9b7b09874aac61b2dc659c8558e0f574d854191d48a5fde856e56eac` |
| `tasks/hard-rod-current/tests/test.sh` | `74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c` |
| `tasks/hard-rod-current/tests/test_hidden.py` | `15421d378c8ca34ddecf0397abe2d3cfeaa31bb4e367d5e1a5022641e6aa958a` |
| `scripts/hard_rod_current_baseline.py` | `5e2483a4c83c7c0aca2260059c3aa9c626e7f1ba705a26b10672ececbaa5d445` |
| `scripts/validate_hard_rod_current.py` | `f89017f86ed3cbfd2cc444f1e09284da25481502ee6f351263d6e18459e22574` |
| `results/expansion-hard-rod-current-r2-validation.json` | `e148f9895f81e57cf1497b5d2b7a14babb3fa4a80e264b987193e39c10915c05` |
