# Hard-rod current revision 1: independent physics review

Reviewed 2026-10-11. **Approved for frozen revision-1 evaluation. No blocking physics, source, reference, calibration, or tolerance issue found.** This review establishes scientific consistency, not target-model difficulty. The task author handles Docker controls separately and reports the expected oracle 7/7 and shortcut 4 passes/3 diagnostic failures. I made no task-source, data, or grading edits.

## Apparatus and mechanical mapping

The public specification is sufficient to determine the experiment: physical center density and rod length, elastic velocity exchange, independent initial velocity marks, a stationary positional hard-rod ensemble, signed center crossings at a fixed independent origin, and thermodynamic limit before long time. Both positions and marks are redrawn between realizations and retained during each trajectory. The three observables are defined as long-time cumulant rates, with negative SCGFs explicitly permitted. There is no unspecified thermal velocity law or finite-ring long-time averaging convention.

Compression `y_n=x_n-a*n` converts velocity exchange into freely moving marked points followed by sorting. A physical current density j shifts the compressed cut by `+a*j*t`, giving the reference's rates `c*p*(v-a*j)` and `c*(1-p)*(v+a*j)`. The moving-cut signs, physical rather than compressed density, and current orientation are correct.

The Poisson-flow entropy contraction gives the implemented implicit field equation. Its derivative with respect to the auxiliary field is strictly positive. The corresponding current saddle is always inside `(-v/a,v/a)` for positive length, so the full disclosed counting-field interval requires neither a branch change nor an endpoint saddle. The reference's small endpoint exclusion is far below the saddle margin in this domain. At zero length the independent bidirectional Poisson process is recovered.

Differentiation gives the oracle's exact mean and variance. Positivity of the variance is uniform on the allowed domain because the packing fraction is at most .72 and `abs(2*p-1)<=.5`. Speed multiplies every observable; it does not enter the implicit dimensionless field equation. The oracle and its weighted speed fit implement these facts correctly.

## Initial origin convention

The stationary physical-origin ensemble is not naively identical to a Poisson process under compression. The author correctly accounts for it. The origin-containing Tonks gap has the size-biased law of `a+Exp(c)`. Splitting its length-a and free portions yields a Palm Poisson configuration with one distinguished point or an ordinary stationary Poisson configuration, with a bounded origin shift. Finite-point changes and bounded-window counts do not alter the speed-t current LDP. This is stronger than assuming that any random O(1) correction is harmless for exponential moments.

The finite-time reference's exact convention, `x_n=y_n+a*n` with index zero the last initial free point left of zero, therefore has the same limiting SCGF as the public stationary Tonks ensemble. Its tail identity `P(J_t>=k)=P(Q_t(a*(k-1))>=k)` uses the correct index. The max-functions in the two Poisson means correctly cover cuts outside the two free ballistic rays. The finite enumeration covers the possible physical integer currents; it is used only at positive rod length.

## Shortcut and independent reference

The completed shortcut is a coherent centered shot-noise process with ordinary quasiparticle crossing rates, fractional dressed fluctuation charge, and a deterministic background. Its status as a real-valued effective current is explicitly documented. Its mean and variance equal the true interacting cumulants, and its full SCGF is exact at zero length. It is a valid completed approximation, not an integer-count claim or a truncated cumulant series. The physical error is freezing the rates and dressing under current conditioning.

The main reference minimizes the explicit Poisson-flow relative entropy and maximizes over physical current, without using the oracle's implicit auxiliary-field equation. The supplementary Skellam-tail calculation constructs the finite-time integer-current law from the microscopic rank mapping. Together these check both the asymptotic algebra and its physical interpretation. Tail-width refinement and zero-field normalization are present. The source does not assert that the observed 1/t correction holds exactly throughout the domain.

The frozen report contains 152 domain cases with maximum oracle/reference discrepancy `5.774e-15`. Hidden reference refinement is below `1.772e-16` in group relative norm. The four finite-time checks agree with the asymptotic contraction after extrapolation, including a largest displayed discrepancy about `4.23e-12`. Reflection, speed scaling, zero bias, positive variance, Jensen's bound, and the first two field derivatives are checked. The derivative-step adjustment recorded in AUTHOR is a numerical diagnostic refinement before evaluation; it did not alter task inputs or grading.

## Additional reviewer checks

In a separate installed numerical environment (NumPy 2.5.2, SciPy 1.18.0), I ran:

- 120 extra corner and short-rod cases, including lengths `1e-8` and `1e-5`; oracle and entropy contraction agreed within `5.685e-14` absolute.
- Sixteen rigid-translation renewal checks at p=1. This is an analytical extension outside the public velocity-fraction interval: all rods translate at speed v, so the spatial renewal-count SCGF has the independent Lambert-W expression `v*(W(a*c*exp(lambda+a*c))/a-c)`. It agreed with the oracle within `2.843e-14`.
- Sixteen nontrivial SCGF-zero checks at `lambda=log((1-p)/p)-2*a*c*(2*p-1)`, with residual below `2.380e-16`.
- Two explicitly negative limiting SCGFs at `rho=.6,a=1.2,p=.7,v=1.12`: lambda -1 gives `-.18045694886522845`, and lambda -2 gives `-.17119194047101946`. Finite-time sums at t=32,64,128 with enlarged Skellam tails agree after Richardson extrapolation within `3.331e-16`.

No warnings occurred in those checks. These supplement the pinned-environment validator. They do not constitute exhaustive proof of floating-point accuracy for every real input.

The author corrected one documentation sentence during review: the original fourth microscopic case has a negative finite-time estimate but a positive limiting SCGF. The current AUTHOR hash below includes that clarification. The extra negative-limit checks above require no changes to the frozen data or scientific sources.

## Calibration, grading, and public/private separation

There are 240 independent Gaussian readings at 48 settings, with fixed sigma .002. Nonzero known variance coefficients make the weighted linear fit strictly identifiable, including symmetric preparations with zero mean. Both controls recover speed `1.120079988864` with reduced chi-square `0.9553729351`. Public and private calibration bytes match, and all frozen hidden input lists match the reference definitions.

Across 256 noise draws, the speed range is `[1.119139810607,1.121227288303]`, and maximum reduced chi-square is `1.276736`. The largest oracle prediction error is `0.001095794` relative; the weakest shortcut diagnostic group remains above `0.5926586`. These margins are comfortably separated by the 4% gate. All four truth-group norms are nonzero, ranging from about 3.981 to 11.622. Grading uses these group norms and never divides by a small individual SCGF. The shared anchors appropriately pass both controls.

The verifier checks output shape and finiteness, reads its private calibration and frozen truth, and writes diagnostic metrics and a numeric reward. Ordinary validation recomputes frozen references; data replacement requires `--generate`. The Dockerfile copies only public environment files. Author notes, true speed, reference, hidden preparations, and physical solution are outside the agent image. Standard 600/60-second limits and submission artifact collection are retained.

This is distinct from Brownian tagged-particle memory and from the conditioned nonuniform density profile of the diffusive ring task. A later trial must be diagnosed from its final model and trajectory: retaining the fixed-state shot-noise closure is a physical error, while a correct mechanical mapping followed by a root, sign, or coding error requires a separate classification.

## Reviewed SHA-256 hashes

| File | SHA-256 |
|---|---|
| `tasks/hard-rod-current/AUTHOR.md` | `71da5b23fea4c7caf539b6eedf2507618a4eb651f6fa719030496d61d167a0ad` |
| `tasks/hard-rod-current/instruction.md` | `23107674ba10df1627ec8f35865c78bef2eab82477351ab18d507ab9327cbe40` |
| `tasks/hard-rod-current/task.toml` | `cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af` |
| `tasks/hard-rod-current/environment/README.md` | `3b14f12cd0cd02ce8e529da27126df29026e90117306da14bab809a34a408d74` |
| `tasks/hard-rod-current/environment/model.py` | `3c1d76aa2cbb9b7234c440ee97e84d463b05ad8794f050ae104afd869bd2135a` |
| `tasks/hard-rod-current/environment/Dockerfile` | `375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70` |
| `tasks/hard-rod-current/environment/test_public.py` | `9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9` |
| `tasks/hard-rod-current/environment/data/calibration.json` | `560f07a1643c242a79e1f6246918c7d62bfa97c836f84aefcf35e6ef03d0c70e` |
| `tasks/hard-rod-current/solution/model.py` | `2f545c9cc96a9c64945d8c31a072880a524a0ec8621723e1e1f50d495061e83d` |
| `tasks/hard-rod-current/solution/solve.sh` | `bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644` |
| `tasks/hard-rod-current/tests/reference.py` | `9219fe45697b5dcd218723bd0289e329421ef80a3138e06fccd074994d4b094c` |
| `tasks/hard-rod-current/tests/test_hidden.py` | `15421d378c8ca34ddecf0397abe2d3cfeaa31bb4e367d5e1a5022641e6aa958a` |
| `tasks/hard-rod-current/tests/test.sh` | `74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c` |
| `tasks/hard-rod-current/tests/metadata.json` | `b9a20f300e1f93cb9193564b3af8439692e8525b3f207e92d4ba1644b7e7b661` |
| `tasks/hard-rod-current/tests/data/calibration.json` | `560f07a1643c242a79e1f6246918c7d62bfa97c836f84aefcf35e6ef03d0c70e` |
| `tasks/hard-rod-current/tests/data/hidden_truth.json` | `764dad0a3bd089e23f945aa63cbd004b071793a66668e820838b23d5bb8f3332` |
| `scripts/hard_rod_current_baseline.py` | `ef031cfdc29f927174ff28b052f1f652580ec8b37c8912d15baaee4838a62c92` |
| `scripts/validate_hard_rod_current.py` | `eb1e7e239ac9176615d26376929bfb10f67d59eb752d807eab7aefa9381f4e73` |
| `results/expansion-hard-rod-current-r1-validation.json` | `ef1767bb60aecb0e562c12b77f80d1e6b46635da6b1e1cf18a76ae67ab1d4d66` |
