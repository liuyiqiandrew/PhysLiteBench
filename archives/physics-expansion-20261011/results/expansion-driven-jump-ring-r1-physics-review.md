# Driven jump ring r1 physics review

Independent source review, 2026-10-11. **Approved for evaluation, with no blocking finding.** The target is three fresh, frozen, unhinted `gpt-5.6-luna` High trials, following the user's changed model selection. Scientific validity does not establish the requested failure rate.

I reviewed the current public apparatus, starter, complete shortcut and oracle, independent reference, validator, frozen data, public/private tests, Docker/configuration files, and AUTHOR.md. The [completed validation report](expansion-driven-jump-ring-r1-validation.json), with wall time `1.272194125`, uses the revised positive-series remainder bound and stronger long-cycle refinement. The parent confirmed its completed run handle and subsequent Docker control outcomes; this review's numerical findings come from source inspection, the saved validation report, and the supplemental checks described below.

## Physics and interface

The public rate convention is complete. The forward/backward ratio on bond `i` is `exp(E_i-E_(i+1)+F/3)`, so the loop affinity is exactly `F`. At zero affinity, the stationary weights are proportional to `exp(-E_i)` and satisfy detailed balance. All barriers exceed all allowed site energies. More importantly, the rates are explicitly the exact defining jump law, so no unstated time-scale correction to an Arrhenius approximation is needed at stage switches.

Clockwise and reverse counts have weights `+1/3` and `-1/3`. The declaration that fractional turns are recorded rules out a completed-winding convention at finite times. The state continues through every boundary, rates do not depend on the counting field, the starting preparation is periodic stationary, and the answer is the natural logarithmic growth per period. Chronological ordering, units, input ranges, and the long-time limit are explicit. All calibration and hidden inputs satisfy the public domain.

The tilted generator correctly multiplies only jump terms by their counting factors. Its escape diagonal remains physical. Each stage propagator is positive, and their chronological product has a unique positive Perron eigenvalue. Thus the oracle's `log(max(eigvals(M).real))` selects the correct growth rate throughout this small irreducible model. No complex-eigenvalue ordering ambiguity arises for this positive matrix.

## Complete shortcut and exact calibration

The shortcut solves the ordinary periodic stationary distribution correctly, propagates the exact tilted law for one complete period, and takes the logarithm of its total mass. This is the correct long-time answer for independent repetitions that redraw the initial state from that distribution between periods. It preserves within-period dynamics and the mean current, but misses correlations between counts in successive periods. Its outputs are finite and nonzero; it is a complete alternate physical process rather than an unfinished prediction method.

Spatially uniform calibration makes the clockwise and reverse jump hazards independent of the occupied state. Each stage then contributes two independent Poisson counts, giving

    duration * [a*(exp(chi/3)-1) + b*(exp(-chi/3)-1)].

Summing this expression over stages is exact for both continuous and independently repeated periods. It is linear in the unknown attempt rate. The supplied calibration contains nonzero coefficients, so its one-parameter weighted least-squares objective is strictly convex and identifies the rate uniquely. The explicit fit is appropriate; no nonlinear search or unknown equilibrium population is required.

The observed fits agree to roundoff: `1.09984364798`, with reduced chi-square `0.8802904`. The implementation's unrestricted scalar least-squares solution is inside the stated rate interval for the frozen data and all validated noise draws.

## Independent reference and convergence

The reference constructs a discrete Poisson-opportunity process directly from source states, destinations, signed jumps, physical rates, and null-event probabilities. It sums weighted opportunity counts and propagates normalized positive vectors through repeated cycles. It uses neither the oracle's matrix-exponential routine nor its eigenvalue solver, and independently reconstructs the reverse-bond indices.

The final remainder test is mathematically valid. Writing `T_n` for the current matrix term, every later norm ratio is at most `r=clock*duration*||step||_1/(n+1)`. Once `r<1`, the omitted sum is bounded by `||T_n||_1*r/(1-r)`. The stopping rule compares this entire bound with the accumulated kernel norm. It does not incorrectly use an unweighted Poisson tail after counting-field tilting.

Long-cycle propagation requires convergence of both the normalized state and the last-cycle mass growth. The latter is the Perron growth, without the initial-boundary bias that would remain in a short cumulative log-mass average. The refined check tightens the state/growth threshold, reduces the series tolerance by a factor of 100, and changes the initial state from uniform to a basis state. The largest hidden change is `1.04e-14`, negligible relative to grading. Positive rates and bounded stage durations avoid a reducible or singular limit.

The validator's 72 random domain/rate comparisons have maximum absolute oracle-reference difference `4.56e-15`. It also checks zero field, cycle-origin invariance, static fluctuation symmetry, and equilibrium detailed balance. I separately tested 48 extreme preparations, spanning both rate bounds, fields `-2,0,2`, minimum/maximum durations, minimum/maximum uniform rates, and strongly heterogeneous energies/barriers. Tight reference propagation from a third initial state agreed with the oracle within `1.31e-15`. Repeating the entire period doubled the asymptotic result within `3.44e-15`. A centered field derivative verified equality of oracle and shortcut mean currents on all hidden preparations within `8.7e-12`. These supplemental checks used the available NumPy 2.5.2/SciPy 1.18.0 environment; task images retain their pinned versions.

## Noise, grading and separation

All 256 simulated calibration datasets produce rates in `[1.09937559,1.10053960]`; maximum calibration chi-square is `1.35894`. Every parameter fit satisfies the 3% gate. The largest oracle hidden error over those fits is `0.0568%`, well below the 4% threshold. The weakest shortcut diagnostic group remains above `18.08%`; the other three minima exceed `21.12%`, `26.57%`, and `41.87%`.

Group RMS signals are `0.07837` for uniform anchors, `0.05903` for varied cycle duration, `0.05336` for barrier drive, `0.02555` for static heterogeneity, and `0.04618` for reversed protocols. The normalization therefore does not exploit a vanishing reference group. The hidden cases separately expose continuity even in a static landscape and under genuine modulation; this task does not claim to isolate only an adiabatic geometric term.

Frozen hidden values are recomputed by ordinary validation and checked against their exact inputs by the verifier. Public/private calibration copies are byte-identical. The ordinary image copies only public files; fitted truth, hidden cases, solution, reference, and author explanations are excluded. The neutral instruction permits replacing all implementation code. Standard 600/60-second limits remain unchanged.

## Evaluation diagnosis

Retaining independent periods, or treating the ordinary stationary state as the tilted path ensemble, is the intended physical mistake. A solver that recognizes uninterrupted counting and then reverses a matrix product, tilts escape diagonals, confuses time and period normalization, or mishandles an eigenvalue has a different failure. Inspect final code and trajectories before classifying any zero. The simple three-state correction may be easy for the target model; only the fresh three-trial batch can establish difficulty.

Reviewed SHA-256 values:

| File | SHA-256 |
|---|---|
| `environment/README.md` | `2344eab1a6fc8396714995340bd1d879c0daf17db13766d786154ef0c2ecb3e8` |
| `environment/model.py` | `c8693258bf54357e3e482987f1099a89aba5b25957861a7fdc1a4b911e19b291` |
| `solution/model.py` | `6f0c25d3cdd41f1163fdcde3f2d5f9f28fb1c8692a521609aac15fb1e2241d53` |
| `tests/reference.py` | `9694137bacfff8e68cec7e088a215163c8fec5271a36604747c3e44ca0ee3afc` |
| `tests/test_hidden.py` | `7814f388084165fb1c371fd216c3e2f3b1525789fb0ca5689258ecf0b2969d42` |
| `scripts/driven_jump_ring_baseline.py` | `21079bb0a2698b16b7e4314a3c8e7a2d532bb579e4f3a041bc083303bbe4d211` |
| `scripts/validate_driven_jump_ring.py` | `876b4983c2f924342bedcc793d8c4894bd83da1603fb10e242f7245b411eb8cf` |
| Public/private calibration | `63252c3ea223204e5f7c8e01476be5614b46f6e662a6f2cea996fcf89382c0a5` |
| Hidden truth | `1266fbfe5c4320e0e3190f3aad5981312377fbd6b97c7e999ac90c428470d559` |
| `AUTHOR.md` | `1b0211c20a37a4afda9a1b4476a6b9a9be2f23078a822ceeb0d10516cc275a57` |
| Validation report | `8a74380653f38881fac4b3100f166e72ef5f16d3e049a9553694b7e0be1efb15` |
