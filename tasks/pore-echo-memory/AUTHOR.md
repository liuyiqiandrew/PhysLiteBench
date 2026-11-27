# Pore echo memory, revision 1

For a path through the gates, let n_j be the sum of the signed exposure counts at site j. Conditional on this path, the accumulated phase is Gaussian with variance (gain × width)² times the sum of n_j². Write F(n) = exp[−(gain × width)² |n|² / 2]. The mean echo is the walk average of F(n).

For the covariance, let m be the counts of an independent walk in the partner scan. The two spins in one pore use exactly the same site offsets. Conjugating the second scan changes its phase sign. The requested covariance is E[F(n−m)] − E[F(n)] E[F(m)], where these expectations now average only over walks. The oracle propagates a probability distribution over position and counts, then sums over end position.

The bulk curvature is width² E[|n|²]. Each calibration measurement constrains this same positive field width; the one-exposure record alone identifies it. The shortcut reproduces the exact phase variance and cross covariance from two-time site coincidence probabilities. It treats the accumulated phases as jointly Gaussian after averaging over motion. This second-cumulant approximation agrees at zero gain through second order but misses the mixture of path-dependent variances at finite gain. General background on restricted spin echoes and this approximation is available in [Grebenkov (2007)](https://pmc.polytechnique.fr/pagesperso/dg/publi/2007_07.pdf).

## Independent verification

The reference propagates site magnetizations at fixed values of the four site phases, then integrates the resulting trigonometric polynomial over the pore distribution. Three relative phases suffice because the signed total number of exposures fixes the sum of all four Fourier indices. The uniform Fourier grid has 2L + 1 points per axis for L exposures across the participating scans. Each coordinate has degree at most L, so the transform is alias-free. Sealed traps give two direct checks: identical scans have variance 1 − |E[M]|², and scans in distinct sealed traps have zero covariance.

Seed 426007 sets the 12 calibration and 21 prediction inputs. Curvatures are rounded to seven decimals. The absolute threshold 0.002 allows a full-scale signal error of 0.2%, well above calibration quantization and numerical error. Both controls fit width 1.073000001 rad/ms with maximum curvature error below 0.000000055. Local maximum absolute errors were:

| Family | Physical model | Temporal Gaussian |
|---|---:|---:|
| Moving echo | 4.2 × 10⁻¹⁰ | 0.11451 |
| Same-scan covariance | 1.4 × 10⁻¹⁰ | 0.25841 |
| Two-scan covariance | 1.2 × 10⁻¹⁰ | 0.08339 |

An independent peer checked the shared-disorder averaging and propagated scans through 180,000 sampled pores. A moving echo measured 0.123851 ± 0.000481 against the oracle's 0.124245. The covariance for two distinct scans was 0.026564 ± 0.000208 against 0.026434.

Run from the repository root. Add `--regenerate` to the validator to write both calibration copies and the private truth deliberately.

```bash
python3 scripts/validate_pore_echo_memory.py
python3 scripts/run_science.py tasks/pore-echo-memory --agent oracle --trials 1
python3 scripts/run_science.py tasks/pore-echo-memory --agent oracle --trials 1 --solution-model scripts/pore_echo_memory_baseline.py
python3 scripts/run_science.py tasks/pore-echo-memory --model gpt-5.6-luna --reasoning-effort high --trials 3 --concurrency 3 --restricted-network
```
