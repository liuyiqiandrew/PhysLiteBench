# Colored spin memory, revision 2

Calibration observes only the undriven transverse coherence of one spin. The stationary Ornstein-Uhlenbeck field process is specified publicly, including its known correlation time and continuity across drive changes. Revision 2 supplies an exact constant-control reduced spin channel in the starter. The missing physical constraint is the joint spin-field memory needed when composing consecutive control intervals.

For stationary detuning covariance sigma^2 exp(-|t-s|/tau), the integrated phase variance is

    Var(phi(t)) = 2 sigma^2 tau [t - tau(1-exp(-t/tau))].

The calibration Ramsey probability is [1+exp(-Var(phi)/2)]/2. Its time dependence identifies sigma uniquely when tau is known. Calibration contains one interval per experiment, so the completed shortcut and oracle fit it identically.

Write the Gaussian-weighted conditional Bloch density in normalized probabilists' Hermite modes. Mode n has OU decay n/tau. Multiplication by detuning couples adjacent modes, giving

    dc_n/dt = Omega cross c_n - (n/tau)c_n
              + sigma zhat cross [sqrt(n)c_(n-1) + sqrt(n+1)c_(n+1)].

Initially c0 is the prepared Bloch vector and all higher modes vanish. The physical oracle evolves the entire coefficient vector continuously across all intervals; the measured Bloch vector is c0. Forty modes converge against sixty throughout the checked control range.

The completed shortcut instead constructs the full 3 by 3 Bloch map for each constant-control interval by evolving each initial Bloch basis vector with an independent stationary field. It then composes these reduced maps. This is a consistent factorized-bath closure: every individual interval channel is correct for its assumed preparation. Its error is treating the spin and field as independent again at each control change. The unconditional OU field marginal remains stationary, but the field is correlated with the spin after the first interval. Stationarity alone does not justify replacing the joint state by a product. No numerical instability, malformed channel, or fitted-parameter error is needed for failure.

The independent reference discretizes the OU coordinate on [-8,8] with a conservative nearest-neighbor generator satisfying exact discrete Gaussian detailed balance. It evolves probability-weighted Bloch vectors with a stiff solver while preserving history. Grading uses 321 cells; 641 cells reduce the error by about four. This joint-noise-coordinate approach follows the stochastic-Liouville formulation used for fluctuating-field spin dynamics; see [Cai et al.](https://arxiv.org/abs/2407.06074) and the broader colored-noise qubit setting in [de Keijzer et al.](https://arxiv.org/abs/2401.11758).

Calibration uses seed 9323, true width 0.95 radians/second, and 200 measurements with independent Gaussian errors at 0.6% of the largest probability. The parameter tolerance is 3% and reduced chi-square limit 1.5. Hidden groups use absolute probability RMSE with limit 0.025, well above parameter and numerical uncertainty. These data, apparatus, oracle, reference, hint, and grading are unchanged from revision 1.

The checked-in data fit width 0.95072628, relative error 0.0765%, and reduced chi-square 0.99084. Oracle hidden RMSE is at most 0.000130. The shortcut agrees with the oracle for the two single-interval groups and fails the changed-axis sequence with RMSE 0.0362009. All 256 additional noise realizations pass calibration and parameter checks, with maximum width error 0.438%. At fitted-width extrema, the oracle remains below 0.000754, and the shortcut's changed-axis error is 0.0361211 to 0.0363432. The extrema check is a sensitivity check, not an exhaustive hidden Monte Carlo calculation.

Independent checks give 321-cell reference error below 1.51e-5 and 641-cell error below 3.77e-6. Forty versus sixty Hermite modes differ by less than 4.7e-15 on hidden sequences. The phase covariance integral, arbitrary initial Bloch channel checks, zero-noise dynamics, maximum controls, and general-solver Ramsey limit agree to numerical precision. Quasistatic and white-noise limiting errors are below 1.5e-9 and 6.6e-6. Splitting one constant drive into two pieces changes the oracle probability by less than 6e-17; the shortcut changes by 0.03107, exposing its artificial bath reset. All checked reduced channels are contractions.

Revision 1 used a scalar time-local free-dephasing equation under transverse drives and scored plain 2/3, hint 3/3. The failed plain submission derived the joint Hermite generator but discarded higher modes between intervals. A fixed-parameter repair carrying its full state lowered the changed-axis RMSE from 0.0362009 to 0.0000676. Revision 2 uses a complete interval-channel closure to test this physical assumption directly. Preserve the distinction between that closure and an accidental array-handling bug when reviewing new traces. Revision 1 sources, scripts, results and source hashes are in `archives/colored-spin-memory-r1/`; raw trials remain in their original frozen jobs.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 scripts/validate_colored_spin_memory.py colored-spin-memory
```

Only use `--generate` to intentionally replace both calibration copies. The new report is `jobs/colored-spin-memory-r2-validation/summary.json`. Local isolated-copy pytest passes all 7 oracle tests; the shortcut passes 6 tests and fails only the changed-axis prediction group. The parent coordinates Harbor controls and paired gpt-5.6-luna/high trials. Revision 2 difficulty remains unmeasured until those trials and trajectory review finish.
