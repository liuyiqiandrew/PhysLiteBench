# Colored spin memory, revision 1

This task tests a temporal noise assumption, distinct from the spatial common-noise correlation in qubit-control and driven-spin-pair. Calibration observes only the undriven transverse coherence of one spin. The full stationary Ornstein-Uhlenbeck field process is specified publicly, including its known correlation time and its continuity across drive changes. Inferring its effect under transverse drives requires retaining correlations between the field and spin, even though calibration is fully described by one scalar decay envelope.

For stationary detuning covariance sigma^2 exp(-|t-s|/tau), the integrated phase variance is

    Var(phi(t)) = 2 sigma^2 tau [t - tau(1-exp(-t/tau))].

The calibration Ramsey probability is [1+exp(-Var(phi)/2)]/2. Its time dependence identifies sigma uniquely when tau is known. The completed shortcut uses the exact corresponding free-evolution rate gamma(t)=sigma^2 tau(1-exp(-t/tau)) in the driven Bloch equation

    dr/dt = Omega cross r - gamma(t) (rx, ry, 0).

It evaluates the calibration envelope analytically and integrates this equation accurately for driven experiments. Its clock is absolute time since preparation and is not reset between segments. Thus there is no fitting error, phase-variance factor error, segment-clock error, or numerical shortcut. Its physical error is assuming that a reduced free-dephasing law can be reused unchanged while a noncommuting drive acts. Positive, time-dependent free-dephasing rates alone do not establish that this reuse is valid.

For the physical oracle, write the Gaussian-weighted conditional Bloch density as a sum of normalized probabilists' Hermite functions. Mode n has OU decay n/tau. Multiplication by the detuning couples modes n to n-1 and n+1 with coefficients sigma sqrt(n) and sigma sqrt(n+1). Hence

    dc_n/dt = Omega cross c_n - (n/tau)c_n
              + sigma zhat cross [sqrt(n)c_(n-1) + sqrt(n+1)c_(n+1)].

The stationary, initially independent preparation has c0 equal to the prepared Bloch vector and all higher modes zero. Evolve the entire coefficient vector continuously across segments; the measured Bloch vector is c0. Forty modes are sufficient for the advertised parameter/control range as checked against sixty modes. The calibration branch uses its exact analytical limit for efficient repeated fitting.

The independent reference discretizes the normalized OU noise coordinate on [-8,8] using a conservative nearest-neighbor generator with exact discrete Gaussian detailed balance. It evolves probability-weighted Bloch vectors with a stiff ODE solver, preserving field history across segments. It uses 321 cells in grading. Doubling the grid to 641 cells gives the expected factor-four error reduction. This joint-noise-coordinate approach follows the stochastic-Liouville formulation used for fluctuating-field spin dynamics; see [Cai et al.](https://arxiv.org/abs/2407.06074) and the broader colored-noise qubit setting in [de Keijzer et al.](https://arxiv.org/abs/2401.11758).

Calibration uses seed 9323, true width 0.95 radians/second, and 200 measurements with independent Gaussian errors at 0.6% of the largest probability. The parameter tolerance is 3% and reduced chi-square limit 1.5. Hidden groups use absolute probability RMSE with limit 0.025, well above parameter and numerical uncertainty. Both correct and shortcut predictions match noiseless calibration exactly.

Checked-in calibration fits width 0.95072628, relative error 0.0765%, reduced chi-square 0.99084. Oracle hidden RMSE is at most 0.000130; shortcut errors are 0.05330 to 0.13618. All 256 additional noise realizations pass calibration and parameter checks, with maximum width error 0.438%. At fitted-width extrema the oracle remains below 0.000754 and the shortcut above 0.05289.

Independent checks: 321-cell reference error below 1.51e-5; 641-cell error below 3.77e-6; forty versus sixty Hermite modes below 4.7e-15 on hidden sequences. The phase covariance integral agrees below 9e-16, zero-noise deterministic dynamics below 1.1e-10, the quasistatic limit below 1.5e-9, and the white-noise limit below 6.6e-6. Additional checks exercise maximum allowed controls and the general Hermite solver's Ramsey limit.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 scripts/validate_colored_spin_memory.py colored-spin-memory
```

Only use `--generate` to intentionally replace both calibration copies. The report is `jobs/colored-spin-memory-validation/summary.json`. Local isolated-copy pytest passes all 7 oracle tests; the shortcut passes 4 interface/calibration/parameter tests and fails the 3 hidden prediction groups. The parent coordinates Harbor controls and paired gpt-5.6-luna/high trials. No empirical failure-rate claim is made before those trials and trajectory review.
