# Exclusion current, revision 1

This task tests whether the solver connects exact single-counter statistics to particle conservation across a channel. Public calibration sees only stationary mean transfer through one boundary. The starter already implements the correct four-state exclusion dynamics and the exact finite-time variance of each individual counter. Its only extra assumption is independence when combining counter signals. The apparatus explicitly records all counters on the same physical trajectory and time window.

A site can hold at most one particle. Reservoir exchange rates have injection/removal ratio exp(mu/k_B T), and the equal-energy internal hopping rates are symmetric. The reservoir laws therefore obey local detailed balance. Reservoir chemical potentials are opposite, so f_right=1-f_left. With common rate Gamma, the stationary density equations are

    0 = Gamma*(f_left - 2*n1 + n2),
    0 = Gamma*(n1 - 2*n2 + f_right).

Hence every bond has mean current Gamma*(f_left-f_right)/3. The mean left count is Gamma*t*tanh(b/4)/3, which identifies Gamma by weighted linear regression. Correct and shortcut calibration predictors agree exactly.

The finite-time counting model uses a state generator G with columns summing to zero. For one measured jump increment q, form off-diagonal matrices A_ij=G_ij*q and B_ij=G_ij*q^2 by summing physical transitions. Keep the original escape diagonal in G. Conditional count moments obey

    p' = G*p,
    m' = G*m + A*p,
    s' = G*s + 2*A*m + B*p.

Initially p is the stationary occupation distribution and m=s=0. Sum m and s over final occupations, then subtract the square of the mean to obtain centered variance. The oracle uses q equal to the entire electronic signal increment w dot DeltaN for each physical jump. The shortcut solves these equations separately for each bond, then returns sum(w_i^2 Var(N_i)). Thus both controls contain exclusion, reverse transitions, finite-time occupation memory, and exact marginal counting variance; neither uses a Poisson approximation for an individual counter.

The missing joint correlations are enforced by exact pathwise identities:

    N_left - N_middle = n1(t) - n1(0),
    N_middle - N_right = n2(t) - n2(0).

Difference counts are bounded occupation changes even when each individual count has variance growing with time. Conversely, at long times all three count currents have the same fluctuating transport part, so their sum has nine times one current's variance growth rate. Independent variance addition gives three times that rate. These are physical consequences of counting the same particles, rather than a numerical precision issue or an unidentified parameter.

The private reference uses explicit transitions in a different state ordering and closed-form stationary probabilities. It uniformizes the continuous-time process using an independent Poisson clock, propagates first and second moments over discrete paths including null events, and sums the Poisson mixture. The clock tail extends 25 events beyond its 1e-15 upper quantile. This avoids the oracle's augmented matrix exponential. A second independent check derives occupation-difference variances from the closed two-density relaxation matrix Gamma*[[-2,1],[1,-2]], without any counting generator. General background on counting generators in exclusion transport is given in [Mesoscopic Full Counting Statistics and Exclusion Models](https://arxiv.org/abs/cond-mat/0312659); finite-time observables require the full propagator, as discussed in [Finite time large deviations via matrix product states](https://arxiv.org/abs/2108.11418).

Calibration has 200 measurements, true rate 1.3 inverse seconds, seed 9326, and independent Gaussian noise with standard deviation 0.6% of the largest noiseless count. The checked-in fit is 1.29950703, relative error 0.0379%, reduced chi-square 1.08204. Both controls pass calibration and parameter recovery. Hidden errors are RMS-normalized within each group, with limit 0.04. Oracle errors are at most 0.000380; shortcut errors are 5.30865, 2.90990, and 0.63050.

All 256 additional calibration-noise trials pass, with maximum rate error 0.1922%. At fitted-rate extrema, oracle hidden error remains below 0.001922 and shortcut error remains above 0.62978. This is an extrema sensitivity check, not an exhaustive hidden Monte Carlo calculation. Reference and oracle agree below 6.9e-13; the independent occupancy-continuity check agrees below 7.3e-15 over bias and rate bounds. Individual-counter controls agree exactly. Stationarity, particle-hole symmetry, short-time shot rates, long-time conservation, and the equilibrium current-noise response relation also pass. Local isolated-copy tests give oracle 7/7 and shortcut 4 passes with 3 intended hidden failures, each in under one second.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 scripts/validate_exclusion_current.py exclusion-current
```

Use `--generate` only to replace both calibration copies intentionally. The report is `jobs/exclusion-current-validation/summary.json`. The parent coordinates Harbor controls and paired Luna/high evaluation. Agent difficulty is unmeasured until those trials are reviewed.
