# Boson transfer noise, revision 1

This task tests reservoir-energy counting fluctuations in a passive quantum bosonic conductor. The completed source uses the exact frequency-dependent transmission, both thermal occupations, Bose-stimulated forward and backward rates, and the full measured frequency band. It fits nonequilibrium mean transport and equilibrium fluctuations exactly for every allowed coupling. Its hidden error is a coherent independent-transfer approximation, not an incorrect scattering calculation or omitted control.

Let t(omega)=|S_RL|² and n_i=(exp(omega/T_i)−1)^−1. The mean energy rate is the positive-frequency integral of `omega*t*(n_L−n_R)/(2*pi)`. The source treats the two stimulated-transfer rates `t*n_L*(1+n_R)` and `t*n_R*(1+n_L)` as independent events. It therefore predicts the variance-rate integrand

`omega²*t*(n_L+n_R+2*n_L*n_R)`.

The exact integrand adds

`omega²*t²*(n_L−n_R)²`.

This term vanishes at equal temperatures and never affects the mean. Thermal Bose number fluctuations and coherent mixing preserve input–output correlations that the independent-event closure discards. The shortcut is positive, reciprocal, energy conserving at the mean level, obeys equilibrium fluctuation–dissipation, and becomes correct to leading order in small transmission. The explicit quantum input states and two-measurement calorimetry in the public apparatus determine the missing term without giving its formula.

A primary derivation is [Agarwalla, Li and Wang, PRE 85, 051142 (2012)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.85.051142), with the variance formula in [arXiv:1111.6182, Eq. 81](https://arxiv.org/pdf/1111.6182). Their full-frequency convention has a factor 1/(4*pi); our explicitly positive-frequency number-conserving scattering convention uses 1/(2*pi). Our task is independently derived and checked below.

The private reference builds the Hermitian number-change matrix `A=S† P_R S−P_R` directly from two linear scattering solves. Thermal input Wick contractions give mean `sum_i A_ii*n_i` and centered variance `sum_ij A_ij*A_ji*n_i*(1+n_j)`. The initial thermal density operator commutes with reservoir energy, so this second moment agrees with the two-projective-measurement second cumulant. Adaptive frequency quadrature is independent of the oracle's fixed Gauss–Legendre integration and closed transmission expression.

A second independent validation sums exact two-mode Fock transition probabilities from a beamsplitter unitary in fixed-total-number blocks, measuring initial and final right occupations. It uses neither the Gaussian moment formula nor the transmission noise formula. Five cases including zero and unit transmission, equilibrium and unequal occupations agree to 2e-15; total-number cutoff48→64 changes moments by at most 1.21e-12. The grading reference itself has no Fock truncation.

Calibration contains 144 independent records, with both readouts and varied temperatures, bands and shifts. All uncertainties are the fixed instrument value .0005, independent of the unknown coupling and response. True g=.55; the disclosed interval is [.3,.8]. Seed735191 generates the checked-in data from the independent reference; seed735193 generates 256 noise controls. A 101-point full-range loss scan has a unique minimum at .55; noiseless recovery at five couplings, including both endpoints, is within 1.75e-8. Public and private calibration copies match.

`results/boson-transfer-noise-validation.json` records all scientific checks and noise realizations. Every calibration, parameter and oracle check passes in all256 realizations. The worst oracle hidden normalized error is .000291; the smallest shortcut hidden-group error is .4856, against the .025 tolerance. Normalization is RMS prediction error divided by RMS reference variance in each hidden group. Actual shortcut errors are .4866, .4858 and .5437. Actual local pytest gives oracle7/7 and shortcut4passes/3hidden failures.

Oracle/reference disagreement is below5.4e-15 across scored cases and supported corners; quadrature80→160 disagreement is below1.1e-14. Tests verify positive Hamiltonian frequencies (minimum .6118), nonnegative variance, scattering unitarity, transmission between0 and1, operator energy conservation, temperature-swap symmetry, zero-coupling transfer, equilibrium variance `2*T²*thermal_conductance`, and the expected fourth-order coupling size of the weak-transmission correction. Root's independent off-grid source review is in `results/root-hardening-physics-review.json`, key `boson-transfer-noise-r1`.

Reproduce with:

```
OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_boson_transfer_noise.py
```

Data regeneration is explicit through `--generate`. The completed shortcut is `scripts/boson_transfer_noise_baseline.py`. The task uses the exact approved neutral instruction, 600-second agent limit and 60-second verifier limit. Source hashes are in `results/boson-transfer-noise-source-provenance.json`. The neutral-r1 conditional screen returned three genuine physical successes out of three: initial hFPPUic, then N6QQF79 and rdacuCc. All final models include the correct transmission-squared nonequilibrium variance. N6QQF79 first used an incorrect transmission*(1-transmission) term, then corrected it using the perfect-transmission limit before completion. The initial success is included in the 3/3 aggregate. Native artifacts remain in jobs/matrix-neutral-r1-20261002-194315/index.json and its listed jobs; detailed trajectory/source reviews are in results/neutrality-brownian-trial-reviews.json. This candidate does not meet the requested failure target.
