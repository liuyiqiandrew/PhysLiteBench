# Triad photon phase, revision 1

This task tests a many-particle phase that cannot be recovered from pairwise distinguishability alone. Calibration uses two-photon Hong-Ou-Mandel coincidences. The starter already sums all bosonic path pairs, includes partial distinguishability, and correctly normalizes repeated output occupations. It replaces polarization inner products by their magnitudes before that sum. Every two-photon probability is unchanged, but a three-photon permutation cycle retains an invariant phase.

For input Jones states z_j, let G_ij=<z_i|z_j>. With output slots o_k and one photon per spatial input, the probability is

    P(counts) = sum_(sigma,pi) A_sigma * conjugate(A_pi)
                * product_k G_(pi(k),sigma(k)) / product_j counts_j!,
    A_sigma = product_k U_(o_k,sigma(k)).

For two photons the nontrivial term contains |G_12|^2. With the stated linear polarizations, the coincidence probability is 0.5*sin^2(alpha*(length_2-length_1)). Calibration lengths are at most 1 cm and alpha is below 1.4 rad/cm, so alpha*length remains below pi/2 and the calibrated probability increases strictly with positive alpha. The fitted gain is identifiable.

For three photons the cycle G_12*G_23*G_31 also matters. Individual phase choices z_j -> exp(i*phi_j) z_j cancel around that cycle. At polarization angles 0, pi/3, 2*pi/3 the overlaps have magnitudes 1/2 but their product is -1/8. The magnitude-only closure changes it to +1/8. In a balanced Fourier tritter, the one-photon-per-output probability changes from 1/12 to 7/36, a difference of 1/9. This collective phase is discussed experimentally in [Menssen et al., Distinguishability and many-particle interference](https://arxiv.org/abs/1609.09804).

The completed shortcut remains a mathematically valid counting model. For at most three normalized states, entrywise overlap magnitudes form a positive semidefinite Gram matrix: its three-state determinant replaces the real cycle product by its modulus and cannot decrease. It therefore describes another set of internal states with the same pairwise visibilities. For the hidden trine-like states, this alternative Gram matrix has rank three, whereas the publicly specified internal polarization space has dimension two. Both the incorrect cycle and this rank mismatch are consequences of the physical closure. Probabilities remain nonnegative and normalized; a broken probability formula is not the intended failure.

The independent reference expands the input creation-operator product in polarization-resolved output Fock states. Each creation step carries its bosonic square-root occupation factor. It then sums squared amplitudes over all unresolved polarization occupations for each spatial count pattern. This construction does not use pair permutations or overlap matrices. The validator also factors the shortcut Gram matrix into alternative internal states and verifies its probabilities with this separate Fock calculation.

Calibration uses 200 measurements, true gain 1 rad/cm, seed 9327, and independent Gaussian errors with standard deviation 0.6% of the largest calibration probability. The fitted gain is 1.00000402, reduced chi-square 0.86939. Both controls match noiseless calibration below 1.2e-16. Hidden errors are RMS-normalized within each output-occupation group, with a 4% limit. This allows errors well above calibration and numerical uncertainty at the stated nonzero output probabilities. Oracle error is at most 2.1e-6; shortcut errors are 1.24836, 0.21266, and 0.31209.

All 256 added calibration-noise realizations pass calibration and parameter checks, with maximum gain error 0.1753%. At fitted-gain extrema, oracle hidden error stays below 0.000904 and shortcut error above 0.21265. This is an extrema sensitivity check, not exhaustive hidden Monte Carlo evaluation. Independent Fock probabilities agree below 5.6e-17. Random individual Jones-vector rephasings change probabilities by less than 3.4e-16; all output patterns normalize within 1.2e-15. The magnitude Gram matrices remain positive semidefinite within rounding, and their alternative-state Fock probabilities agree below 8.9e-16. Pairwise closure agreement, fully distinguishable multinomial statistics, identical-photon tritter suppression, and the trine phase example also pass.

Local isolated-copy tests give oracle 7/7 and shortcut 4 passes with 3 hidden failures, each in under half a second. The parent coordinates Harbor controls and paired Luna/high trials. Scientific validation does not establish the desired agent failure rate.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 scripts/validate_triad_photon_phase.py triad-photon-phase
```

Use `--generate` only when intentionally replacing both calibration copies. The report is `jobs/triad-photon-phase-validation/summary.json`.
