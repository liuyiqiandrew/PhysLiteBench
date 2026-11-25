# Coupled oscillator equilibrium, revision 1

Calibration measures the probe's static mean displacement in the canonical state of two coupled quantum oscillators. It identifies the bare probe spring even at finite coupling. The starter statically eliminates the auxiliary coordinate and applies an isolated quantum-oscillator Gibbs law to the resulting spring. This uses every supplied material parameter and exactly reproduces calibration. The hidden observable instead requires the probe's reduced state of the whole interacting thermal pair.

For potential matrix K=[[k,g],[g,2.56]], displacement of both coordinates removes the static force. The probe mean is

    <q> = F*(K^-1)_qq = F/(k-g^2/2.56).

This holds quantum mechanically and classically at every temperature. Centered fluctuations are unchanged by the force. The allowed parameters give det K >= .358, so all preparations are stable. Static calibration with known forces and couplings identifies k uniquely.

The oracle uses an orthogonal normal-mode transformation K=O diag(omega_a^2) O^T, applied to positions and momenta together. The probe variances are

    Var(q) = sum_a O_(q,a)^2 * coth(omega_a/(2*T))/(2*omega_a),
    Var(p) = sum_a O_(q,a)^2 * omega_a*coth(omega_a/(2*T))/2.

Tracing out the second oscillator retains both mode contributions. The shortcut instead sets omega_eff=sqrt(k-g^2/2.56) and uses the one-oscillator variances at temperature T. Its static susceptibility is correct, and its high-temperature classical limit is correct, but its finite-temperature quantum reduced state is wrong. A reduced equilibrium state at finite coupling need not be the Gibbs state of the static effective unit-mass oscillator; see [Yeo and Shim, Structure of Quantum Mean Force Gibbs States for Coupled Harmonic Systems](https://arxiv.org/abs/2412.02074).

The verifier independently computes poles and residues of the coordinate susceptibility

    chi_qq(z) = (2.56-z^2)/det(K-z^2 I).

It combines the two thermal pole contributions without diagonalizing K. A further author-side reference constructs the full interacting Hamiltonian in the product of bare oscillator number bases, forms its canonical state, and evaluates exact matrix elements of q, q^2, and p^2 before tracing out the other mode. Increasing the cutoff from 26 to 36 levels per oscillator reduces the maximum error over three checks from 8.18e-7 to 1.17e-9. These checks include temperatures 0.08, 0.3, and 0.5 with both coupling signs and displaced states. The initial 26-level cutoff did not meet the chosen 2e-7 validation target at the warmest point; increasing only this independent diagnostic basis resolved it. The task oracle and grading are analytic and unaffected.

Additional checks cover the zero-coupling limit, classical high-temperature limit, allowed stability corners, uncertainty product Var(q)*Var(p)>=1/4, and the equality of the full and statically eliminated zero-frequency susceptibility. At near-zero temperature and coupling 1.3, the reduced probe purity is 0.8765; the global ground state is pure while the probe remains mixed. This is a useful independent diagnostic of why a single effective ground-state oscillator cannot describe the probe.

Calibration contains 200 measurements, true spring 1.05, seed 9328, and independent Gaussian noise with standard deviation 0.6% of the largest mean displacement. Both controls recover 1.04960292 with reduced chi-square 0.97349 and match exact calibration below 1.2e-16. Hidden group errors are RMS-normalized, with a 4% limit. Oracle error is at most 0.000529; shortcut errors are 0.05875, 0.17170, and 0.14627.

All 256 additional calibration-noise trials pass, with maximum spring error 0.1747%. At fitted-spring extrema, oracle hidden error remains below 0.002449 and shortcut error above 0.05590. This is an extrema sensitivity check, not exhaustive hidden Monte Carlo evaluation. Susceptibility-residue and normal-mode covariances agree below 3.4e-16 on hidden cases; allowed-corner checks also pass. The high-temperature relative error is below 1.3e-9 at T=10^4. Local isolated-copy tests give oracle 7/7 and shortcut 4 passes with 3 hidden failures.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 scripts/validate_coupled_oscillator_equilibrium.py coupled-oscillator-equilibrium
```

Use `--generate` only to replace both calibration copies intentionally. The report is `jobs/coupled-oscillator-equilibrium-validation/summary.json`. The parent coordinates Harbor controls and paired Luna/high trials. Scientific validation alone does not establish the desired failure rate.
