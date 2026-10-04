# Qubit control, revision 9

This revision tests initial system–bath correlations after preparation in the full interacting Gibbs state. It replaces the two-wait pulse-sequence task, whose initial trial and both conditional followups all solved the physics. The complete previous task, scripts, validation, local controls, cutoff review, source peer review and all three trial reviews are preserved in `archives/qubit-control-r8`. Native jobs and their frozen source snapshots remain unchanged. The generic neutral instruction, parameter interval, .03 absolute probability-RMSE threshold and 600/60-second runtime limits are unchanged. The new apparatus deliberately changes the initial ensemble and calibration; it does not alter wording to conceal the missing physics.

## Exact state and shortcut

For spin sector r, define `s_r=w1*z1+w2*z2` and `C=sum_k a_k/omega_k`. Completing each oscillator square gives the spin-sector Gibbs probability

`p_r = exp(gamma*C*s_r^2/T) / sum_j exp(gamma*C*s_j^2/T)`.

Conditioned on r, oscillator k is a thermal oscillator displaced by `-sqrt(gamma*a_k)*s_r/omega_k`. Thus the exact initial state is the classical-quantum mixture `sum_r p_r |r><r| tensor rho_B(r)`. Its bath marginal is a mixture of product oscillator states; it is generally not a product of the separate oscillator marginals. The supplied model retains this complete bath marginal, the exact reduced spin state and all quantum conditional propagator phases. It approximates only the initial spin–bath state by the product of those two correct marginals.

A preparation rotation R maps each spin sector to `R|r>`. For two evolved spin states s,s', tracing a conditional displaced thermal oscillator gives a coherence exponent containing the usual thermal decay and forced-oscillator phase, plus

`+2 i gamma*s_r*(s-s') sum_k a_k*sin(omega_k*t)/omega_k^2`.

The oracle pairs this bath coherence with the matching rotated spin-sector coefficient before averaging over r. The completed shortcut separately averages the spin coefficient and bath coherence. Both are mathematically complete positive states evolved by the same Hamiltonian; the failure is an initial-ensemble assumption, not a missing retarded phase or incorrect dynamics.

Without a preparation pulse, the spin density remains diagonal throughout the wait. Every calibration detector probability depends only on the exact spin-sector populations, so the shortcut and oracle are identical for every allowed gamma. Individual spin marginals are maximally mixed; paired nonzero couplings and the joint detector identify gamma from the spin correlation. With identity readout, `p++=1/[2*(1+exp(-4*gamma*C*w1*w2/T))]`, strictly monotone in gamma whenever both weights are nonzero. Thus the calibration has no parameter alias. The product closure preserves bath energy through its exact bath marginal but changes the interaction energy; that is a consequence of the intended initial-state error, not a claim that the product is itself the interacting equilibrium state. The public specification explicitly permits full spin-population equilibration before isolation and specifies that the coupling and bath are not reset by local pulses.

## Independent reference

The verifier constructs each finite-Fock conditional Hamiltonian directly and diagonalizes it. Its eigenvalues determine the canonical partition weights and full conditional Gibbs matrices. It then evolves those matrices with conditional unitary propagators and traces the oscillators, rather than using the analytic displacement formula. The private cutoff is 64 states per resonator; each oscillator is traced separately inside its common spin sector. This is small because the pure-dephasing Hamiltonian has four spin sectors, not because initial correlations are discarded.

The author validator compares cutoff64 with cutoff80 on every hidden case and at extreme coupling, temperature and long waits. Cutoff48 was initially checked but had a 1.6e-8 discrepancy at a hot, strongly displaced, long-wait corner; the reference was refined before any agent evaluation. The prediction tolerance is unchanged.

## Validation and reproducibility

Calibration has 288 independent scalar records, each with fixed instrument sigma .002. Noise is independent of gamma and the noiseless response. Generation seed is 56291; the 256 noise-control seed is 56294. Calibration spans signed paired weights, positive temperatures, readout rotations and waits, with identity preparation rotations. The oracle and shortcut have exactly the same calibration map, including its nonlinear dependence on gamma.

`results/qubit-neutral-r4-validation.json` records actual-data controls, all 256 noisy fits, independent Fock convergence, full joint no-pulse stationarity, the Gibbs partition and energy, trace/positivity, zero-wait and zero-coupling limits, and noiseless parameter recovery across the entire interval. Hidden cases apply local spin rotations and read out phase quadratures, including signed coupling weights and a single rotated probe. Both controls always return valid probabilities. All 256 calibrations and parameter fits pass. The oracle passes all 256 sets with maximum hidden RMSE .001163; the shortcut fails all 256, with minimum hidden-group RMSE .10136. The actual-data shortcut errors are .24498, .20132 and .10139. Local controls give oracle 7/7 in .57 seconds and shortcut four passes plus three intended hidden failures in .62 seconds.

Run:

```
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_qubit_control.py
```

Calibration regeneration requires `--generate`. `scripts/qubit_control_baseline.py` is the completed shortcut; `solution/model.py` is the oracle. Both are exercised through the actual public and private pytest harness in the validation report. Current source hashes and the prior archive are recorded in `results/qubit-neutral-r4-source-provenance.json`; independent source review is recorded in `results/constraint-hardening-physics-review.json`.

The neutral-r4 conditional screen completed with three genuine physical successes. The initial `qubit-control__7oDze3Y` and followups `qubit-control__JZbj8vV` and `qubit-control__KU9x4Mi` all recognize the missing initial correlations and replace the separate averages by the correct joint spin-sector contraction. Each checks the quantum formula against direct finite-Fock Gibbs evolution. Final maximum hidden RMSE is .000391. All three results, including the initial success, are retained in `results/neutrality-brownian-trial-reviews.json`; this revision does not meet the difficulty target. Native jobs are `qubit-control-neutral-r4-initial-plain-20261002-190456` and `qubit-control-neutral-r4-followup-plain-20261002-191048`.
