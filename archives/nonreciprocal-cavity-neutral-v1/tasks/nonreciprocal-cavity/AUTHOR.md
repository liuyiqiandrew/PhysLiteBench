# Nonreciprocal-cavity revision 1

This task tests the physical distinction between absorption from an incoming channel and thermal emission into an outgoing channel. The completed shortcut has the correct coherent scattering matrix, external thermal background, loss-rate fit, and total thermal power. It applies the reciprocal directional emission law to a statically biased circuit. That is a wrong physical closure, not a transpose implementation mistake in the supplied scattering calculation.

Write K=W W†, Gamma=loss_rate*diag(1,0.4,1.8), and

    G = [i(H-frequency*I) + (K+Gamma)/2]^-1,
    S = I-W† G W.

Propagation of the independent internal Langevin inputs gives the thermal output covariance n_body*W† G Gamma G† W. The resolvent identity G+G†=G(K+Gamma)G† makes this n_body*(I-S S†). Absorption of an incident field is instead measured by I-S† S. The two positive matrices have the same trace, but unequal diagonals in a nonreciprocal circuit. The public starter and completed shortcut use the diagonal of I-S† S for thermal emission, while correctly propagating each external occupation through S.

All calibration readouts sum the output ports. Thus the shortcut is exactly equivalent for every permitted loss rate even at nonzero flux phase. Varied frequencies and magnetic phases identify the one loss rate. A scan over the entire allowed interval has a single minimum, and noiseless fits near both endpoints and in the interior recover the input rate within 3e-9. Hidden measurements resolve individual ports under both magnetic-bias signs and with a nonzero external thermal background.

The independent reference solves the three mode amplitudes separately for each external and internal bath input, then adds their independent covariance contributions. It never uses either scattering-defect formula. It agrees with the oracle within 1.34e-15 over the checked geometry and parameter corners. Additional checks establish positive absorption/emission defects, lossless unitarity, reciprocal limits, bias reversal, and the matrix fluctuation-dissipation identity. At equal internal and external occupation n, the correct outgoing covariance is n*I; the shortcut preserves its trace but can violate the individual port balance.

The apparatus is a static, passive, number-conserving Markov resonator model. The carrier is positive and much larger than the reduced detunings, rates, and couplings. Known bath occupations refer to the measured frequency and are flat over the stated narrow band. Normally ordered photon spectra exclude vacuum offsets. This avoids adding an unstated broadband Planck spectrum or confusing net heat flow with the measured outgoing photon spectrum.

Calibration seed 30021 gives loss_rate=0.270042799, relative error 0.0159%, and reduced chi-squared 0.902314. The private true loss rate is 0.27. Measurement standard deviations are 0.006 times the larger of the noiseless total output and 0.1. With noise seed 40021, all 256 independent calibrations pass; maximum parameter error is 0.209% and maximum reduced chi-squared is 1.3661. Oracle hidden absolute RMSE is at most 6.13e-5 for the nominal data and 8.06e-4 across all noise realizations. Shortcut errors are 0.1311, 0.1937, and 0.1595; the smallest over every noise realization is 0.1310. The hidden threshold is 0.025 in normalized photon spectral-density units, far above fitting uncertainty and numerical error.

Local isolated controls: oracle 7 tests passed in 0.29 seconds; shortcut 4 tests passed and all 3 intended hidden groups failed in 0.30 seconds. The frozen gpt-5.6-luna high trials scored 0/3 unhinted and 2/3 hinted. All six submissions, public trajectories, and verifier metrics were reviewed. The three unhinted submissions fitted the loss scale accurately but accepted the supplied reciprocal directional-emission closure. In-memory diagnostics changed only that emission closure at each submitted loss rate; all hidden groups then passed with maximum error 6.17e-5, while calibration changed by less than 9e-16. Retained trial artifacts were not edited.

Two hinted submissions explicitly propagated the internal Langevin inputs and passed. The remaining hinted trial, `nonreciprocal-cavity__znb48ey`, correctly stated the outgoing defect I-S S† but misread the axis of the batched scattering array. Its self-check repeated the same incorrect axis operation and labeled it a row sum. This is an implementation failure after identifying the correct physics, and is not counted as an additional physical-model failure. Correcting just that axis at its fixed fitted loss rate also passes all hidden groups.

The six reviewed classifications, public statements, source hashes, and fixed-loss diagnostics are in [the trial review report](../../results/hardening-materials-trial-reviews.json), under `nonreciprocal-cavity__ABS23B2`, `nonreciprocal-cavity__PAJJGkh`, `nonreciprocal-cavity__xujzLdy`, `nonreciprocal-cavity__NQVXbUM`, `nonreciprocal-cavity__oUJEEgt`, and `nonreciprocal-cavity__znb48ey`. These are the observed outcomes of the frozen paired batch; they do not guarantee failure on future trials.

Reproduce scientific validation from the repository root:

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_nonreciprocal_cavity.py
```

Use `--generate` only for intentional regeneration of both calibration copies. Scientific results and local control transcripts are in [the validation report](../../results/nonreciprocal-cavity-validation.json). The [independent source review](../../results/brownian-cross-review-nonreciprocal-cavity.json) records the reviewed hashes and 96 additional off-grid comparisons.

Primary context: Miller, Zhu and Fan, [“Universal modal radiation laws for all thermal emitters”](https://pmc.ncbi.nlm.nih.gov/articles/PMC5410787/), PNAS 114, 4336–4341 (2017), distinguishes the reciprocal modal radiation law from the total emission/absorption equality that remains valid for nonreciprocal objects. The task's finite-dimensional Langevin equations provide its own explicit derivation and do not require a general electromagnetic radiation solver.
