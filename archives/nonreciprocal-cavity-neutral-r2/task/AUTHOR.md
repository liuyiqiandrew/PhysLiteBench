# Nonreciprocal-cavity revision 2: instrument uncertainty repair

This task tests the physical distinction between absorption from an incoming channel and thermal emission into an outgoing channel. The completed shortcut has the correct coherent scattering matrix, external thermal background, loss-rate fit, and total thermal power. It applies the reciprocal directional emission law to a statically biased circuit. That is a wrong physical closure, not a transpose implementation mistake in the supplied scattering calculation.

Write K=W W†, Gamma=loss_rate*diag(1,0.4,1.8), and

    G = [i(H-frequency*I) + (K+Gamma)/2]^-1,
    S = I-W† G W.

Propagation of the independent internal Langevin inputs gives the thermal output covariance n_body*W† G Gamma G† W. The resolvent identity G+G†=G(K+Gamma)G† makes this n_body*(I-S S†). Absorption of an incident field is instead measured by I-S† S. The two positive matrices have the same trace, but unequal diagonals in a nonreciprocal circuit. The public starter and completed shortcut use the diagonal of I-S† S for thermal emission, while correctly propagating each external occupation through S.

All calibration readouts sum the output ports. Thus the shortcut is exactly equivalent for every permitted loss rate even at nonzero flux phase. Varied frequencies and magnetic phases identify the one loss rate. A scan over the entire allowed interval has a single minimum, and noiseless fits near both endpoints and in the interior recover the input rate within 3e-9. Hidden measurements resolve individual ports under both magnetic-bias signs and with a nonzero external thermal background.

The independent reference solves the three mode amplitudes separately for each external and internal bath input, then adds their independent covariance contributions. It never uses either scattering-defect formula. It agrees with the oracle within 1.34e-15 over the checked geometry and parameter corners. Additional checks establish positive absorption/emission defects, lossless unitarity, reciprocal limits, bias reversal, and the matrix fluctuation-dissipation identity. At equal internal and external occupation n, the correct outgoing covariance is n*I; the shortcut preserves its trace but can violate the individual port balance.

The apparatus is a static, passive, number-conserving Markov resonator model. The carrier is positive and much larger than the reduced detunings, rates, and couplings. Known bath occupations refer to the measured frequency and are flat over the stated narrow band. Normally ordered photon spectra exclude vacuum offsets. This avoids adding an unstated broadband Planck spectrum or confusing net heat flow with the measured outgoing photon spectrum.

Revision 2 changes only the uncertainty model and intentionally regenerated calibration values. Previously each sigma was 0.006 times the noiseless response (subject to a floor). That metadata depended on the unknown loss rate and could reveal it independently of the noisy observations. It is now exactly 0.006 times the known `body_occupation` control, hence independent of the loss rate, frequency-dependent transfer function, and noiseless output. The three supplied uncertainty levels are 0.003, 0.009, and 0.015. Public apparatus, starter, oracle, reference, neutral instruction, parameter bounds, grading thresholds, and time limits remain byte-identical to the preceding neutral task. The same calibration seed 30021 is used intentionally, with the new instrumental uncertainty. Both public and private copies are regenerated together.

Current calibration gives loss_rate=0.270101233, relative error 0.0375%, and reduced chi²=0.901647. Noise seed 40021 produces 256 independent noisy calibrations; every fit and calibration check passes. Maximum parameter error is 0.287%, oracle hidden RMSE stays below 0.001108, and shortcut hidden RMSE stays above 0.1310. The unchanged hidden threshold is 0.025. All original independent physics, passivity, reciprocity, equilibrium, and identifiability checks pass. Local isolated controls give oracle7 passed (0.45s) and shortcut4 passed with3 intended hidden failures (0.40s).

The complete preceding neutral task, controls, validator, validation report, peer review and all seven reviewed historical trials are archived in `archives/nonreciprocal-cavity-neutral-v1`, with a SHA256 manifest. The original paired batch was0/3 plain and2/3 hinted; its failed hinted trial used the wrong array axis despite stating the correct outgoing-emission physics. The later neutral-instruction trial was0/1 plain. These are measurements of the archived data containing the uncertainty side channel; they must not be presented as outcomes of this repaired revision. No new agent outcome is claimed here. The controller will evaluate the frozen repaired data separately.

Reproduce scientific validation from the repository root:

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_nonreciprocal_cavity.py
```

Use `--generate` only for intentional regeneration of both calibration copies. Scientific results are in [the revision-2 validation report](../../results/nonreciprocal-cavity-r2-validation.json). The [independent source review](../../results/brownian-cross-review-nonreciprocal-cavity.json) records the unchanged physical-source review and 96 additional off-grid comparisons. The separate sigma-repair peer review and current local-control transcript are recorded with the revision-2 reports.

Primary context: Miller, Zhu and Fan, [“Universal modal radiation laws for all thermal emitters”](https://pmc.ncbi.nlm.nih.gov/articles/PMC5410787/), PNAS 114, 4336–4341 (2017), distinguishes the reciprocal modal radiation law from the total emission/absorption equality that remains valid for nonreciprocal objects. The task's finite-dimensional Langevin equations provide its own explicit derivation and do not require a general electromagnetic radiation solver.
