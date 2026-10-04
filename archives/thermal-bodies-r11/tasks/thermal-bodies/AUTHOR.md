# Thermal bodies, revision 11

This scientific revision replaces the repeatedly solved gas-piston apparatus with two harmonic calorimeters connected to local reservoirs with finite memory. The complete previous task, scripts, validation, source provenance, peer review and all three successful neutral-instruction trials are preserved in `archives/thermal-bodies-r10`. Earlier sources and trials remain in `archives/thermal-bodies-neutral-v1`. No previous result is discarded or reclassified. The generic neutral instruction and 600/60-second limits are unchanged. The neutral-instruction conditional screen completed with three genuine physical successes; all three trials remain in the audit ledger.

## Physical model and exact calibration equivalence

The public Hamiltonian and six-dimensional Markov embedding specify all dynamics, reservoir labels and the heat convention. The bath forces are continuous. Reservoir power is therefore the ordinary stationary expectation `J_i=<p_i z_i>`, positive into the mechanical pair. For the full drift A and force-noise covariance D, the stationary covariance C satisfies `A C + C A^T + D=0`. The oracle solves this equation and uses `exp(A t) C` for two-time correlations.

Write the orthogonal stiffness eigenvectors as O. Because both reservoirs have the same friction gamma and memory time tau, their summed memory kernel is proportional to the identity. Each normal mode has exactly the scalar Drude deterministic response and marginal noise temperature `T_alpha=sum_i O_i,alpha^2 T_i`, even when the physical reservoirs have different temperatures. Thus every calibrated single-mode autocorrelation is exactly reproduced by a model that gives each mode independent copies of the two physical reservoirs. The copies retain their bath labels, temperatures, damping weights `gamma O_i,alpha^2`, memory and force variables. This is the completed shortcut supplied in `scripts/thermal_colored_baseline.py`; its fit is the same weighted one-parameter optimization as the oracle.

The physical reservoirs, however, generate correlations between the modes. Treating the bath copies as independent is a secular projection that preserves each mode's marginal dynamics but changes bath-resolved heat currents. The shortcut is internally consistent: it has positive stationary covariance, nonzero heat transport, conserved energy, and nonnegative reservoir entropy production. Its error is not an unfinished fit or a dropped damping, memory, temperature, coupling or reservoir index.

This differs from the previously screened heated-filament task. That task measured a linear mixture's position spectrum in a white-noise system. This apparatus has structured reservoirs and measures the work of a named physical reservoir on the mechanical pair. All marginal mode spectra remain exact; the missing correlations enter the reservoir power.

## Independent reference

The heat reference integrates the classical transmission spectrum with frequency-dependent friction `Gamma(omega)=gamma/(1-i omega tau)`. If `G=[K-omega^2 I-i omega Gamma I]^-1`, then

`J_1=(T_1-T_2)/(2 pi) integral_0^infinity 4 omega^2 Re(Gamma)^2 |G_12|^2 d omega`,

and `J_2=-J_1`. This uses a scalar frequency quadrature, independent of the oracle's stationary Lyapunov equation. Calibration references use a three-dimensional total-force modal process and its known equilibrium covariance. An additional direct velocity-spectrum quadrature checks the Fourier normalization and two-time correlations.

## Validation and provenance

`results/thermal-r11-validation.json` contains the 256 noise realizations (seed 876034). Calibration has 192 independent records with fixed absolute instrument sigma .003; generation seed is 876031. Sigma uses no hidden friction or noiseless response. The unknown is gamma=.7 in the public interval [.3,1.2]. Rich lag/memory/frequency variation identifies it; noiseless fits across the interval recover it within 3.3e-8.

Both completed controls fit all 256 calibrations and recover the parameter. All oracle predictions pass; all completed shortcuts fail. Oracle maximum hidden NRMSE is .00193; the best shortcut hidden-group error over all realizations is .603. Actual-data shortcut errors are 16.698, .670 and .607. The hidden tolerance remains .05, normalized by the RMS physical current within each nonzero-current group. Hidden preparations remain within the public range and include both signs of the temperature bias and both bath labels.

Independent current disagreement is 1.86e-14; exact calibration equivalence is within 7.1e-14. The validator checks the first law, positive stationary covariance, nonnegative reservoir entropy production, zero current at equal temperatures or zero coupling, the white-noise limit and independent modal spectral integrals. `results/thermal-r11-local-controls.json` adds 64 parameter corners, both controls' positive covariances/entropy/energy balance, and noiseless parameter recovery. Actual local harness results: oracle 7/7 in .43 seconds; completed shortcut passes four interface/calibration/parameter tests and fails only the three hidden predictions in .58 seconds.

Reproduce with:

```
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 python scripts/validate_thermal_colored.py
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/check_thermal_colored_controls.py
```

Regenerating calibration requires the explicit `--regenerate` option. `results/thermal-r11-source-provenance.json` records current hashes, exact neutral-instruction preservation and the complete prior archive. Peer review is recorded separately in `results/constraint-hardening-physics-review.json`.

## Observed neutral-instruction outcome

The initial trial `thermal-bodies__xPFF5Do` and conditional followups `thermal-bodies__mMTW6WL` and `thermal-bodies__kcL7XYD` all pass. All three public trajectories immediately identify cross-mode reservoir correlations, replace the source with the full six-dimensional stationary covariance, and calculate the local bath power correctly. Hidden maximum NRMSE is .000601. These are genuine physical successes, with no sampling or numerical exception. The revision therefore does not meet the requested difficulty target. Native jobs are `thermal-bodies-neutral-r3-initial-plain-20261002-184348` and `thermal-bodies-neutral-r3-followup-plain-20261002-184724`; detailed reviews are in `results/neutrality-brownian-trial-reviews.json`.
