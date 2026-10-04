# Quantum residence time

The detector measures stationary population inside a finite region divided by incident particle flux. This is a density measurement, not an arrival-time or transmitted-subensemble convention. The leads are identical, and the scattering phases are referenced to the two physical region faces.

The completed starter computes the full flux-normalized two-port scattering matrix and its energy derivative. Its incident-column Wigner–Smith delay is

    Q_11 = -i [r* dr/dE + t* dt/dE]

in hbar=m=1 units. Both reflection and transmission are included, with correct derivatives of the lead wave number and interior transfer matrix. This is a mathematically valid scattering delay. The physical approximation is to identify it with the region's population-to-flux ratio.

Differentiating the stationary Schrodinger equation with respect to energy and integrating its Wronskian over the detector region gives

    tau_population = Q_11 + Im(r)/(2E).

The additional term is the entrance contribution from interference between incident and reflected waves. Its sign follows the public conventions psi_left=exp(ikx)+r exp(-ikx) and psi_right=t exp(ik(x-L)). When the potential is zero, r=0 and both predictions are exactly L/sqrt(2E) for every allowed L. Free-region population measurements therefore identify width but cannot expose the observable mismatch.

The oracle obtains the exact transfer-matrix derivative with a matrix-exponential Frechet derivative and applies the boundary identity. The independent verifier solves a four-variable interface-matching system for reflected, transmitted and interior wave amplitudes, then integrates the interior density and divides by the incoming speed. Its cosine/sine-over-wave-number basis remains regular at E=V. It contains no scattering-delay formula.

A second independent check adds a weak imaginary potential -i eta in the region and measures missing outgoing probability flux. Extrapolating [1-|r|²-|t|²]/(2 eta) to zero absorption reproduces the density measurement. Probability-current conservation inside the region, scattering unitarity, Wigner–Smith Hermiticity and an independent finite-difference scattering derivative also pass. Thus the shortcut's failure is not derivative accuracy, flux normalization, or a missing reflected amplitude.

Calibration has 144 free-region records with fixed instrument sigma=0.002, independent of width and noiseless response. Seed 945061 generates the frozen data; seed 945063 generates 256 independent noise checks. Both controls fit width 0.899637236 and reduced chi-square 0.989993. All 256 calibration, parameter and oracle checks pass; no completed shortcut passes. The largest oracle hidden error is 0.001287 and the smallest shortcut group error is 0.10695, against the 0.03 cutoff. Hidden groups cover evanescent interiors, the regular E=V limit and propagating interiors. The shortcut's predictions remain positive throughout the validated domain corners.

`results/quantum-residence-validation.json` records interior-density agreement 2.45e-15, density-quadrature refinement 2.89e-15, scattering-derivative agreement 8.8e-11 and weak-absorption extrapolation error 1.43e-8. `results/quantum-residence-local-controls.json` records oracle 7/7 and shortcut 4 passed/3 hidden failures. The teammate review is recorded under `quantum-residence-r1` in `results/materials-hardening-physics-review.json`.

The task uses the exact neutral instruction, pinned environment and 600/60-second limits. Public files define the physical detector, phase references, boundary conditions, preparation and units without naming the missing interference term. No agent evaluation has been claimed before the task's frozen run. Reproduce with `python scripts/validate_quantum_residence.py`; use `--generate` only for deliberate regeneration of both calibration copies.
