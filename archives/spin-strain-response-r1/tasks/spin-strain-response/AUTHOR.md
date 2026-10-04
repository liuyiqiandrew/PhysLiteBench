# Spin-strain response, revision 1

The apparatus measures an isothermal mechanical response of one spin-1 molecule. The full Hamiltonian, Gibbs preparation and displacement-dependent energy determine the force. The source diagonalizes the complete Hamiltonian and computes the exact equilibrium quadrupole variance, then substitutes the classical static fluctuation relation for the measured force slope.

For Q=Sz², the molecular force is lambda*<Q>. In the energy basis, the correct slope divided by lambda² is

    [sum_i p_i Q_ii² - (sum_i p_i Q_ii)²]/T
    + 2 sum_{i<j} (p_i-p_j)/(E_j-E_i) |Q_ij|².

The source instead uses Var(Q)/T, with both the Gibbs state and the variance evaluated exactly. It is a positive, mathematically complete fluctuation approximation. It uses every control and the full operator; it has no missing level, malformed covariance or derivative implementation. It coincides with the physical response when Q commutes with H and in the classical high-temperature limit. Applying it at finite transverse field assigns all equal-time quantum fluctuations to static thermal response. The distinction between variance and static susceptibility is discussed by [Frérot and Roscilde, Physical Review B 94, 075121 (2016)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.94.075121).

The public readout is the slope of molecular force on a controlled actuator, not the total elastic stiffness of the transducer. Holding a displacement and allowing full reservoir equilibration selects the isothermal response. The empty apparatus is subtracted; no unknown background elastic term exists. Positive lambda removes the sign ambiguity in fitting lambda². Public information gives the Hamiltonian and measurement protocol but no fluctuation-response identity or required solution method.

## Calibration and independent validation

There are 216 calibration readings, all at transverse field zero, varying anisotropy, longitudinal field and temperature. The source and oracle are exactly equal on this subset. Response is lambda² times a known positive coefficient, so the positive coupling is globally identifiable throughout [0.8,1.4]. Both completed models use the same bounded weighted least-squares fit. The calibration seed is 995021 and the 256-noise seed is 995027. Every reading has fixed instrument sigma=0.001, independent of its noiseless response. Public and private calibration copies are identical.

The independent verifier differentiates the entire Gibbs matrix exponential with a Fréchet derivative, including partition-function normalization. It does not use energy-basis divided differences. A third check directly compares force at small positive and negative displacements and extrapolates a fourth-order derivative. Across 243 parameter/domain combinations, the Fréchet reference agrees within 2.34e-15, and direct displacement within 4.61e-13. Halving the displacement changes the result by at most 1.03e-12. Checks cover exact commuting Gibbs response, zero-field degeneracies, energy-shift invariance, field reversal, positive susceptibility, the variance bound, the classical high-temperature limit and the ground-state virtual-transition limit. No numerical truncation or finite-size approximation is used.

The fitted coupling is 1.099930986965864 versus true 1.1, with reduced chi-square 1.0371791870. Oracle hidden normalized errors are 0.0001255. Shortcut errors are 1.5602, 2.5106 and 1.2922; commuting anchors pass. Across all 256 noise draws, both models pass calibration and parameter recovery; every oracle passes and every completed shortcut fails the three noncommuting groups. Maximum oracle error is 0.000377, while minimum shortcut error is 1.29159. Scored physical signals range from 0.1841 to 0.4973. The 4% group RMS tolerance is far above calibration uncertainty and numerical error and far below the physical discrepancy.

Local isolated pytest controls give oracle 7/7 and shortcut 4 passes plus three intended physical-prediction failures. Docker controls and model evaluation are root-coordinated and pending. No agent difficulty or future failure rate is established by these scientific controls.

## Relation to previous candidates

This task shares quantum noncommutativity with the archived work-cumulant tasks, but its measurement is an equilibrium static mechanical response rather than a distribution of sequential energy records. It is not the retained superconducting heat task's frozen spectral coherence factor. Its approximation is the classical equal-time fluctuation relation applied to an otherwise exact quantum Gibbs calculation. This is a related theoretical theme, not a claim of independence between all physical principles. The three-dimensional solver also makes a correct physical repair inexpensive, so scientific separation alone does not establish difficulty for Luna.

Trial reviews must distinguish retaining the variance-response assumption from a failed implementation of a correct Gibbs derivative or spectral quotient. A causal physical repair should hold the submitted fitted coupling fixed and change only the response law. Preserve correct-physics exploratory calculations and implementation-induced regressions when classifying outcomes.

## Reproduction

From the repository root after promotion, or from this staged root:

    uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_spin_strain_response.py

Ordinary validation reads frozen data. Only `--generate` regenerates both calibration copies. The neutral instruction is unchanged, SHA256 6a5ef18a542d5fd42298ccab151355ecf224974f3b465d9a6299768988739b62. Reports are in `results/spin-strain-response-validation.json`, `results/spin-strain-response-r1-local-controls.json` and `results/spin-strain-response-r1-source-provenance.json` relative to the staged/repository root. The prototype source and report are retained under `prototype/` in the staged bundle. Final physics peer review is recorded separately in `results/physics-review-spin-strain-response-r1.json`.
