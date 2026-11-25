# Equilibrium measurements of a coupled resonator network

Three ideal quantum harmonic resonators have fixed laboratory quadratures q_j,p_j, with [q_j,p_k]=i delta_jk and all other same-kind commutators zero. Planck's constant and Boltzmann's constant are one. These dimensionless quadratures retain the same calibration for every setting. The exact static Hamiltonian is

    H = energy_scale/2 * [q^T (A+pairing*B) q + p^T (A-pairing*B) p],

where

    A = [[ 1.8,  .21, -.13],
         [ .21, 2.2,   .17],
         [-.13,  .17, 2.6 ]]

    B = [[1.2,  .10,  .18],
         [.10, 1.6,  -.14],
         [.18, -.14, 1.7 ]].

The matrices and the control `pairing` are known. Both quadratic forms are positive throughout the stated range. This is a time-independent passive quadratic network; there is no periodic pump, rotating-frame approximation, nonlinear term, or omitted mode. Equivalently, the fixed local operators a_j=(q_j+i*p_j)/sqrt(2) are bosonic resonator operators. Additive constants in H do not affect the measurements.

Each preparation reaches the full Gibbs state exp(-H/T)/Tr(exp(-H/T)) through a weak thermal contact that is negligible during measurement. A zero-temperature preparation means the ground state of the same interacting Hamiltonian. The common unknown `energy_scale` lies in [.8,1.3].

Two ideal measurements are available:

- `frequency`: weak linear spectroscopy measures one of the three positive angular resonance frequencies of this Hamiltonian, sorted from smallest to largest. `index` selects that frequency, starting at zero.
- `thermal_excess`: calibrated local quadrature measurements in separate equilibrium preparations at T and at zero temperature return ([<q_j^2>_T+<p_j^2>_T]-[<q_j^2>_0+<p_j^2>_0])/2. `index` selects local resonator j, starting at zero. The couplings and quadrature calibration are identical in the two preparations. Both quadrature variances are measured while the Hamiltonian is set to the specified controls.

Quadrature variances are ensemble estimates from separate identically prepared copies. There is no unknown efficiency, offset or gain. The thermal-excess output is dimensionless; the frequency output has the angular-frequency units defined by the Hamiltonian.

## Data and interface

Every experiment is a dictionary with `readout` equal to `frequency` or `thermal_excess`, integer `index` in {0,1,2}, `pairing` in [0,1], and `temperature` equal to zero or in [.15,.8]. Temperature does not change the linear resonance frequencies of this exact quadratic model.

Calibration records in `data/calibration.json` give an `input` experiment, measured `value`, and independent Gaussian standard uncertainty `sigma`. Calibration uses the frequency measurement with fixed instrumental uncertainty .001 in angular-frequency units.

Implement `Model` in `/app/model.py`. `fit(records)` must return self and store the fitted numeric attribute `energy_scale`. `predict(experiments)` must return a finite one-dimensional NumPy array with one output in input order. Implement any method consistent with the apparatus and preserve this API. The visible checks run with `python -m pytest -q /app/test_public.py`.
