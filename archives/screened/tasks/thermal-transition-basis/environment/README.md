# Relaxation of interacting spins

Two distinguishable spin-1/2 systems have the static Hamiltonian

    H = 0.5*sigma_z1 + 0.85*sigma_z2 + coupling*sigma_x1*sigma_x2.

The sigma operators are Pauli matrices, with sigma_z=+1 defining an excited
bare spin. Use reduced units with hbar=k_B=1. The prescribed coupling is
between 0 and 0.7. Both spins contact independent equilibrium heat reservoirs
at the same known temperature T=0.65. Reservoir 1 couples only to sigma_x1 and
reservoir 2 only to sigma_x2. Their fluctuations have zero cross-correlation.

The reservoirs have identical spectral response. For an energy w lost by the
spin system, the transition-rate spectral factor is

    G(w) = rate*w/(1-exp(-w/T)),  G(0)=rate*T.

The unknown common rate is between 0.004 and 0.016. Squared matrix elements of
the coupled system operator multiply this spectral factor. Use the weak-coupling,
Markov, energy-resolved secular limit for the stated static Hamiltonian. All
relaxation rates are small compared with distinct transition-frequency gaps.
Neglect reservoir-induced Hamiltonian shifts. There are no other noise channels,
external drives, measurements, or feedback during relaxation.

For each fresh run, set coupling first, then prepare the exact energy eigenstate
whose index is initial_level, with eigenvalues ordered from lowest 0 to highest 3.
All four energies are nondegenerate throughout the allowed coupling range.
The reservoirs are already stationary when they are connected at time zero.
For protocol transient, measure after time between 0 and 400. For stationary,
wait until all relaxation has completed; the time field is then ignored.

The observable is excitation_1 or excitation_2, the probability of sigma_zi=+1,
or xx_correlation, the expectation of sigma_x1*sigma_x2. Each input contains
coupling, initial_level, protocol, time, and observable. Outputs are dimensionless.

Implement Model.fit(records), returning self and setting rate, and
Model.predict(experiments), returning a finite NumPy array of shape
(len(experiments),). You may edit the entire implementation. The same rate applies
to all preparations. Calibration measures excited-spin probabilities at zero
coupling. Records in data/calibration.json contain input, value, and sigma,
the independent Gaussian measurement standard deviation. Settings are exact.
Run python -m pytest -q test_public.py.
