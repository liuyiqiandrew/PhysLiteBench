# Isolated oscillator calorimeter

A classical probe coordinate q and momentum p, of mass 1, are coupled to two classical harmonic internal modes (Q1,P1) and (Q2,P2), also of mass 1. Their frequencies are omega1=1.3 and omega2=2.1. The apparatus has no other energy-storage degrees of freedom. Use fixed reduced energy, length, mass, and time units with k_B=1. The probe potential is

    U(q) = quadratic*q^2/2 + quartic*q^4/4.

The two nonnegative coefficients are prescribed independently for each experiment, lie between 0 and 2.5, and are not both zero. The minimum of every potential is assigned energy zero. Neglecting the coupling energy, the total Hamiltonian is

    H = p^2/2 + U(q)
        + (P1^2 + omega1^2*Q1^2)/2
        + (P2^2 + omega2^2*Q2^2)/2.

For each shot, set the potential before preparation, then prepare the entire apparatus with the same fixed total energy `total_energy`, unknown in [0.9,3.0]. Preparation energy spread is negligible. Isolate the apparatus from external heat and work for the rest of the shot. Weak nonlinear mixing couplers exchange energy between the probe and both internal modes and do not conserve the individual mode energies. Assume their mixing is ergodic on each connected constant-total-energy surface before taking the negligible-coupling-energy limit. Take the long equilibration limit before the limit of negligible coupling energy. Use the Hamiltonian written above for equilibrium phase-space weights. All variables are classical canonical coordinates and momenta. No thermostat, feedback, or other energy reservoir is present during equilibration and measurement.

After equilibration, measure q over independent preparations. The detector reports one of the dimensionless quantities `second_moment` = <q^2>, `fourth_moment` = <q^4>, or `kurtosis` = <q^4>/<q^2>^2. These refer to the same normalized position distribution; kurtosis is the ratio of ensemble moments, not the mean of a ratio. Equilibrium symmetry makes <q>=0.

Calibration sets `quartic` to zero, varies `quadratic`, and records only the second moment. The errors in the reported moments are independent Gaussian errors with the listed standard deviations. The fitted total energy is shared by every allowed potential and observable.

Each experiment has `quadratic`, `quartic`, and `observable`. Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Expose `Model.total_energy`. Records in `data/calibration.json` contain `input`, `value`, and `sigma`. The starter supplies moment quadrature; determine the equilibrium weight for the specified apparatus.
