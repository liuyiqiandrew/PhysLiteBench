# Gas snapshot energy

Consider an infinite homogeneous classical ideal gas of identical, noninteracting massive particles in three spatial dimensions. Particle number is conserved, and quantum statistics, creation and annihilation are absent. These assumptions define the ideal model throughout the stated range.

In its rest frame the gas has uniform spatial density and the canonical momentum probability measure

    Z(T,m)^(-1) exp[-sqrt(m^2 + |p_star|^2)/T] d^3 p_star.

Positions on a simultaneous rest-frame slice are independent of momenta. The temperature T is the known temperature in that rest frame. The common particle rest mass m is unknown, with `0.8 <= mass <= 1.2`. Use units c=k_B=1, with energy, temperature, momentum and mass expressed in a fixed known energy unit E0. Velocities are in units of c. Total particle energy includes rest energy.

The gas rest frame moves with signed velocity beta along the laboratory +x axis. There are no external fields or further thermalization in the laboratory. Take the thermodynamic interior limit, so no walls or boundary transients enter the observation.

A distributed ideal detector records every particle within a fixed laboratory spatial volume at a single simultaneous laboratory time. Detection efficiency is uniform. Each recorded particle contributes once with equal weight. Repeated snapshots determine the mean per recorded particle. The detector records the laboratory four-momentum `(E_lab, p_x_lab, p_y_lab, p_z_lab)`.

The recorded energy is then expressed in an analysis frame moving at signed velocity a along laboratory +x. The reported output is the number-weighted mean of

    E_analysis = (E_lab - a*p_x_lab)/sqrt(1-a^2)

for the particles in that laboratory snapshot, in units E0. The analysis frame changes only how the recorded four-momenta are expressed; the detector selection and simultaneity remain those of the laboratory.

Each experiment specifies:

- `temperature`: rest-frame T, from 0.2 to 1;
- `gas_speed`: beta, from -0.9 to 0.9;
- `analysis_speed`: a, from -0.9 to 0.9.

`data/calibration.json` contains independent records `{"input": {...}, "value": ..., "sigma": 0.004}`. Gaussian measurement errors have the fixed instrument standard deviation 0.004 E0. Controls are exact.

Implement `Model.fit(records)`, returning `self` and storing the fitted common mass in `self.mass`. Implement `Model.predict(experiments)`, returning a finite one-dimensional NumPy array of mean energies in input order. Keep this interface and the stated units.
