# Noise in a biased single-electron device

A spin-polarized island has only two accessible charge states, containing zero or one excess electron. It is coupled through a left and a right tunnel junction to two ideal equilibrium metallic reservoirs. All other island states are excluded. Each reservoir has a flat single-particle density of states over the relevant energy range. Transport is in the classical sequential-tunneling limit: coherent tunneling, cotunneling and lead-memory corrections are neglected. The junction tunneling amplitudes are independent of voltage.

The junctions have capacitances `C_left = fraction * C_sum` and `C_right = (1 - fraction) * C_sum`, with `C_sum = 20 aF`. There are no other capacitances or dissipative circuit elements. Independent ideal voltage sources fix the left and right electrode potentials at `voltage_left` and `voltage_right`. All electrostatic readjustment is instantaneous on the tunneling and measurement timescales. The island has no other electrical connection.

Energies are in meV, potentials in mV, and times in ns. The known `offset` is the zero-bias energy cost to add the electron, including static charging and background terms. The island's addition energy at the applied voltages is

    epsilon = offset - fraction * voltage_left - (1 - fraction) * voltage_right.

The lead chemical potentials are `-voltage_left` and `-voltage_right` in meV. Both leads have thermal energy `thermal_energy = k_B T`. The bare rates are `rate * left_factor` and `rate * right_factor`; the unknown common `rate` is in `[0.7, 1.4] ns^-1`. An empty island accepts an electron from lead l at its bare rate times the Fermi occupation of the addition energy in that lead. An occupied island loses the electron to that lead at the bare rate times one minus that occupation. These continuous-time Markov rates define the kinetic approximation for every preparation.

After stationarity is reached, an ideal amplifier in the external wire between the left voltage source and its electrode measures electrical current. Positive current means electron charge supplied by the source toward the device. The reported current is in electron charges per ns. The output is its connected stationary noise spectrum

    S(omega) = 2 integral_{-infinity}^{infinity}
                   exp(i omega t) <delta I(t) delta I(0)> dt.

`omega` is angular frequency in ns^-1. This convention gives `S = 2 r` for a Poisson stream of unit-charge impulses of rate r. Output units are electron-charge squared per ns. The ideal amplifier adds no noise or impedance. All frequencies satisfy the classical, instantaneous-electrostatics regime specified above.

Each experiment contains:

- `offset` in `[-0.4, 0.4]` meV;
- `voltage_left`, `voltage_right` in `[-0.8, 0.8]` mV;
- `thermal_energy` in `[0.15, 0.4]` meV;
- `fraction` in `[0.15, 0.85]`;
- `left_factor`, `right_factor` in `[0.5, 1.6]`;
- `omega` in `[0, 10]` ns^-1.

The same `rate` applies to every preparation. `data/calibration.json` contains independent measurements with `input`, `value`, and fixed known `sigma = 0.002` in the output units.

Implement `Model.fit(records)` and `Model.predict(experiments)`. `fit` returns `self` and stores the inferred `self.rate`. `predict` returns a finite NumPy array with shape `(len(experiments),)`. You may use any numerical method consistent with the apparatus.
