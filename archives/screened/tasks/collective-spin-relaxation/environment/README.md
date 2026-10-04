# Collective spin relaxation

Two distinguishable spin-1/2 centers have the fixed Hamiltonian

    H = 0.9 * (s1_x*s2_x + s1_y*s2_y + s1_z*s2_z)
        - field * (s1_z + s2_z).

Here s_i = sigma_i/2. Use reduced units with hbar=k_B=1; all energies, the field, and the bath temperature use the same fixed energy unit. Each experiment has a nonzero static `field` with 0.1 <= abs(field) <= 1.5. The exchange coefficient 0.9 is known and shared by all experiments. The single unknown is the bath `temperature` in [0.3,1.5].

Every shot begins in |up up>, where up denotes s_z=+1/2, after the static Hamiltonian has been set. The pair is then left in contact with a weak equilibrium heat bath at the common temperature. The only system operators coupled to bath fluctuations are s1_x+s2_x and s1_y+s2_y. Their spectra are nonzero at every nonzero transition frequency they connect, and the weak-coupling Markov transition rates obey thermal detailed balance with the stated Hamiltonian. The bath resolves energy differences. There is no independent local relaxation, local dephasing, field mismatch, or other coupling. Preparation and measurement are ideal. Wait long enough for every bath-connected relaxation process to reach stationarity, then measure. No transient is requested.

The detector either reports the unconditioned mean (`selection: "all"`) or accepts only the total-spin-one subspace before reporting a conditional mean (`selection: "triplet"`). The latter is an ideal projective gate immediately before measurement. It accepts |up up>, (|up down>+|down up>)/sqrt(2), and |down down>. The data report the conditional mean, with no acceptance-rate measurement. This detector setting does not change the preceding evolution.

Every input has `field`, `selection`, and `observable`. The allowed observables are:

- `magnetization`: the mean of s1_z+s2_z.
- `parallel_probability`: the probability that a subsequent joint s_z measurement has equal signs.
- `spin_correlation`: the mean of s1_x*s2_x+s1_y*s2_y+s1_z*s2_z.

All outputs are dimensionless. Calibration varies the field and records the triplet-selected magnetization. The independent measurement errors are Gaussian with the listed standard deviations. The same fitted temperature applies to every allowed detector setting and observable.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Expose `Model.temperature`. Calibration records in `data/calibration.json` contain `input`, `value`, and `sigma`. The starter includes spin operators and thermal matrix calculations; determine the stationary state for this apparatus.
