# Expansion of lattice bosons

Three identical bosons occupy an open chain of eight sites, numbered 0 through 7. Set the lattice spacing and hbar to one. Site j has coordinate x_j=j−3.5. While confined to the chain, the particles obey the exact single-band hard-core boson model: each site can contain zero or one particle, with ordinary bosonic exchange between different sites. There is no coupling between sites 0 and 7.

For each preparation, the Hamiltonian is

    H = -J sum_{j=0}^6 (b_j^dagger b_{j+1} + b_{j+1}^dagger b_j)
        + sum_{j=0}^7 (trap*x_j^2 + tilt*x_j) n_j,
    n_j = b_j^dagger b_j.

The hopping J in [.8,1.2] is the only unknown and is common to all preparations. The known controls are `trap` in [0,.15], `tilt` in [-.3,.3], and `duration` in [0,2]. At time zero, sites 2, 3 and 4 each contain one particle and all other sites are empty, with no preparation uncertainty. The chain then evolves in isolation under H for the specified duration. There is no loss, dephasing or coupling to other bands.

Each measurement uses a fresh preparation. Two readouts are available:

* `observable="density"`: site-resolved imaging measures the mean occupation of the specified integer `site` from 0 through 7 at the end of the evolution.
* `observable="momentum"`: at the end of the evolution, the lattice, confining potential and particle interactions are switched off suddenly without changing the many-particle state. The particles then expand freely without collisions. Far-field imaging records the momentum distribution at the known dimensionless `wave_number` in [-pi,pi]. The common single-site orbital envelope is independently calibrated and divided out. The reported distribution is normalized per lattice site: its average over one interval of length 2*pi is 3/8. The release introduces no additional site-dependent phase.

Calibration records in `data/calibration.json` contain `input`, measured `value`, and the independent Gaussian standard uncertainty `sigma=.001`, fixed by the instrument. All control values and detector normalizations are known exactly.

Implement `Model.fit(records)`, returning self and storing J as `self.hopping`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order. Any numerical implementation may be used while preserving this interface.
