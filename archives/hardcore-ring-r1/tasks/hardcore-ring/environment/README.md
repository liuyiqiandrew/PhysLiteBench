# Density dynamics of a bosonic ring

Identical bosons occupy eight sites arranged in a ring, numbered 0 through 7. Set hbar and the lattice spacing to one. The isolated single-band system is described by the exact hard-core boson Hamiltonian: each site can contain zero or one particle, with ordinary bosonic exchange between different sites. Indices in the Hamiltonian are taken modulo eight, so the bond between sites 7 and 0 is present.

    H = -J sum_{j=0}^7 [exp(i*flux/8)*b_j^dagger*b_{j+1}
                       + exp(-i*flux/8)*b_{j+1}^dagger*b_j]
        + sum_{j=0}^7 [amplitude*cos(2*pi*j/8)
                      + asymmetry*sin(2*pi*j/8)] n_j,
    n_j = b_j^dagger*b_j.

The displayed link phases define the sign of the known synthetic gauge control `flux`. The hopping J in [.8,1.2] is the only unknown and is common to all preparations. The known controls are `flux` in [-pi,pi], `amplitude` in [0,.7], `asymmetry` in [-.4,.4], and `duration` in [0,4]. These controls are constant during each evolution. There is no particle loss, dephasing or coupling to other bands.

The known integer `number` is 2, 3 or 4. Each run begins in a number-state product with one particle at every site in the corresponding list and no particles elsewhere:

    number=2: sites [3,4]
    number=3: sites [2,3,4]
    number=4: sites [2,3,4,5]

The ring evolves under H for `duration`. Site-resolved imaging then measures the mean occupation at the specified integer `site` from 0 through 7. Every observation uses a fresh copy of the stated preparation. The output is dimensionless occupation per site; all controls and preparation occupations are known exactly.

Calibration records in `data/calibration.json` contain `input`, measured `value`, and the independent Gaussian standard uncertainty `sigma=.001`, fixed by the instrument.

Implement `Model.fit(records)`, returning self and storing J as `self.hopping`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` with site occupations in input order. Any numerical implementation may be used while preserving this interface.
