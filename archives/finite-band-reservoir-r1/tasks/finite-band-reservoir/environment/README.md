# Orbital occupation after contact

A spinless fermionic orbital is coupled to the endpoint of a one-dimensional fermion chain. There are no particle interactions, disorder, additional reservoirs, measurements during evolution, or relaxation terms. The following quadratic Hamiltonian is the exact model. Units set hbar, Boltzmann's constant and the chain hopping energy to one.

For a chain of N sites, the Hamiltonian after contact is

    H_N = orbital_energy*d†d
          - sum_(j=1)^(N-1) (c_j† c_(j+1) + c_(j+1)† c_j)
          + V*(d† c_1 + c_1† d),
    V = coupling_scale * contact_multiplier.

The operators d and c_j obey the usual fermionic anticommutation relations. All coefficients are real. The remote end of the chain is open.

Before time zero the contact is absent. The orbital is occupied with certainty. Independently, the chain is in its grand canonical equilibrium state at the specified `temperature` and chemical potential zero, for its uncoupled hopping Hamiltonian. The initial total state is the product of these two preparations. At time zero the contact is switched to V instantaneously and remains fixed. The complete orbital-plus-chain system subsequently evolves unitarily under H_N.

The readout is the mean orbital occupation averaged over a long observation window. The chain thermodynamic limit is taken first:

    output = limit_(t -> infinity) (1/t)*integral_0^t
                 [limit_(N -> infinity) <d†(s)d(s)>_N] ds.

This is an ensemble expectation followed by a time average. It is a number between zero and one. The switch and the readout do not change the orbital energy.

Each input has `orbital_energy` in `[0,.8]`, `temperature` in `[.2,.8]`, and `contact_multiplier` equal to one of the known hardware settings `{1,3,4}`. The only unknown is the positive `coupling_scale` in `[.65,.95]`, common to all experiments.

`data/calibration.json` contains independent measurements with fields `input`, `value`, and `sigma`. The uncertainty is the fixed instrument value `sigma=.0005`, independent of the unknown parameter and the measurement value.

Implement `Model.fit(records)` and `Model.predict(experiments)`. `fit` must return `self` and expose the fitted parameter as `self.coupling_scale`. `predict` must return a finite one-dimensional NumPy array in input order. Any numerical implementation may be used while preserving this interface.
