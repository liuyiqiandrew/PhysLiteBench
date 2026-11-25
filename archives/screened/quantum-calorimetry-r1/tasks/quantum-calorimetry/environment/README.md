# Equilibrium calorimetry of a damped quantum oscillator

Use hbar=k_B=1 and oscillator mass one. A coordinate q and its conjugate momentum p have an unknown positive natural_frequency in [.8,1.2]. The oscillator is linearly attached to a harmonic reservoir. For a finite set of reservoir modes, the complete Hamiltonian is

    H = p^2/2 + natural_frequency^2*q^2/2
        + sum_j [P_j^2/(2*m_j)
          + m_j*omega_j^2/2 * (X_j - c_j*q/(m_j*omega_j^2))^2].

All coordinates and momenta are canonical quantum operators. The square in this expression is part of the specified Hamiltonian. The reservoir spectral density is defined by

    J(omega) = (pi/2)*sum_j c_j^2/(m_j*omega_j)*delta(omega-omega_j),

and its continuum form for positive omega is

    J(omega) = damping*omega*cutoff^2/(cutoff^2+omega^2).

The known coupling control `damping` is either zero or lies in [1,3]. The known cutoff is in [1,4]. No frequency or coupling parameter depends on temperature. This quadratic Hamiltonian is the exact ideal model; there are no other interactions or unmodeled degrees of freedom.

Two matched calorimeters undergo the same infinitesimal, quasistatic temperature increase about a common positive `temperature` in [.1,1]. The sample device contains the oscillator and its attached reservoir. The blank contains the same reservoir modes with Hamiltonian sum_j[P_j^2/(2*m_j)+m_j*omega_j^2*X_j^2/2]. Each device is in its full Gibbs equilibrium throughout the heating protocol. The readout is the extra heat supplied to the sample device relative to the blank, divided by the common temperature increase. For each finite set of reservoir modes, the two heat inputs are subtracted; the reported readout is the continuum limit of these differences. All mechanical and coupling controls remain fixed during heating. There are no additional backgrounds or detector factors.

Each experiment contains `temperature`, `damping` and `cutoff`. The common `natural_frequency` is the sole unknown. Calibration uses damping=0 at varied temperatures and cutoffs. Records in `data/calibration.json` have `input`, measured `value` and independent Gaussian standard uncertainty `sigma`. The fixed instrument uncertainty is .0004 in heat-capacity units, independent of the response or fitted frequency. All controls are exact.

Implement `Model.fit(records)`, returning self and storing `natural_frequency`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order, containing the stated calorimeter heat capacities.
