# Rotational kinetic energy with internal coupling

A neutral quantum particle moves on a fixed circular guide, with angular coordinate phi, angular momentum L=-i*d/dphi, and four internal states in a fixed laboratory basis. Set hbar=k_B=1. Its full four-component wavefunction is periodic under phi→phi+2*pi. The unknown positive moment of inertia `inertia` is in [.8,1.2].

Write the internal space as the tensor product of two two-state spaces. The matrices tau_x,tau_y,tau_z and sigma_x,sigma_y,sigma_z are the usual Pauli matrices on the first and second spaces, respectively. Let I2 and I4 denote the two- and four-dimensional identities. For integer controls q and r and a known angle theta, define

    Q0 = (I2 - tau_z)/2 tensor I2
    U(phi) = exp[-i*q*phi*(I2 tensor sigma_x)/2]
             exp[-i*r*phi*((cos(theta)*tau_z + sin(theta)*tau_x) tensor sigma_z)/2].

The displayed product order is part of the definition. The exact Hamiltonian is

    H_Delta = L^2/(2*inertia)*I4 + Delta*U(phi)*Q0*U(phi)†,

where Delta is a positive energy scale. This ideal Hamiltonian defines the model. The guide supplies only the circular constraint; there is no other potential, orbital gauge field or interaction.

For every preparation, a thermal reservoir at the known temperature establishes the global canonical state of H_Delta. It can exchange angular momentum and change all internal populations; no angular-momentum or internal sector is fixed. Each measurement uses a freshly equilibrated copy. An ideal readout measures the expectation of the rotational kinetic energy L^2/(2*inertia)*I4, summed over the internal states. Its gain and energy zero are known.

The reported output is the limit of that mean kinetic energy as Delta→infinity, taken after equilibration at fixed inertia and all other controls. Each input has `temperature` in [.04,.3], `q` in {1,2,3}, `r` in {1,2}, and `theta` in [.3,pi/2].

## Interface

`Model.fit(records)` returns `self` and stores the fitted numeric attribute `inertia`. Records in `data/calibration.json` contain `input`, the measured energy `value`, and independent Gaussian standard uncertainty `sigma=.0005`. This is a fixed instrument uncertainty, independent of response and inertia.

`Model.predict(experiments)` returns a finite NumPy array with shape `(len(experiments),)` containing the mean kinetic energies in input order.
