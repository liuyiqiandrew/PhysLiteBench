# Rotational kinetic energy with an internal coupling

A neutral quantum particle moves on a fixed circular guide. Its angular coordinate is phi, and it has two internal states described in a fixed laboratory basis. Use hbar=k_B=1. Its angular momentum is L=−i*d/dphi, and its unknown positive moment of inertia `inertia` lies in [.8,1.2]. The full two-component wavefunction is periodic under phi→phi+2*pi.

The exact Hamiltonian is

    H_Delta = L^2/(2*inertia) * identity
              + Delta * [cos(tilt)*sigma_z
                + sin(tilt)*(cos(winding*phi)*sigma_x
                             + sin(winding*phi)*sigma_y)].

The sigma matrices are the usual Pauli matrices in that fixed laboratory basis. The winding is an integer, the tilt is a known angle, and Delta is a positive energy scale. The guide supplies only the stated kinematic constraint. There is no other potential, orbital gauge field, particle interaction or additional internal level. This ideal Hamiltonian defines the model.

For each preparation, a thermal reservoir at known `temperature` establishes the global canonical state of H_Delta. The reservoir can exchange angular momentum and change internal populations; no angular-momentum sector is fixed. Each measurement uses a freshly equilibrated copy. The measured observable is the mean rotational kinetic energy, the expectation of L^2/(2*inertia) times the internal identity. An ideal kinetic-energy readout measures the particle's rotational degree of freedom and sums over the internal states. Its gain and energy zero are known.

The reported output is the limit of this mean kinetic energy as Delta→infinity, taken after equilibration and at fixed inertia, temperature, winding and tilt. Each experiment is a dictionary with `temperature` in [.04,.4], `winding` in {0,1,2,3,4}, and `tilt` in [0,pi]. All controls are exact.

Calibration records in data/calibration.json contain `input`, measured energy `value`, and independent Gaussian standard uncertainty `sigma=.0005`. This is a fixed instrument uncertainty, independent of the response and unknown inertia.

Implement Model.fit(records), returning self and storing the fitted numeric attribute inertia. Model.predict(experiments) must return a finite one-dimensional NumPy array with one mean kinetic energy per input, in the same order. Any implementation satisfying this interface may be used.
