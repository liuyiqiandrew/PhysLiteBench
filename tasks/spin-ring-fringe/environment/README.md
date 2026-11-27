# Spin ring fringe

A probe spin controls the drive on a periodic ring of N spin-1/2 sites. Initially
the ring is in thermal equilibrium at inverse temperature beta. All spin
configurations participate. No bath acts during the drive. Every bond, including
the bond joining the last and first sites, has the same coupling.

Use units with Planck's reduced constant and Boltzmann's constant equal to one.
For a setting `[h, J, d, phi]` the Hamiltonian is

    H = sum_j [h Z_j/2 - J (S+_j S-_{j+1} + S-_j S+_{j+1})
               + d (exp(i phi) S+_j S+_{j+1} + exp(-i phi) S-_j S-_{j+1})].

Here X, Y and Z are Pauli matrices, S± = (X ± iY)/2, and indices wrap around
the ring. The initial setting is `prep`. On probe arm 0 the ring keeps this
setting throughout the experiment. On arm 1 it passes through `pulses` in the
listed time order. Each pulse is `[duration, setting]`. Switching is
instantaneous. Both arms have the same total duration.

The analyzer compares the final arm-1 amplitude against arm 0. For a pure
initial ring state its signed signal is the real part of
`exp(-i angle) <final_0 | final_1>` times the unknown detector visibility.
For thermal preparation use the ensemble mean of that signal. The visibility
is common to all experiments and lies between 0.65 and 0.95.

An input has keys `n` (4 to 28 sites), `beta` (0 to 4 in inverse energy
units), `prep`, `pulses` (1 to 5), and `angle` (radians). Fields and
couplings lie between -2 and 2 in energy units. Each duration lies between
0 and 4 in inverse energy units. Angles and phases are in radians.
Each record in `data/calibration.json` contains an `input`, a measured signal
`value`, and its independent Gaussian standard error `sigma`.

Implement `Model.fit(records)` to set the floating-point attribute
`visibility` and return the model. Implement `Model.predict(inputs)` to
return a finite NumPy array of shape `(len(inputs),)` with one signed
dimensionless signal per input, including an empty array for an empty list.
