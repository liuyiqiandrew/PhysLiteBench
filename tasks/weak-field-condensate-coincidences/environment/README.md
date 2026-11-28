# Weak-field condensate coincidences

A three-dimensional ideal gas of identical bosons in a periodic cubic box has exactly N particles. A single-particle mode is labelled by its momentum k and its internal index j. There are two to four internal states. In a box of volume V the one-particle energy is

    abs(k)^2/(2*mass) + splitting[j]/V,

with the usual periodic momenta of a cube. The input gives the `splitting` vector, the target `density` N/V and positive `temperature`. All are held fixed while V and N go to infinity. Temperature, momentum and mass use hbar and kB equal to one. The preparation is the canonical quantum ensemble with exactly N particles, with every internal label equally available in counting states.

Then isolate the gas and apply `pulses`, a list of real symmetric internal one-particle Hamiltonian matrices and durations. Each pulse acts within each momentum independently and the particles never interact. An ideal final detector resolves all occupations in the original momentum and internal basis. For `zero_fraction`, return the mean fraction of all particles with momentum zero. For `coincidence`, sample distinct particles uniformly from the final gas, after reading the occupation numbers in that shot. Return the probability that every sampled particle has momentum zero and has the internal index given by the corresponding entry in `detectors`. This asks for one ordered pattern of one to four sampled particles, with specified repetitions allowed. Average over full detection shots. Take the thermodynamic limit after defining these probabilities at finite V.

The only unknown common parameter is the mass in [0.63,1.36]. Implement `Model.fit(records)`, returning self and storing the fitted mass as `self.mass`. Each JSON calibration row contains `input`, `value` and positive Gaussian standard deviation `sigma`. Implement `Model.predict(experiments)` to return a one-dimensional NumPy array in input order, including an empty array for no inputs.
