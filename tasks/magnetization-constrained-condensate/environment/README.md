# Magnetization-constrained condensate

An ideal gas of identical spin-one bosons is in a periodic cubic box of volume V. The exact total particle number and exact total magnetization before a pulse, divided by V, approach the inputs `density` and `magnetization`, with bounded rounding errors in the extensive quantities. The spin projections j=-1,0,+1 contribute j each to magnetization. The density is in [0.08,2.8] and the magnitude of magnetization is at most 0.95 times the density. No other occupation number is fixed. The preparation is the canonical quantum ensemble restricted to those two exact totals. At positive `temperature` in [0.25,1.65], a one-particle mode with wave vector k and spin j has energy

    abs(k)^2/(2*mass) + quadratic*j^2 + split[j]/V.

Use the usual momenta of a periodic cube, and hbar and kB equal to one. The input `quadratic` is in [-0.55,0.55]. The three entries in `split` use spin order -1,0,+1 and are held fixed along with the other intensive inputs as V tends to infinity.

After isolation, apply a uniform pulse that leaves j=0 alone and mixes j=-1 and j=+1 at each momentum independently. Its one-particle matrix on those two spins is cos(a) times the identity minus i*sin(a) times the exchange matrix, with input angle a given by `pulse_angle`. The bosons undergo the corresponding lossless quantum evolution. Measure all three spin occupations at k=0 immediately after the pulse, obtaining n_minus, n_zero and n_plus. The window test uses the signed imbalance Y=(n_plus-n_minus)/N and the input closed interval `window`. Preparations avoid window endpoints at any deterministic limiting readout.

Return the quantity in `observable`. For `zero_fraction`, return the mean (n_minus+n_zero+n_plus)/N. For `window_probability`, return the probability Y falls in the window. For `neutral_window_mean`, return the mean of n_zero/N times the indicator that Y falls in the window, over all shots. Take the thermodynamic limit after defining the measurement at finite V.

The common mass lies in [0.69,1.30]. Implement `Model.fit(records)`, returning self and storing the fitted value as `self.mass`. Each JSON row contains `input`, `value`, and positive Gaussian standard deviation `sigma`. Implement `Model.predict(experiments)` to return a one-dimensional NumPy array in input order, including an empty array for no inputs.
