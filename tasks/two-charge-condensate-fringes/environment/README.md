# Two-charge condensate fringes

An ideal Bose gas occupies a periodic cubic box of volume V with seven internal modes per momentum. In the stated internal order each particle carries charges

    (1,0),(-1,0),(0,1),(0,-1),(0,0),(1,1),(-1,-1).

Both charges and the total particle number are fixed exactly in the initial canonical quantum ensemble. Their values divided by V approach `charges` and `density`, with bounded rounding errors. The ordinary one-particle energies are k²/(2*mass) for the first five modes and k²/(2*mass)+`gaps`[j] for the last two. Use hbar=kB=1. There is a further finite preparation energy that depends on the occupations `n` of the first five modes at k=0 only:

    (weak dot n)/V + (n^T interaction n)/(2*V²).

The supplied matrix `interaction` is real symmetric. Both terms are diagonal in the occupation basis. The common unknown mass is in [.62,1.34].

For `line_density`, return the limiting mean total occupation of the five zero-momentum modes without a gap, divided by V. For other observables, isolate the gas and apply the stated sequence `pulses` of real symmetric 5-by-5 matrices H with one-particle action exp(-i*H) among just those five zero-momentum modes. There is no additional dynamics. These pulses may change the internal charges. The final occupation of output i divided by V is I_i. For `port_density`, return the mean I[`port`]. For `weighted_difference`, return the expectation of

    (I[ports[0]]-I[ports[1]]) * exp(-sum_i ((I[i]-center[i])/width[i])²/2).

The five Gaussian detector weights and the difference use counts from the same shot. Take V to infinity after defining every observation at finite V.

Temperature is in [.43,1.20], density in [1.05,3.0], gaps in [.24,1.23], entries of weak and interaction have magnitude at most 2.7, pulse entries at most 1.2, detector center entries in [0,1.9] and width entries in [.13,.8]. The supported preparations have positive excess density after thermal saturation of the nonground occupations at zero chemical potentials. The residual charge pair after that saturation lies strictly within the charge diamond of those excess particles, with a margin of at least .06 in density units at every allowed mass.

Implement `Model.fit(records)`, returning self and storing the estimate as `self.mass`. Each calibration row contains `input`, `value` and positive Gaussian standard deviation `sigma`. Implement `Model.predict(experiments)` to return a one-dimensional NumPy array in input order.
