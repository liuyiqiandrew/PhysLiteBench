# Simultaneous particle counters in a narrow channel

Two binding sites connect large particle reservoirs at fixed temperature T. A site is either empty or contains one particle; double occupancy of a site is forbidden. Both site energies are zero, and there is no interaction beyond this exclusion. Dynamics are classical incoherent jumps in continuous time. Reservoirs remain at their specified chemical potentials and are the only source or sink of particles. Rates do not depend on the counting apparatus.

The input `bias` is b=(mu_left-mu_right)/(k_B T), between -8 and 8. Set mu_left/(k_B T)=b/2 and mu_right/(k_B T)=-b/2. Write f_left=1/(1+exp(-b/2)) and f_right=1/(1+exp(b/2)). The apparatus has the following fixed kinetic laws, which obey local detailed balance:

- An empty site 1 receives a particle from the left reservoir at rate `rate`*f_left. An occupied site 1 loses its particle to that reservoir at rate `rate`*(1-f_left), regardless of site 2.
- An empty site 2 receives a particle from the right reservoir at rate `rate`*f_right. An occupied site 2 loses its particle to that reservoir at rate `rate`*(1-f_right), regardless of site 1.
- A particle jumps between sites at rate `rate` in either direction, only if the destination is empty.

There are no other jumps. The unknown common attempt rate `rate`, in inverse seconds, lies between 0.5 and 2.0 and is the same in every experiment. Each experiment first reaches the stationary distribution at its chosen bias. The particles are not observed or fixed during preparation.

At time zero, three ideal passive counters start simultaneously on the same channel trajectory. N_left counts left-reservoir injections minus removals; N_middle counts site-1-to-site-2 jumps minus reverse jumps; N_right counts right-reservoir extractions minus injections. All counters initially read zero. They stop after `duration` seconds, with duration between 0 and 12. The bias and rates remain constant throughout that window. Measurements repeat this entire preparation and counting procedure independently.

`weights` is a length-three list [w_left,w_middle,w_right], each entry between -1 and 1. The electronics form the signed count

    Q = w_left*N_left + w_middle*N_middle + w_right*N_right.

Return its ensemble mean for `statistic: "mean"`, or its centered ensemble variance E[(Q-E[Q])^2] for `statistic: "variance"`. Counts are dimensionless. Every experiment provides `bias`, `duration`, `weights`, and `statistic`. Calibration contains only measurements of the mean left count. These identify the common attempt rate. Calibration errors are independent Gaussian errors with the stated standard deviations.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Expose `Model.rate`. Records in `data/calibration.json` have keys `input`, `value`, and `sigma`. The starter evaluates the four-state occupation dynamics and the fluctuations of each individual counter; check how its output follows from the complete apparatus.
