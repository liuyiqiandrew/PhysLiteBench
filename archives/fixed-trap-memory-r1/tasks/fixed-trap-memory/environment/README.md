# Passage through a channel of localized traps

An ideal dilute hopping experiment follows one particle through a narrow channel with sites labeled 0,1,...,N. Neighboring sites are separated by the fixed length unit L0. Site N is an absorbing detector. A particle is placed at site 0 at time 0, independently of the channel preparation; its first arrival at N ends the measurement. Travel between sites takes negligible time. There are no interactions, extra loss channels or detector delays.

Each repetition uses a newly fabricated channel. Independently at every site 0,...,N-1, the site is type A with probability 1-s and type B with probability s. These site properties remain fixed throughout the repetition. Type A has total escape rate gamma, and type B has total escape rate gamma/c. Each visit has an exponential holding time at the corresponding rate. Conditional on the fabricated channel, all holding clocks and departure-direction choices are independent. This continuous-time hopping law is the specified ideal model of thermal escape from the localized traps.

At site 0, the only possible departure is to site 1 at the site's total escape rate. At an interior transient site i, departure goes to i+1 with probability p and to i-1 with probability 1-p. These probabilities are independent of the site's type. The detector absorbs on arrival without another holding time.

The measured statistic is either the mean first-arrival time or its centered variance across independent repetitions. The variance is the overall second moment minus the square of the overall mean, including both channel fabrication and particle motion in that ensemble. Time is measured in the fixed unit t0; mean and variance have units t0 and t0 squared, respectively.

The sole unknown is the common escape-rate scale `0.6 <= escape_scale <= 1.4`, in units t0^-1. Each input gives these exact known controls:

- `size`: integer N, from 2 to 9;
- `right_probability`: p, from 0.45 to 1;
- `residence_ratio`: c, from 1 to 25;
- `slow_probability`: s, from 0 to 1;
- `statistic`: either `"mean"` or `"variance"`.

`data/calibration.json` contains independent records `{"input": {...}, "value": ..., "sigma": ...}`. Instrument errors are additive Gaussian. Their fixed standard deviation is 0.1 t0 for means and 0.5 t0 squared for variances, independent across records.

Implement `Model.fit(records)`, returning `self` and storing the fitted common scale in `self.escape_scale`. Implement `Model.predict(experiments)`, returning a finite one-dimensional NumPy array of the requested statistics in input order. Preserve this interface and the stated units.
