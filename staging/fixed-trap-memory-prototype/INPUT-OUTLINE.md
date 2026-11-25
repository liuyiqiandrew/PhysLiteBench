# Proposed public apparatus outline

A dilute particle moves between localized sites in a narrow channel. Label the sites0,1,...,N. SiteN is an absorbing detector. The particle starts at0 at time0 and the measurement ends on its first arrival atN. The site spacing is the fixed length unit L0, and time is measured in t0.

Each experimental repetition uses a newly prepared channel. Independently at each site0,...,N-1, fabrication selects typeA with probability1-s and typeB with probability s. The resulting site types stay fixed during that repetition. The particle's preparation at0 is independent of those types. SiteA has total escape rate gamma and siteB has total escape rate gamma/c, where gamma is a common unknown inverse-time scale and c is a known positive residence-time ratio. Each visit has an exponential holding time at the corresponding rate; conditional on the fabricated channel, the waiting clocks and jump directions are independent. There is no interaction, loss, additional waiting time during travel, or time spent in the detector.

At site0, the only departure is to1 at that site's escape rate. From an interior transient site i, a departure goes to i+1 with known probability p and to i-1 with probability1-p. These probabilities do not depend on the site's type. All clocks run at the stated time-independent rates.

The readout is either the mean first-arrival time or its centered variance across the complete set of independent repetitions. For variance, the square of the overall mean is subtracted from the overall second moment, with both channel fabrication and particle motion included in the repetition ensemble. Mean and variance have units t0 and t0 squared, respectively.

Prospective controls are N, p, c, s and the requested statistic; gamma in[.6,1.4] is the sole fitted parameter. The final package would use fixed instrument uncertainties, a None parameter default, the neutral full-implementation instruction and no named disorder approximation, visit-count identity or specified solution method in the public description. This outline is not yet an evaluated task.
