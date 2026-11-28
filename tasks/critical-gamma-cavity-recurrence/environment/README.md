# Critical gamma cavity recurrence

Let the two entries of `shapes` be v_0,v_1. A size-N sample has floor(exp(N v_0 I)) families and floor(exp(N v_1 I)) sites per family, where

    I = log(1+tilt)−tilt/(1+tilt).

One independent family energy X is Gamma-distributed with shape N v_0 and scale gain. Every site has an independent additional energy Y with shape N v_1 and the same scale. The family energy is reused by all sites of that family. For a site define

    a_j = gain*v_j/(1+tilt)
    x = (X−N a_0)/sqrt(N)
    y = (Y−N a_1)/sqrt(N)
    c = tilt/gain.

Use physical energy X+Y+curvature*(x−y)²−field*(x−y) and inverse temperature c+window/sqrt(N). The apparatus uses the actual gain in those settings.

For `contrast_gate`, make one Gibbs draw and return the probability that x−y lies in `contrast_band`. For `site_recurrence`, make two independent Gibbs draws from the same frozen sample. Require the same actual site, its x−y in contrast_band, and its unmodified X+Y−N(a_0+a_1) in `energy_window`. Return sqrt(N) times that joint probability. For both readouts use the original equally weighted disorder samples and form the finite-sample probabilities before taking the thermodynamic limit.

The main input keys are exactly `observable`, `shapes`, `tilt`, `curvature`, `field`, `window`, and `contrast_band`, with energy_window added for site_recurrence. Shapes lie in [.43,1.23], tilt in [.81,1.93], curvature in [.09,.98], and field and window magnitudes at most .79 and 1.14. All intervals have increasing endpoints. Contrast endpoints are in [−1.9,1.9] or null for an unbounded side. Energy-window endpoints are finite and in [−1.9,1.9].

For `warm_energy` use only observable, shapes, tilt and the supplied high-temperature `inverse_temperature` below c throughout the fit range. With zero curvature and field, return the limiting mean of X+Y per N. Fit gain in [.68,1.33]. Calibration records contain `input`, `value` and positive Gaussian standard deviation `sigma`. Implement `Model.fit(records)` to set `self.gain` and return self, and `Model.predict(experiments)` to return a one-dimensional NumPy array in input order including the empty array for no inputs.
