# Aperture-selected critical roots

A size-N random tree has three branching depths j=0,1,2. Each vertex at depth j branches into floor(exp(N ratio*v_j)) edges, where `variances` gives v_0,v_1,v_2. Each edge at depth j has an independent centered Gaussian energy X_j of variance N gain² v_j, reused by all paths through that edge. For a path put

    c=sqrt(2*ratio)/gain
    x_j=(X_j+N gain sqrt(2*ratio) v_j)/sqrt(N).

Use the same frozen tree for the two supplied `settings`. Setting i defines an energy sum_j(X_j−fields[j]*x_j) and inverse temperature c+window/sqrt(N). Its `aperture` is a finite interval for the unperturbed sum x_0+x_1+x_2. Remove all paths outside that aperture before forming the Gibbs probabilities for this setting. If either allowed population is empty the measurement event fails.

For `root_echo`, make one conditionally independent Gibbs draw from each of the two respective restricted populations. Require that the draws share the same actual first edge but use different actual second edges. Require the shared X_0+N gain sqrt(2*ratio)*v_0 to lie in `root_window`. Finally, for draw i, the two entries of `site_bands[i]` bound x_1 and x_2 of that draw, respectively. These last constraints are readout conditions after sampling. Return sqrt(N) times the probability of the full event over original frozen trees, taking the limit after forming the finite-tree probabilities.

`root_echo` uses exactly `observable`, `variances`, `ratio`, `settings`, `root_window`, and `site_bands`. Each setting contains `fields` (three entries), `window`, and `aperture`. Apertures have increasing endpoints in [0,2.35]. The root window has increasing finite endpoints in [−2.25,2.25]. Each site band has increasing endpoints in [−2.2,2.2], with null for an unbounded side. Variances lie in [.42,1.24], ratios in [.43,1.14], window magnitudes at most 1.1, and field magnitudes at most .85. The apparatus uses the actual gain in the above scaling.

For `warm_energy` inputs use only `observable`, `variances`, `ratio`, and the high-temperature `inverse_temperature` below c throughout the fit range. Remove all apertures and fields and return the limiting mean Gibbs energy sum_j X_j per N averaged over disorder. Fit gain in [.68,1.33]. Calibration rows contain `input`, `value`, and a positive Gaussian standard deviation `sigma`. Implement `Model.fit(records)` to set `self.gain` and return self. Implement `Model.predict(experiments)` to return a one-dimensional NumPy array in input order, including the empty array for no inputs.
