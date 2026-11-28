# Entropy-shifted branch echo

A tree at size N has three branching depths j=0,1,2. Every vertex at depth j branches into floor(exp(N ratio*v_j+sqrt(N)*entropy_shifts[j])) edges. The three entries in `variances` give v_j. Each edge at depth j draws an independent centered Gaussian energy X_j with variance N gain² v_j. The edge energy is shared by all paths through that edge. Define

    c=sqrt(2*ratio)/gain,
    x_j=(X_j+N gain sqrt(2*ratio)*v_j)/sqrt(N).

A path's physical energy is sum_j(X_j−fields[j]*x_j). Use inverse temperature c+window/sqrt(N). The apparatus uses the actual gain in its settings and energy coordinates.

For `branch_echo`, make two independent Gibbs draws from the same frozen sample, with replacement. The two paths must share their actual first and second edges and have different third edges. Require the shared x_0 to lie in `first_band`. For the pair's shared prefix define

    D=X_0+X_1+N gain sqrt(2*ratio)*(v_0+v_1)
      +sqrt(N)*(entropy_shifts[0]+entropy_shifts[1])/c.

Require D in `prefix_window`. Each entry in `leaf_bands` bounds x_2 for the corresponding draw. When `outside` is true, make a third independent draw from that same Gibbs law and require it to use a different actual first edge from the pair. There is no further condition on that draw. When outside is false, there are just the two draws. Return sqrt(N) times the probability of the complete event over equally weighted disorder samples, taking the limit after defining the finite-sample probabilities.

The top-level keys for branch_echo are exactly `observable`, `variances`, `ratio`, `window`, `entropy_shifts`, `fields`, `first_band`, `prefix_window`, `leaf_bands`, and `outside`. Variances lie in [.43,1.26] and ratios in [.43,1.16]. The fields have magnitude at most .87, entropy shifts at most .94, and window at most 1.12 in magnitude. Prefix endpoints are finite and in [−2.2,2.2]. Other band endpoints can be null for an open side or finite in [−2.2,2.2]. All intervals have increasing endpoints.

For `warm_energy` use exactly observable, variances, ratio and the high-temperature `inverse_temperature` below c over the gain fit range. Use zero fields and entropy shifts. Return the limiting mean of X_0+X_1+X_2 per N under the Gibbs law averaged over disorder. Gain lies in [.68,1.33]. Calibration records contain `input`, `value`, and a positive Gaussian standard deviation `sigma`. Implement `Model.fit(records)` to set `self.gain` and return self. Implement `Model.predict(experiments)` to return a one-dimensional NumPy array in supplied order, including an empty array for no inputs.
