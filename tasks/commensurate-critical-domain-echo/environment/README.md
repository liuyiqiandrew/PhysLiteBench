# Commensurate critical domain echo

In a two-level disordered material each family carries an integer count K_0 of excited microscopic domains. Each site in that family has its own independent count K_1. At level j the count is a binomial draw with N trials and excitation probability p_j. Draw the family counts once and reuse each in all its sites. Draw all site counts independently of the family counts and of each other.

Let rho_j be fractions[j]/20, where the two entries of `fractions` are integers. Take N through positive multiples of 20. Define tau=`tilt` and the two microscopic probabilities by

    p_j / (1−p_j) = exp(tau)*rho_j/(1−rho_j).

The level j branching factor is floor(exp(N s_j)), where

    s_j = rho_j*log(rho_j/p_j)+(1−rho_j)*log((1−rho_j)/(1−p_j)).

These are the family count for j=0 and the sites per family for j=1. Logs are natural. For each site put x=(K_0−N rho_0)/sqrt(N) and y=(K_1−N rho_1)/sqrt(N). The physical energy is gain*(K_0+K_1)−fields[0]*x−fields[1]*y. Use k_B=1 and inverse temperature beta_N=(tau+window/sqrt(N))/gain, with the actual gain used by the apparatus.

For `coordinate_band`, draw one Gibbs site from one sample at beta_N and require x in `family_band` and y in `site_band`. Return the ensemble probability. For `domain_echo`, take one independent full Gibbs draw per row of `sibling_bands` followed by one per row of `other_bands`, all conditional on the same sample. The first two to four draws must use the same actual family but distinct sites in that family. The integer K_0−N rho_0 of the common family must be one of `addressed_levels`. Each y in this group must fall in its corresponding sibling band. The zero to two other draws must use distinct families, all different from the common family. Each other_bands entry contains [family band,site band] for x,y of one such draw. Return sqrt(N) times the ensemble probability of the entire ordered event. Apply all conditions at finite N before taking the limit.

The main readouts share exactly `observable`, `fractions`, `tilt`, `window`, and `fields` plus the readout keys described above. Fractions are integers in [3,14], tilt in [.67,1.98], window magnitude at most 1.36, and field magnitudes at most .85. Bands are two increasing endpoints in [−1.72,1.72] with null for an unbounded endpoint. Addressed_levels is a nonempty list of distinct integers from −5 through 5.

For `domain_occupation` calibration use an isolated two-state system with excited energy gain and ground energy zero. The prior statistical weights of the excited and ground states are `prior` and 1−prior. At `inverse_temperature` return the equilibrium excited-state probability. Use only observable, prior and inverse_temperature. Fit gain in [.69,1.31]. Calibration rows contain `input`, `value`, and a positive Gaussian standard deviation `sigma`. Implement `Model.fit(records)` to store `self.gain` and return self. Implement `Model.predict(experiments)` to return a one-dimensional NumPy array in supplied order, including an empty array for empty input.
