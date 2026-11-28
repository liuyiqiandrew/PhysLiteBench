# Coexisting frozen landscapes

A system of size N can occupy a configuration in either of two phases. The phases have independent random two-level energy trees. In phase r, each family has a Gaussian energy (X_{r0}) with mean zero and variance (N\,\text{gain}^2 v_{r0}). Each leaf in that family has an independent Gaussian increment (X_{r1}) with variance (N\,\text{gain}^2 v_{r1}). Phase r contains floor(exp(N s_{r0})) families and floor(exp(N s_{r1})) leaves per family. All draws at different nodes are independent. The two-by-two arrays `entropies` and `variances` give the positive s and v values, indexed [phase][depth] with family depth 0 and leaf depth 1. In each phase (s_{r0}/v_{r0}<s_{r1}/v_{r1}).

A bias circuit adjusts the energy of every leaf in phase r by adding

    sum over depths j of [N gain sqrt(2 s_rj v_rj) - log(N)/(2 beta_rj)]
      + log_bias[r] * log(N) + offset[r],
    beta_rj = sqrt(2 s_rj/v_rj) / gain.

The circuit uses the actual parameters of the gas. Each leaf energy is its own path sum of X values plus that phase bias. The log is natural. The two arrays `log_bias` and `offset` have one entry per phase. All inputs are fixed along the limit. The measurement uses one Gibbs law at `inverse_temperature`, with a normalizer summed over both phases in the same frozen sample. The inverse temperature is above all four beta_rj throughout the fit range.

In `overlap_record` make one independent Gibbs draw per row of `record`, with one to six rows. A row gives [family label, leaf label, phase filter]. A phase filter of −1 accepts either phase, and 0 or 1 requires exactly that phase. Families in distinct phases are distinct families. Two draws must have the same family exactly when their family labels agree. Within a family they must have the same leaf exactly when their leaf labels agree. Leaf labels are local to that family. Return the probability of the whole exact labelled event, averaging over the original frozen samples after forming the Gibbs probabilities, and taking N to infinity. The top-level keys are `observable`, `entropies`, `variances`, `log_bias`, `offset`, `inverse_temperature`, and `record`.

In the calibration experiment `bare_energy_density` use only one phase whose index is `phase`, with no applied bias circuit. Use the same `entropies` and `variances` keys as above, and a high-temperature `inverse_temperature` below that phase's beta_rj values. Return the disorder-averaged expectation of its unshifted leaf energy divided by N as N tends to infinity. Fit the shared gain in [.65,1.35]. Each calibration row contains `input`, `value` and positive Gaussian standard deviation `sigma`. Implement `Model.fit(records)` to return self and store `self.gain`. Implement `Model.predict(experiments)` to return a one-dimensional NumPy array in input order, including an empty array for empty input.
