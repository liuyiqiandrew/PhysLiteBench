# Two phase sites in selected families

A sample of size N contains floor(exp(N s_F)) families. Each family independently contains floor(exp(N s_0)) sites of phase 0 and floor(exp(N s_1)) sites of phase 1. The three entries of `entropies` are [s_F,s_0,s_1] and the positive `variances` entries are [v_F,v_0,v_1]. Each family has an independent Gaussian energy X of mean zero and variance N gain² v_F. Each phase-r site has an independent Gaussian Y of mean zero and variance N gain² v_r. The Gaussian draws of different sites and families are independent.

For the overlap experiment all sites of phase r use the energy

    X + Y + N gain sqrt(2 s_r v_r) - log(N)/(2 beta_r) + offsets[r],
    beta_r = sqrt(2 s_r/v_r)/gain.

The offsets give one fixed number for each site phase. This tuning uses the physical gain. The root ratio s_F/v_F is strictly below both s_r/v_r. The inverse temperature `inverse_temperature` is larger than all three freezing inverse temperatures over the gain range.

First select one family. Under `protocol` equal to `thermal`, make a Gibbs draw over all families and phases at inverse_temperature and retain only the family label. Under `uniform`, choose a family uniformly independently of its energy. Both protocols start from equally weighted frozen samples. For `site_record` make one independent Gibbs draw inside the selected family per row of `record`, with the same temperature and the same phase biases. The initial draw used for family selection is absent from these rows. A row gives [site label, phase filter]. Matching site labels require the same actual site, and different labels require distinct sites. A phase filter of −1 accepts either phase, while 0 or 1 requires that phase. There are one to six rows. Return the probability of the exact labelled event, averaging after each frozen-sample draw and then taking N to infinity.

The top-level keys are `observable`, `entropies`, `variances`, `inverse_temperature`, `offsets`, `protocol`, and `record`. For the calibration observable `bare_energy_density`, use only the first four keys and the additional key `phase` giving 0 or 1. Discard the other site phase, turn off the tuning shifts and offsets, and use the supplied high-temperature inverse temperature. Return the limiting mean unshifted Gibbs energy per N, averaged over disorder.

Fit the shared gain in [.66,1.33]. Each calibration record contains `input`, `value`, and a positive Gaussian standard deviation `sigma`. Implement `Model.fit(records)` to return self and store `self.gain`. Implement `Model.predict(experiments)` to return one one-dimensional NumPy array of predictions in input order, including the empty array for empty input.
