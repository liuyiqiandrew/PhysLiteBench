# Double critical sibling cloud

A sample of size N has a two-level random energy tree. Let v_0,v_1 be the two entries of `variances`, r be `ratio`, and d be `detuning`. There are floor(exp(N r v_0+d sqrt(N))) families and floor(exp(N r v_1−d sqrt(N))) sites per family. Draw an independent centered Gaussian X_i of variance N gain² v_0 for each family, and an independent centered Gaussian Y_ij of variance N gain² v_1 for each site. Define

    x_i  = (X_i  + N gain sqrt(2r) v_0)/sqrt(N),
    y_ij = (Y_ij + N gain sqrt(2r) v_1)/sqrt(N).

The Gibbs energy of site (i,j) is X_i+Y_ij−fields[0]*x_i−fields[1]*y_ij. Use k_B=1. For the measurements below set beta_N=sqrt(2r)/gain+window/sqrt(N). The apparatus uses the actual gain in these settings. All other controls are fixed as N grows.

For `thermal_band`, draw one independent Gibbs site over the entire frozen tree at beta_N. Require the drawn x and y to be in `family_band` and `site_band`, respectively. Return the ensemble probability in the limit.

For `scaled_group`, take one independent Gibbs draw from the entire tree per entry of `inside_bands`, followed by one per entry of `outside_bands`, all conditional on the same frozen tree. The first two to four draws, those in inside_bands, must choose the same actual family and mutually distinct sites in it. Their y coordinates must satisfy the respective bands in inside_bands. The zero to two remaining draws must choose mutually distinct actual families, each different from the first family's identity. Each outside_bands entry is a pair [family band,site band] for x and y of that draw. Return sqrt(N) times the original ensemble probability of the exact ordered event, in the limit. The family of the first group is unconstrained in x.

A band is a two-entry interval with null for an unbounded endpoint. Each main input has exactly the top-level keys `observable`, `variances`, `ratio`, `detuning`, `window`, and `fields`, plus `family_band` and `site_band` for thermal_band or `inside_bands` and `outside_bands` for scaled_group. The variances are in [.39,1.3], ratio in [.38,1.18], magnitudes of detuning and window at most .9 and 1.4, respectively, field magnitudes at most .98, and finite band endpoints in [−2.3,2.3].

For `warm_energy` calibration turn off the fields and use the supplied high-temperature `inverse_temperature`, smaller than sqrt(2r)/gain throughout the fit range. Return the limiting mean unperturbed Gibbs energy per N averaged over frozen samples. Use only observable, variances, ratio, detuning, and inverse_temperature. Fit gain in [.68,1.33]. Calibration rows contain `input`, `value`, and a positive Gaussian standard deviation `sigma`. Implement `Model.fit(records)` to store `self.gain` and return self. Implement `Model.predict(experiments)` to return a one-dimensional NumPy array in supplied order, including an empty array for empty input.
