# Twin resonant family calorimeter

An ordinary size-N sample has floor(exp(N ratio*v_F)) families and floor(exp(N ratio*v_L)) sites per family. The two `variances` entries are v_F,v_L. Gaussian family energies X have mean zero and variance N gain² v_F. Gaussian site increments Y have mean zero and variance N gain² v_L. The Gaussians are independent across actual labels. A family energy is reused by every site in that family. Write

    a_j=gain*sqrt(2*ratio)*v_j
    c=sqrt(2*ratio)/gain
    x=(X+N a_F)/sqrt(N),    y=(Y+N a_L)/sqrt(N).

Add two labelled families (tags 0 and 1) to the same sample, each with the same number of sites and the same independent site increments. Their family energies are set deterministically to −N a_F+`tag_energies[j]`. The ordinary energy is X+Y−fields[0]*x−fields[1]*y. In tag j use energy −N a_F+tag_energies[j]+Y−tag_fields[j]*y. Use one Gibbs law over all sites in all families at inverse temperature c+window/sqrt(N). Apparatus controls use the actual gain.

For `tagged_population`, make one Gibbs draw and require tag `tag` (0 or 1) and y in `band`. Return its probability. For `tagged_repeat`, make two independent Gibbs draws requiring the same actual site in tag tag and that site's unmodified Y+N a_L in `site_window`. If `opposite_band` is a band, make a third independent draw and require the other tag and y in that band. If opposite_band is null there is no third draw. Return sqrt(N) times the latter complete probability. For both readouts average over original equally weighted frozen samples and take N to infinity after forming the finite-sample probabilities.

The main keys are exactly `observable`, `variances`, `ratio`, `window`, `fields`, `tag_energies`, `tag_fields`, `tag` and the specified readout keys. Variances are in [.43,1.24], ratio in [.43,1.14], window and field magnitudes at most 1.12 and .87, and tag energy magnitudes at most 1.16. Bands have increasing endpoints in [−2.1,2.1]. Site-window endpoints are finite, and other bands can use null for an unbounded side.

For `warm_energy` use only observable, variances, ratio and a positive inverse_temperature below c throughout the gain range. Omit both added families, set zero fields, and return the Gibbs mean of X+Y per N in the thermodynamic limit. Fit gain in [.68,1.33]. Calibration records contain `input`, `value` and a positive standard deviation `sigma`. Implement `Model.fit(records)` to set `self.gain` and return self, and `Model.predict(experiments)` to return a one-dimensional NumPy array in order including an empty array for no inputs.
