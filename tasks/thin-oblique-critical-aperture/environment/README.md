# Thin oblique critical aperture

A size-N sample has floor(exp(N ratio*v_F)) families and floor(exp(N ratio*v_L)) sites per family. Its family energies X are centered Gaussians of variance N gain² v_F. Its site energies Y are centered Gaussians of variance N gain² v_L. Energies are independent across labels, with X reused at all sites of its family.

Let a=gain*sqrt(2 ratio), c=sqrt(2 ratio)/gain, x=(X+Na v_F)/sqrt(N), y=(Y+Na v_L)/sqrt(N), and delta=N^(−1/4). Each entry of `settings` has `window`, two `fields`, `mixing` and a `gate` interval. In setting j use inverse temperature c+window/sqrt(N) and energy X+Y−fields[0]*x−fields[1]*y. With m=mixing, its Gibbs law includes only sites for which

    (x+m*y−(1−m)*center)/delta

lies in its gate interval. These restrictions are applied before normalizing either law. The apparatus uses the actual gain. Both settings use the same frozen sample.

For `thin_echo`, make one independent Gibbs draw per setting. Require the same actual site, (x−center)/delta in `address_band`, and

    X+Y+Na(v_F+v_L)−exponent*log(N)/c

in `line_window`. Here exponent is 1/2 when center is zero and 1/4 when center is strictly positive. Return the limit of N^(3/4−exponent) times the joint probability averaged over equally weighted samples. An empty gate gives zero event probability. Form all gates and draws at finite N.

Use exactly the named fields, `variances`=[v_F,v_L] and `observable`. Variances are in [.49,1.24], ratio in [.31,.76], center in [0.,1.01], mixing in [.19,.81], fields and window in [−1.07,1.07], and gate endpoints in [.13,1.62]. The other interval endpoints are in [−1.67,1.67]. Each interval is increasing.

For `warm_energy` calibration use just observable, ratio, variances and inverse_temperature in [.08,.34]. Remove the gates and all fields and return the thermodynamic limiting Gibbs mean of X+Y per N at that inverse temperature. Fit gain in [.69,1.31]. Calibration records have input, value and positive Gaussian standard deviation sigma. Implement `Model.fit(records)` to store `self.gain` and return self. Implement `Model.predict(experiments)` returning one-dimensional NumPy predictions in input order including the empty array for empty input.
