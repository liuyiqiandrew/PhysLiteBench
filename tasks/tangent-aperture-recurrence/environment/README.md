# Tangent aperture recurrence

A size-N tree has floor(exp(N ratio*v_F)) families with floor(exp(N ratio*v_L)) sites per family. Draw centered Gaussian family energies X of variance N gain² v_F and site increments Y of variance N gain² v_L. These draws are independent across actual labels, with X reused at all sites of its family. Set a=gain*sqrt(2 ratio), c=sqrt(2 ratio)/gain, x=(X+Na v_F)/sqrt(N) and y=(Y+Na v_L)/sqrt(N).

For setting j use energy X+Y−settings[j].fields[0]*x−settings[j].fields[1]*y at inverse temperature c+settings[j].window/sqrt(N). Its Gibbs law is restricted before drawing to sites with

    x+y >= settings[j].curvature*(x−center−settings[j].shift/N^(1/4))².

The apparatus uses the actual gain. Both settings share one frozen tree. An empty restricted law contributes zero to the event.

For `tangent_echo` make one independent Gibbs draw from each setting. Require the same actual site, N^(1/4)*(x−`center`) in `address_band`, and X+Y+Na(v_F+v_L) in `line_window`. If `outside` is an object, make a third independent Gibbs draw from its `setting` (0 or 1). Require another family and y in its `band`. If outside is null omit that draw. Return the limit of N^(3/4) times the complete joint probability averaged over equally weighted original samples. Form all gates and Gibbs probabilities at finite N first.

Use exactly the named fields and `observable`. The two `variances` v_F,v_L are in [.49,1.26], ratio in [.31,.77], center in [0.,.97], and curvature in [.37,1.53]. Window, field and shift magnitudes are at most 1.11. Both endpoints of each increasing interval are in [−1.68,1.68].

For `warm_energy` calibration use just observable, variances, ratio, and inverse_temperature in [.08,.34]. Turn off the aperture and fields. Return the thermodynamic limiting mean unperturbed energy X+Y per N at that inverse temperature. Fit gain in [.69,1.31]. Calibration records contain input, value, and positive Gaussian standard deviation sigma. Implement `Model.fit(records)` to store `self.gain` and return self. Implement `Model.predict(experiments)` returning one-dimensional NumPy predictions in input order including the empty array for empty input.
