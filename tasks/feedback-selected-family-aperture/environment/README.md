# Feedback-selected family aperture

A size-N sample has floor(exp(N ratio*v_F)) families and floor(exp(N ratio*v_L)) sites per family. Family energies X are centered Gaussians of variance N gain² v_F. Site increments Y are centered Gaussians of variance N gain² v_L. Draw all energies independently across actual labels. The same X is used by every site of its family. Use the supplied variances [v_F,v_L], a=gain*sqrt(2 ratio), c=sqrt(2 ratio)/gain, x=(X+Na v_F)/sqrt(N), and y=(Y+Na v_L)/sqrt(N).

The energy in setting j is X+Y−settings[j].fields[0]*x−settings[j].fields[1]*y, with inverse temperature c+settings[j].window/sqrt(N). Use the actual gain. A first full Gibbs draw in setting 0 records a site, its family and its y value y*. Require its X+Na v_F in `root_window` and y* in `first_band`.

After this draw, set a threshold h=`aperture.bias`+`aperture.slope`*y*. In setting 1 include only sites with x+`aperture.mixing`*y≥h. This is the normalization rule for every subsequent draw. It uses each candidate site's own x and y, including the recorded site's coordinates if it remains eligible. Require a second Gibbs draw in setting 1 to land on a different actual site of the recorded family and its y in `sibling_band`. If `outside_band` is an interval, make a third independent draw in that same setting and require a different family and its y in that interval. If outside_band is null omit the third draw. An empty restricted catalog gives probability zero.

For `feedback_aperture` form the full joint probability at finite N using the aperture set by each actual first draw. Return the limit of sqrt(N) times this probability averaged over equally weighted original samples.

Use exactly the named keys and `observable`. Ratio is in [.31,.76], variances in [.49,1.26], fields and windows in [−1.08,1.08], bias and slope in [−1.01,1.01], and mixing is either zero or in [.37,1.32]. All bands are increasing and have finite endpoints in [−1.71,1.71].

For `warm_energy` calibration use just observable, variances, ratio, and inverse_temperature in [.08,.33]. Turn off fields and the aperture and return the thermodynamic Gibbs mean X+Y per N at that inverse temperature. Fit gain in [.69,1.31]. Calibration rows contain input, value, and positive Gaussian standard deviation sigma. Implement `Model.fit(records)` to store `self.gain` and return self. Implement `Model.predict(experiments)` to return a one-dimensional NumPy array in input order including the empty array for empty input.
