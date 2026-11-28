# Frozen root critical leaf echo

A size-N tree has floor(exp(N ratio*v_F)) families with floor(exp(N ratio*v_L)) sites each. Family energies X and site increments Y are independent centered Gaussians of variances N gain² v_F and N gain² v_L. Every site of a family uses the same X. Put a=gain*sqrt(2 ratio), c=sqrt(2 ratio)/gain, x=(X+Na v_F)/sqrt(N), and y=(Y+Na v_L)/sqrt(N).

Prepare two independent Gibbs draws on the same frozen tree. In setting 0 use energy `amplification`*X+Y−`cold_field`*y at inverse temperature c+`cold_window`/sqrt(N). In setting 1 use energy X+Y−`critical_fields`[0]*x−critical_fields[1]*y at inverse temperature c+`critical_window`/sqrt(N). The entire tree is normalized independently in each setting. The apparatus uses the actual gain.

For `frozen_echo` require the two draws to select the same actual site, with the two unmodified energies satisfying

    X+Na v_F−log(N)/(2c) in root_window,
    Y+Na v_L in leaf_window.

If `outside_band` is an interval, make an extra independent draw in setting 0 and require a different family and y in that interval. If outside_band is null omit the extra draw. Return the limit of N times the probability of the complete joint event averaged over original samples. Apply all requirements at finite N before taking the limit.

The main inputs are the named fields, `observable` and `variances`=[v_F,v_L]. Variances are in [.51,1.23], ratio in [.32,.75], amplification in [1.21,2.42], field and window magnitudes at most 1.03, and intervals have increasing finite endpoints in [−1.57,1.57]. For `warm_energy`, use the whole tree with unmodified energy X+Y at the supplied inverse_temperature in [.08,.34]. Use just observable, variances, ratio and inverse_temperature. Return the limiting mean unmodified energy per N.

Fit gain in [.69,1.31]. Calibration rows contain input, value and positive sigma. Implement `Model.fit(records)` storing `self.gain` and returning self. Implement `Model.predict(experiments)` returning a one-dimensional NumPy array in input order including the empty array for empty input.
