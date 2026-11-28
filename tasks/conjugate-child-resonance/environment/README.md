# Conjugate child resonance

A size-N tree has floor(exp(N ratio*v_F)) families and floor(exp(N ratio*v_L)) sites per family. The two entries of `variances` are v_F and v_L. Each family has a centered Gaussian X of variance N gain² v_F. Every site has an independent centered Gaussian increment Y of variance N gain² v_L and an independent moment m=±1 of either sign with equal probability. All energies and moments are fixed in one frozen sample.

With s_0=+1, s_1=−1, c=sqrt(2*ratio)/gain and A=gain sqrt(2*ratio), setting j uses energy X+s_j Y−fields[j]*m and inverse temperature c+s_j*window/sqrt(N). The apparatus uses the actual gain. The sign reversal changes the site increment only.

For `conjugate_return`, make one independent Gibbs draw in each setting. Require the same actual site, its family's X+N A v_F in `family_window`, and the common Y/sqrt(N) in `central_band`. The filter `moment` is −1 or +1 for a required moment, or 0 for either. If `satellite` is an object, make one more independent Gibbs draw at its setting (0 or 1). Require another actual site of the same family and its (s_j Y+N A v_L)/sqrt(N) in the satellite band. Null satellite omits the extra draw.

Return sqrt(N) exp(3 N ratio*v_L) times the whole event probability averaged over equally weighted frozen samples. Form all Gibbs probabilities over the full tree at finite N before applying filters and taking N to infinity.

The main keys are exactly `observable`, `variances`, `ratio`, `window`, `fields`, `family_window`, `central_band`, `moment`, and `satellite`. A satellite object has keys `setting` and `band`. Variances are in [.43,1.24], ratio in [.43,1.14], window magnitude at most 1.08 and field magnitudes at most .81. All intervals have increasing finite endpoints in [−2.1,2.1].

For `warm_energy` use only observable, variances, ratio and a positive `inverse_temperature` below c throughout the gain range. With energy X+Y and zero fields, return the thermodynamic Gibbs mean per N. Fit gain in [.68,1.33]. Calibration records contain `input`, `value`, and positive Gaussian standard deviation `sigma`. Implement `Model.fit(records)` to set `self.gain` and return self, and `Model.predict(experiments)` to return a one-dimensional NumPy array in order, including an empty array for no inputs.
