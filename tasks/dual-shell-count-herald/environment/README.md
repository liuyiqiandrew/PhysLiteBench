# Dual shell count herald

A size-N sample has floor(exp(N ratio*v_F)) families and floor(exp(N ratio*v_L)) sites per family, with [v_F,v_L] from `variances`. Each family draws a centered Gaussian X of variance N gain² v_F. Each site draws a centered Gaussian Y of variance N gain² v_L. All draws are independent except that each family's sites reuse its X. Put V=v_F+v_L, c=sqrt(2 ratio)/gain, a=gain*sqrt(2 ratio), and

    R=(X−v_F*(X+Y)/V)/sqrt(N)
    U=X+Y+NaV−log(N)/(2c).

Setting j can see the actual sites whose spectral coordinate U−fields[j]*R lies in `shells[j]`. A site can be visible in both settings. Keep the original frozen energies and labels in both settings. The sample is heralded exactly when the two visibility counts equal the supplied positive integers `counts[0]` and `counts[1]). The same site counts once in each setting if visible in both.

For `herald_rate` return the limiting ensemble probability of this herald. For `sibling_record` sample the ensemble conditional on the herald. Draw one site uniformly from the visible identities in each setting, independent conditional on that frozen sample. The selected sites must use the same actual family and have the same actual site identity if `same_site` is true, or different site identities otherwise. The common family must have unmodified X+Na v_F in `parent_window`. In setting j the selected site's spectral coordinate must also be in `marks[j]`. If `outside` is an object make a third independent uniform draw in its `setting` (0 or 1). Require a family different from the first pair and require its spectral coordinate in the object's `mark` and R in its `bridge`. If outside is null make no third draw. The samples are with replacement. Return the limit of sqrt(N) times the full event probability conditional on the finite-size herald. Form every count and draw at finite N before taking the limit.

Both observables use exactly the fields named and `observable`. Ratio is in [.29,.74], variances in [.48,1.27], and the two field magnitudes at most .96. Counts are integers from 1 through 4. All band endpoints are finite in [−1.8,1.8] with the upper strictly greater than the lower. The shells [l0,h0],[l1,h1] obey l1−l0≥.15, h0−l1≥.15 and h1−h0≥.15.

For `warm_energy` calibration use just observable, variances, ratio and `inverse_temperature` in [.08,.34]. Use no spectral selections and return the limiting mean of X+Y per N in the Gibbs ensemble at that inverse temperature. Fit gain in [.68,1.31]. Each calibration row contains `input`, `value`, and positive Gaussian standard deviation `sigma`. Implement `Model.fit(records)` to store `self.gain` and return self. Implement `Model.predict(experiments)` to return a one-dimensional NumPy array in input order including the empty array for empty input.
