# Row-column bond echo

A size-N random lattice has floor(exp(N ratio*v_A)) rows and floor(exp(N ratio*(v_B+v_C))) columns. Each row has a centered Gaussian energy X of variance N gain² v_A, each column a centered Gaussian energy Y of variance N gain² v_B, and each row-column bond a centered Gaussian energy Z of variance N gain² v_C. All independent energy draws are reused at every occurrence of their actual label. Each bond represents one physical state.

Use [v_A,v_B,v_C] from `variances`, c=sqrt(2 ratio)/gain, a=gain*sqrt(2 ratio), and the coordinates

    x=(X+Na v_A)/sqrt(N)
    p=(Y+Z+Na(v_B+v_C))/sqrt(N)
    r=(Y−v_B*(Y+Z)/(v_B+v_C))/sqrt(N).

Setting j uses energy X+Y+Z−fields[0]*x−fields[1]*p−fields[2]*r and inverse temperature c+window/sqrt(N), with fields and window from `settings[j]`. The apparatus uses the actual gain. Make one Gibbs draw in each setting, independent conditional on the same frozen lattice.

For `row_echo`, require the same actual row and different columns. Require the row's X+Na v_A in `line_window`. Require the two draws' p and r to lie in their corresponding `pool_bands` and `contrast_bands`. For `bond_echo`, require the same actual bond, its x in `bridge_band` and r in `contrast_band), and X+Y+Z+Na(v_A+v_B+v_C) in line_window. For either readout, if `outside` is an object, make one additional independent draw in its `setting` (0 or 1) and require a different actual row and its r in that object's `contrast_band`. Set outside to null to omit the draw.

Return the limit of sqrt(N) times the full event probability over equally weighted original samples. Form all finite Gibbs normalizations before applying the limit.

Use exactly the named fields and `observable`. Variances are in [.48,1.24], ratio in [.29,.76], windows and field entries in [−1.09,1.09], and both endpoints of each increasing interval in [−1.71,1.71].

For `warm_energy` calibration use just observable, variances, ratio, and inverse_temperature in [.08,.33]. Use the unmodified lattice and return the limiting mean energy X+Y+Z per N at that inverse temperature. Fit gain in [.69,1.31]. Calibration rows contain input, value, and positive Gaussian standard deviation sigma. Implement `Model.fit(records)` returning self and storing `self.gain`, and `Model.predict(experiments)` returning one-dimensional NumPy predictions in input order including the empty array for empty input.
