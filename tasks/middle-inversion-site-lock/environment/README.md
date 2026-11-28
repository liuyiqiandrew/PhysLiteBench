# Middle inversion site lock

A size-N energy tree has three branching depths i=0,1,2. Each vertex at depth i has floor(exp(N ratio*v_i)) children, where the three v_i are `variances`. Each edge has an independent centered Gaussian energy X_i of variance N gain² v_i. A site is a complete path to depth 2. Sites that use a shared edge use the same Gaussian draw for that edge.

Let c=sqrt(2 ratio)/gain, a=gain*sqrt(2 ratio) and V=v_0+v_1+v_2. The two settings on one frozen tree have inverse temperatures c+`temperature_windows[j]`/sqrt(N). The site energies in settings 0 and 1 are X_0+X_1+X_2 and X_0−X_1+X_2 respectively. The apparatus uses the actual gain. Use k_B=1.

For `site_lock`, draw independently from the two full Gibbs laws, one draw per setting. Require the same actual site. Its signed total energy plus NaV in setting j must lie in the corresponding entry of `line_windows`. Require X_0+Na v_0 for its first edge in `root_window`. If `additional` is an object, draw one more site from its specified setting (0 or 1). For region "different_root" this draw must use a different actual first edge. For region "same_root_new_middle" it must use the same actual first edge and a different actual middle edge. If additional is null there is no third draw.

Let K=ratio*v_1*(v_1+v_2)/v_2. Return the limit of N^(3/2)*exp(N K) times the complete event probability averaged over equally weighted original samples. Form all finite-N Gibbs draws and selections before taking the limit.

The main keys are exactly those named and `observable`. Ratio is in [.25,.63], each variance in [.46,1.24], and temperature windows in [−1.19,1.19]. All line and root endpoints are in [−1.62,1.62] with strictly increasing intervals.

For `warm_energy` use only observable, variances, ratio and `inverse_temperature` in [.08,.32]. Use the ordinary unflipped tree, with no selections. Return the thermodynamic limiting Gibbs mean of the total energy per N. Fit gain in [.68,1.31]. Calibration records contain `input`, `value`, and positive Gaussian standard deviation `sigma`. Implement `Model.fit(records)` to store `self.gain` and return self. Implement `Model.predict(experiments)` to return a one-dimensional NumPy array in input order including the empty array for empty input.
