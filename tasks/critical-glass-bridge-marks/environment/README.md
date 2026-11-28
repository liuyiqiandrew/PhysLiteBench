# Critical glass bridge marks

A frozen tree for each system size N has four branching depths, numbered 0 through 3. Each node at the preceding depth has floor(exp(N s_j(N))) children at depth j. A branch at depth j has an independent Gaussian energy increment X_j of mean zero and variance N gain² v_j. The array `variances` gives the four positive values of v_j. The positive `root_entropy` sets s_0(N). At inner depths j=1,2,3 use

    s_j(N) = inner_ratio * v_j + detuning[j-1] / sqrt(N).

The three entries of `detuning` sum to zero. For both experiment types the positive `inner_ratio` is strictly larger than root_entropy/v_0. Only sufficiently large N is used. Let Y=X_1+X_2+X_3 and V=v_1+v_2+v_3. An individual leaf has two quantities calculated from its own unperturbed path,

    r_1 = (X_1 - v_1 * Y / V) / sqrt(N)
    r_2 = (X_1+X_2 - (v_1+v_2) * Y / V) / sqrt(N).

For `bridge_record` the total energy is X_0+Y−field[0]*r_1−field[1]*r_2. Take one independently labelled Gibbs draw per row of `record` at the specified `inverse_temperature` using a single frozen sample. Every row contains `path`, a list of four integer labels, and `lower` and `upper`, each a list of two numbers or nulls for unbounded sides. Each r component must lie in the corresponding interval for that row. For every pair of rows, the number of shared initial edges in their draws must equal the longest matching initial segment of their paths. For example [2,3,1,7] and [2,3,9,7] prescribe two shared edges. A row may use the same leaf as another row, in which case both interval requirements apply to that leaf's two quantities. Return the joint probability of the entire labelled event. Form Gibbs probabilities in each frozen sample before averaging over samples and taking N to infinity. All bridge records use an inverse temperature larger than sqrt(2*inner_ratio)/gain over the fit range.

A `bridge_record` input contains `observable`, `variances`, `root_entropy`, `inner_ratio`, `detuning`, `inverse_temperature`, `field`, and `record`, all at the top level. For the separate `energy_density` experiment, omit `field` and `record` and treat the field as zero. Return the disorder-averaged Gibbs mean energy per N at the specified high-temperature inverse temperature in the limit.

Fit the shared gain in [0.65,1.34]. Each JSON calibration record contains `input`, `value`, and positive Gaussian standard deviation `sigma`. Implement `Model.fit(records)` to return self and store `self.gain`. Implement `Model.predict(experiments)` to return one one-dimensional NumPy array of predictions in input order, including the empty array for empty input.
