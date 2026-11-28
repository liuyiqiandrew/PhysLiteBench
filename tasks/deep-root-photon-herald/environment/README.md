# Deep root photon herald

A Gaussian tree has floor(exp(N ratio*v_F)) families, each with floor(exp(N ratio*v_L)) sites. Family energies X and site increments Y are independent centered Gaussians of variances N gain² v_F and N gain² v_L. Each family shares its X. Set a=gain*sqrt(2 ratio), c=sqrt(2 ratio)/gain, x=(X+Na v_F)/sqrt(N), and y=(Y+Na v_L)/sqrt(N).

A detector first searches for families with X+Na v_F+`depth`*sqrt(N) in `root_window`. A shot records anything only if this finds exactly one family. In that family retain each individual site independently with probability min(1,`retention`*sqrt(N)*exp(−c*depth*sqrt(N))). Use the same independent retention marks in both settings. All sites of other families remain visible in both settings.

In setting j a retained site of the selected family is visible only if its raw line energy t=X+Y+Na(v_F+v_L) is in that setting's `gate`. On all visible sites the Gibbs energy is X+Y−fields[0]*x−fields[1]*y and the inverse temperature is c+window/sqrt(N). `settings` contains two objects each with `gate` (an interval), `fields` (two numbers) and `window` (one number). The centering and detector use the actual gain.

For `herald_return`, require exactly one visible site of the selected family in setting 0. Make one independent Gibbs draw in each setting and require the same actual site of that family, with t in `line_window`. If `outside` is true, make one further independent Gibbs draw in setting 1 and require a different family. Return the limit of sqrt(N)*exp(c*depth*sqrt(N)) times the probability of the entire record averaged over original shots. Apply all requirements and Gibbs normalizations at finite N. An empty visible population cannot give a record.

Inputs are the named fields and `observable` and `variances`=[v_F,v_L]. Variances are in [.51,1.23], ratio in [.33,.74], depth in [.22,1.16], retention in [.12,1.07], fields and window in [−1.02,1.02], and all interval endpoints are increasing in [−1.48,1.48].

For `warm_energy`, turn off the fields and all visibility restrictions. Use just observable, variances, ratio and inverse_temperature in [.08,.34], and return the limiting mean X+Y per N. Fit gain in [.69,1.31]. Calibration records have input, value and positive sigma. Implement `Model.fit(records)` storing `self.gain` and returning self, and `Model.predict(experiments)` returning one-dimensional NumPy predictions in input order including an empty array for empty input.
