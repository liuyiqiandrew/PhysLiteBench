# Phase-tagged Bell return

At size N there are floor(exp(N ratio*v_F)) families with floor(exp(N ratio*v_L)) sites each. The family energy X and site increment Y are independent centered Gaussians with variances N gain² v_F and N gain² v_L. Each family shares its own X. Each site also has an independent quenched sign m, positive with probability `positive_fraction`. Two canonical preparations use the identical sample of energies and signs.

Put a=gain*sqrt(2 ratio), c=sqrt(2 ratio)/gain, x=(X+Na v_F)/sqrt(N), and y=(Y+Na v_L)/sqrt(N). Each site has a spin one-half space. Setting j has the direct sum of site Hamiltonians

    (X+Y−fields[0]*x−fields[1]*y)*I + (internal + m*disorder) dot Pauli,

with the standard x,y,z Pauli matrices and inverse temperature c+window/sqrt(N). `settings` contains two objects, each with `fields` (two numbers), `window` (one number), `internal` and `disorder` (three numbers each). The apparatus uses the actual gain in the centering and temperatures.

For `bell_return`, prepare the two settings independently. Measure each actual site without measuring its spin. Require the same site, its x in `bridge_band`, and its unmodified displacement X+Y+Na(v_F+v_L) in `line_window`. Apply exp(−i gain `pulse` dot Pauli) to the second retained spin. Measure the rank-one projector onto (|+z,−z>−exp(i (`phase`+`phase_slope`*x)) |−z,+z>)/sqrt(2) on the two retained spins. If `outside` is an object, make a further independent canonical preparation with its `setting` (0 or 1) and require a different family with y in its `band`. Sum over the third spin. If outside is null, omit that draw.

Return the limit of sqrt(N) times the probability of the whole event averaged over original samples. Apply the measurements at finite N before taking the limit. The main inputs are the named fields, `observable` and `variances`=[v_F,v_L]. Variances are in [.49,1.25], ratio in [.31,.77], positive_fraction in [.19,.81], and settings entries and the three pulse components are bounded in magnitude by 1.08. Phase and phase_slope are in [−3.2,3.2]. Interval endpoints are increasing in [−1.66,1.66].

For `warm_energy` calibration use only observable, variances, ratio and inverse_temperature in [.08,.34]. Set the fields to zero and return the limiting mean X+Y per N. Fit gain in [.69,1.31]. Each calibration record supplies input, value and positive sigma. Implement `Model.fit(records)` storing `self.gain` and returning self, and `Model.predict(experiments)` returning a one-dimensional NumPy array in input order, including the empty array for empty input.
