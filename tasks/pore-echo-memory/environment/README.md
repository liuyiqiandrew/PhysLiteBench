# Echoes from a four-site pore

Each pore has four equivalent traps numbered clockwise around a ring. Trap j carries an angular frequency offset ω_j in radians/ms. Fabrication draws the four offsets independently from a normal distribution with zero mean and common standard deviation width in [0.55, 1.45] radians/ms. A pore keeps all four offsets throughout every scan. We average over a large batch of pores with equal weight per pore.

Each scan loads many independent spins at a specified trap (start = 0, 1, 2 or 3). Every spin starts with unit transverse magnetization on the real axis. During each exposure its trap is sealed for 1 ms. Its transverse phase advances by gain × sign × ω_j × 1 ms at its current site j. The signs list gives one +1 or −1 for each exposure (1 to 9 exposures). Between successive exposures the offsets are switched off and the ring gates open briefly. For each opening, hops gives the probability p in [0,1] of moving one site. Left and right are equally likely; otherwise the spin stays put. Choices are independent between spins and openings. There is no gate before the first exposure. The offsets return to their original values after each opening. All other evolution and relaxation are negligible.

The coil averages the complex transverse magnetization exp(i phase) over the spins at the end of a scan. Take the limit of infinitely many spins per pore. Denote the complex reading by M. The receiver supports two modes:

- An echo returns the real part of the batch mean of M.
- For a covariance the instrument performs two scans on each pore with freshly loaded spins. It subtracts each scan's own complex batch mean from that scan's reading, multiplies the first centered reading by the complex conjugate of the second centered reading, then reports the real part averaged over pores. Sites and gate settings in both scans refer to the same physical ring. The two scans use the same gain.

Calibration records contain scan (start, signs, hops) and curvature. Curvature is minus the second derivative of the batch echo with respect to dimensionless gain, at gain = 0, rounded to seven decimals. Prediction inputs contain mode ("echo" or "covariance"), gain in [0,2], and scan. Covariance inputs also contain a second scan under partner. All reported means and covariances are dimensionless.

Model.fit(records) stores the inferred width in self.width and returns self. Model.predict_calibration(records) returns the calibrated curvature for each input. Model.predict(experiments) returns the requested outputs. Both prediction methods return a finite 1D NumPy array in input order. The arrays may be empty.

Prediction accuracy is assessed to 0.002 in absolute signal units. Calibration accuracy is assessed to 0.00001 in curvature units.
