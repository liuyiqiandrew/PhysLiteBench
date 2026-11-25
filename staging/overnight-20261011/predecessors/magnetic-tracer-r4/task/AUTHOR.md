# Magnetic tracer R4

This revision retains the R3 finite-mass apparatus and calibration and adds an explicitly specified preparation tau=m*kappa. The two bath temperatures may differ. Position has mobility (gamma+zeta-BJ)^-1 and diffusion 2(gamma*T+zeta*Tb) M M^T. Its limiting covariance alone does not determine the mechanical area.

For fast variables y=(sqrt(m)*v,sqrt(m)*f), define A=[[-gamma I+BJ,I],[-zeta I/kappa,-I/kappa]], Q=diag(2gamma*T I,2zeta*Tb I/kappa²). Their stationary covariance solves A S+S A^T+Q=0. The one-sided velocity covariance integral is Z=(-S A^-T)[:2,:2]. The mechanical area rate contains Z_xy-Z_yx in addition to the position-drift contribution. Replacing this by the anomaly of one bath at the effective position temperature loses the kinetic timescale and nonequilibrium circulation. In equilibrium the extra stationary circulation vanishes. At zero memory drag it reduces to the ordinary Lorentz area anomaly.

The independent reference solves the original six-dimensional finite-mass OU process with tau=m*kappa at each of three masses before extrapolating the actually defined observable. Calibration is unchanged. The completed shortcut is exact for fixed-time contacts and for position covariance, but approximates simultaneous fast baths by a single white thermal bath. All results are retained; a fresh three-trial batch is needed before qualification.
