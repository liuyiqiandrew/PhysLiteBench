# Two trapped beads

Two identical spherical beads of radius 0.5 micrometers are suspended in an unbounded incompressible Newtonian liquid at 300 K. Their mean centers lie on the x axis. Only their small longitudinal displacements x1,x2 are measured. The liquid has an unknown dynamic viscosity `viscosity` in [0.0005, 0.003] Pa s, constant across experiments. Boltzmann's constant is 1.380649e-23 J/K.

Use the overdamped, zero Reynolds number, point-force Oseen approximation. Keep the mobility tensor at the specified center separation, neglect wall effects, inertial memory, and all corrections beyond leading Oseen order. The surrounding fluid is the sole thermal bath at 300 K. There is no external random forcing, bead interaction potential, or mechanical connection between the traps. This prescribed constant-mobility approximation is the model to solve; do not add higher-order hydrodynamic terms.

Each bead has a harmonic trap of stiffness 0.3 times its `factors` entry in pN/micrometer. Separation is 5 to 15 micrometers, and factors are 0.6 to 1.6. With positions in micrometers and forces in pN, the mobility in micrometers/(pN s) has diagonal 1/(6*pi*viscosity*0.5) and off-diagonal 1/(4*pi*viscosity*separation). The deterministic velocity is mobility times the two-vector of trap forces.

Every run starts with both bead positions pinned exactly at x1=x2=0 by ideal clamps. At time zero the clamps release and the trap centers are set to `shift` (each -0.15 to 0.15 micrometers). The fluid is already equilibrated; all observations are in the overdamped regime after neglected inertial transients. Initial position covariance is zero. The bath temperature and hydrodynamic approximation remain unchanged after release. Times are 0 to 0.5 seconds.

The `measurement` is either `mean` or `covariance`. For `mean`, return the ensemble mean displacement of `bead` (0 or 1), in micrometers. For `covariance`, return Cov(x[pair[0]], x[pair[1]]) in square micrometers, with `pair` entries 0 or 1. This covariance is centered about the ensemble means. Every input includes `measurement`, `bead`, `pair`, `factors`, `separation`, `shift`, and `time`; unused selector fields do not affect the result. Bead labels retain their initial ordering throughout each run.

The supplied measurements use mean displacement. Their independent errors are Gaussian with the stated standard deviations. They identify the viscosity; use the same fitted viscosity for every prediction. Other allowed measurements use the same apparatus.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Expose the fitted scalar as `Model.viscosity`. Calibration records in `data/calibration.json` have keys `input`, `value`, and `sigma`. The starter supplies an evolution calculation, but its physical assumptions need to be checked.
