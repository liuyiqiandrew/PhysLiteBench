# A finite ionic cell in two dimensions

A rigid impenetrable circular disk of radius a carries uniform positive
charge +N*q on its rim. Exactly N mobile classical point ions, each of
charge -q, occupy the planar annulus a<r<R. Here N is 1, 2, or 3. The outer
circle is an uncharged hard wall. The central disk is fixed and supplies
no translational ideal-gas pressure. No ions enter either wall, leave the
annulus, or change charge. There is no salt or particle reservoir.

Use genuine two-dimensional logarithmic electrostatics in a homogeneous
dielectric, with no dielectric images. The only interactions are the
stated electrostatics and hard walls. The finite-particle canonical Gibbs
ensemble retains all ion-ion interactions. The configurational energy is

    U/(k_B*T) = Gamma * [N*sum_i log(r_i/a)
                         - sum_(i<j) log(|r_i-r_j|/a)],
    Gamma = u*g.

The positions in the pair distance are two-dimensional vectors. Natural
logarithms are used. Same-sign point ions repel at coincidence; the finite
disk radius prevents approach to the central charge singularity. The known
dimensionless multiplier u=T_ref/T sets the temperature. The unknown
reference coupling g is dimensionless, lies in [0.7,1.4], and is shared by
all preparations. Radii are measured in nm.

Let F be the canonical free energy at the specified finite outer radius R.
The outer pressure is P=-dF/dA, where A=pi*(R^2-a^2), holding N, a, Gamma
and T fixed during this derivative. With mean ion density nbar=N/A, the
measured dimensionless osmotic coefficient is

    phi = P/(k_B*T*nbar).

R/a is finite in every experiment. No dilution or large-N limit is taken.
Ions are otherwise free to explore the annulus in thermal equilibrium.

Each input dictionary contains integer `ion_count` N in {1,2,3},
`temperature_multiplier` u in [0.1,3.0], `radius_ratio` R/a in [1.25,12],
and `disk_radius` a in [0.5,2] nm. Predictions cover the full domain.

`data/calibration.json` gives independent noisy records with an `input`,
measured `value` of phi, and known additive Gaussian standard deviation
`sigma`. Calibration uses one-ion and two-ion preparations.

Implement `Model.fit(records)`, returning self and setting `self.coupling`
to fitted g. Implement `Model.predict(experiments)`, returning a finite
one-dimensional NumPy array with one phi per experiment, in input order.
Prediction inputs contain no measured value or sigma. NumPy and SciPy are
available. The supplied response is a starting model and may be changed;
preserve these methods and the parameter attribute.
