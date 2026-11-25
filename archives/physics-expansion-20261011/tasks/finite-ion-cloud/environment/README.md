# A finite ionic cloud in two dimensions

A rigid impenetrable circular disk of radius a has uniform positive charge
+N*q on its rim. Exactly N mobile classical point ions, each of charge -q,
are confined to the planar annulus a<r<R by an uncharged hard outer circle.
Here N is 1, 2, 3, or 4. The central disk is fixed; it supplies no
translational ideal-gas pressure. No ions enter either wall, leave the
annulus, or change charge. There is no added salt or particle reservoir.

Use genuine two-dimensional logarithmic electrostatics in a homogeneous
dielectric. There are no dielectric images or interactions other than the
stated electrostatics and hard walls. This is a finite-particle canonical
Gibbs ensemble at temperature T, retaining all ion-ion interactions.
It is not a three-dimensional 1/r gas or a Poisson-Boltzmann mean-field
specification. Coincident same-sign ions repel; the finite disk radius
prevents an ion from reaching the central charge singularity.

For clarity, the configurational energy convention is

    U/(k_B*T) = Gamma * [N*sum_i log(r_i/a)
                         - sum_(i<j) log(|r_i-r_j|/a)],
    Gamma = u*g.

The positions in the pair distance are two-dimensional vectors. Natural
logarithms are used. Additive constants independent of R have no effect.
The known multiplier u=T_ref/T is dimensionless, and the unknown positive
reference coupling g lies in [0.7,1.4], shared by all preparations. Both
Gamma and g are dimensionless. Radii are measured in nm.

At each finite R, let F_R be the canonical free energy. Define the outer
mechanical pressure as P_R=-dF_R/dA, with A=pi*(R^2-a^2), holding N, a,
Gamma, and T fixed during this derivative. The mean mobile ion number
density is nbar_R=N/A. The reported dimensionless quantity is

    phi = lim_(R/a -> infinity) P_R/(k_B*T*nbar_R).

Take equilibrium at each finite R first, then this limit with finite N,
a, and Gamma fixed. No large-N limit is taken. The radii of individual
ions are not otherwise constrained inside the annulus.

An experiment is a dictionary with integer `ion_count` N in {1,2,3,4},
`temperature_multiplier` u in [0.1,1.8], and `disk_radius` a in [0.5,2] nm.
Predictions must cover the full domain, including one-ion preparations.

`data/calibration.json` contains independent records with an `input`
experiment, a noisy `value` of phi, and its known additive Gaussian
standard deviation `sigma`. Calibration uses u<=0.3.

Implement `Model.fit(records)`, returning self and setting `self.coupling`
to the fitted g. Implement `Model.predict(experiments)`, returning a finite
one-dimensional NumPy array with one phi per experiment, in input order.
Prediction inputs contain no measured value or sigma. NumPy and SciPy are
available. The supplied response is a starting model and may be changed;
preserve these methods and the parameter attribute.
