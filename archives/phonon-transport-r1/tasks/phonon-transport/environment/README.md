# Angular phonon transport

A homogeneous three-dimensional insulating crystal has one isotropic acoustic
phonon branch with dispersion omega(q)=v*|q| and speed v=1 mm/s. Spatial
variation is only along x, but propagation directions cover the full sphere.
Let mu be the cosine of a phonon's direction with +x. All preparations are
axisymmetric around x. The crystal is infinite, with no boundaries or imposed
flow. Work in linear response about a uniform Bose equilibrium with zero drift.

The two independent collision classes are described by local relaxation
models. Resistive events exchange crystal momentum with the stationary lattice
but conserve phonon energy. Normal events conserve both phonon energy and all
components of crystal momentum. Each class relaxes toward the local
maximum-entropy Bose distribution with exactly the energy and, when conserved
by that class, momentum of the instantaneous local phonon distribution.
Phonon number is not conserved. Linearize both local target distributions
about the stated equilibrium. The relaxation rates are independent of phonon
frequency and direction. The unknown resistive rate is in [0.15,0.6] s^-1.
Each experiment supplies a known normal_rate in [0,3] s^-1. There are no other
collisions, heat sources, or interactions. Both collision classes act
throughout each experiment.

Let f(x,mu,t) denote the frequency-integrated angular energy perturbation
divided by the isotropic equilibrium angular energy density. This normalization
is fixed. The angular average is one-half the integral over mu from -1 to 1.
At time zero,

    f(x,mu,0) = [initial[0]*P0(mu) + initial[1]*P1(mu)
                 + initial[2]*P2(mu)] * cos(wavenumber*x),
    P0=1, P1=mu, P2=(3*mu^2-1)/2.

The real initial coefficients obey sum(abs(initial))<=0.2. The wavenumber is
in [0,3] mm^-1 and time in [0,8] s. Free propagation transports angular energy
at velocity v*mu along x. The material remains in the stated linear-response
regime. No closure in angular moments is imposed.

The detector selects moment 0,1,or2 and reports the cosine or sine coefficient
of the spatial Fourier mode in the angular moment

    M_l(x,t) = (1/2) integral[-1,1] P_l(mu)*f(x,mu,t) dmu.

For nonzero wavenumber a coefficient is twice the spatial average against
the corresponding cosine or sine over a period. At wavenumber zero, the cosine
coefficient is the spatially uniform value and the sine coefficient is zero.
Moment 0 measures normalized energy and v times moment 1 measures its flux.
Every input has initial, wavenumber, normal_rate, time, moment, and quadrature
("cosine" or "sine"). Outputs are dimensionless perturbation coefficients.

Calibration observes the decay of spatially uniform P2 preparations with
several nonzero normal rates. It identifies the one common resistive rate.
Settings are exact; independent Gaussian errors have the recorded sigma.
Implement Model.fit(records), returning self and storing resistive_rate in
s^-1, and Model.predict(experiments), returning a finite NumPy array of shape
(len(experiments),). Read input/value/sigma records from data/calibration.json.
Run python -m pytest -q test_public.py.
