# Tagged particle in an annihilating gas

A stationary tagged point particle is inserted at the origin of an infinite
one-dimensional line. Other point particles start from an independent
homogeneous Poisson process of known density rho. Independently for each
background particle, its velocity is -v, 0, or +v with probabilities
(1-p)/2, p, and (1-p)/2. The tagged particle is always stationary, regardless
of p. The common unknown positive speed v lies in [0.8,1.3] mm/s.

Particles keep their velocities until a collision. Whenever any two
particles meet, both disappear immediately and cannot collide again.
This rule applies to collisions between background particles, including
stationary ones, as well as to collisions involving the tagged particle.
Particles exert no forces before contact. They are not reflected, replaced,
or allowed to pass through one another at contact. The continuous Poisson
positions make simultaneous triple collisions have probability zero.

The apparatus measures the probability S(t) that the tagged particle still
exists at a finite observation time t. Every realization redraws the entire
initial Poisson configuration and all velocity marks. This probability is
an ensemble average at the stated time, not an infinite-time limit. There
are no boundaries, later injection, diffusion, or external forces.

An experiment is a dictionary with `density`=rho in [0.2,1.2] mm^-1,
`stationary_fraction`=p in [0,1], and `time`=t in [0,60] s.
All inputs except v are known. `data/calibration.json` contains independent
noisy measurements `{ "input": experiment, "value": S, "sigma": sigma }`.
Each sigma is the known standard deviation of additive Gaussian readout
noise. Calibration uses p=0, at several densities and observation times.
The tagged particle remains stationary in these calibration experiments.

Implement `Model.fit(records)` to estimate the shared speed, store it as
`model.speed`, and return `self`. Implement `Model.predict(experiments)` to
return a finite NumPy array of shape `(len(experiments),)` in the same order.
Predictions must cover the full stated domain. The supplied response is a
starting approximation and may be changed. Run `pytest -q test_public.py`
for the visible interface and calibration check.
