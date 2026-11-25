# Mobile traps on a line

One point target starts at the origin on the infinite one-dimensional line.
Its position satisfies dX=sqrt(2*D_A)*dW_A. Point traps start at the points of
an independent homogeneous Poisson process of density rho and move as
dY_i=sqrt(2*D_B)*dW_i. All Brownian drivers are independent. Traps pass through
one another without interaction. The target is removed at its first contact
with any trap. There are no boundaries, trap creation, or resetting.

For each realization, the initial Poisson trap positions and every Brownian
path are freshly drawn. S(t) is the probability that the target has survived
until t, averaged over this entire ensemble. The readout is

    C = lim_{t -> infinity} -log(S(t))/sqrt(t).

The average defining S is taken before its logarithm. The infinite line is
defined before the time limit. This readout is the leading asymptotic
coefficient, not a measurement at a specified finite time.

Lengths are in mm and time in seconds. Density is in mm^-1, diffusivities in
mm^2/s, and C in s^-1/2. The common unknown trap diffusivity D_B lies in
[0.5, 1.5] mm^2/s and is strictly positive. Target diffusivity D_A is known
and held fixed for each experiment.

Each experiment is a dictionary with `density` in [0.1, 1.0] and
`target_diffusivity` in [0.0, 4.0]. `data/calibration.json` contains independent
noisy measurements `{ "input": experiment, "value": C, "sigma": sigma }`.
Every sigma is the known standard deviation of additive Gaussian readout
noise. Calibration uses immobile targets at several trap densities.

Implement `Model.fit(records)` to estimate the shared trap diffusivity,
store it as `model.trap_diffusivity`, and return `self`. Implement
`Model.predict(experiments)` to return a finite NumPy array of shape
`(len(experiments),)`, in the same order as the supplied dictionaries.
Predictions must cover the full stated domain. The supplied response is a
starting approximation and may be changed. Run `pytest -q test_public.py`
for the visible interface and calibration check.
