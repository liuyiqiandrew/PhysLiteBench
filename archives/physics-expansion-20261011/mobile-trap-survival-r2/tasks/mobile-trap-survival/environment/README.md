# Driven target among mobile traps

One point target starts at the origin on the infinite one-dimensional line.
It is driven by a constant force F in a bath of noise temperature T. Its overdamped
position obeys

    dX = mu*F*dt + sqrt(2*mu*T)*dW_A.

Boltzmann's constant is one. The common unknown mobility mu lies in [0.8,1.3].
At T=0 this equation specifies deterministic target motion; the trap noise
is controlled separately and remains present.

Point traps start at the points of an independent homogeneous Poisson
process of density rho. Each moves as dY_i=sqrt(2*D_B)*dW_i, with known
D_B>0. All Brownian drivers are independent. Traps pass through one another
without interaction. The target is removed at its first contact with any
trap. There are no boundaries, trap creation, or resetting.

Every realization freshly draws the initial Poisson trap positions and all
paths. S(t) is the probability that the target survives until t, averaged
over this entire ensemble. The readout is the exponential decay rate

    K = lim_{t -> infinity} -log(S(t))/t.

The average defining S is taken before its logarithm. The infinite line is
defined before the time limit. Each force, temperature, density and trap
diffusivity is fixed while this limit is taken. This is an asymptotic rate,
not a measurement at a specified finite time.

Use fixed apparatus units of length, time and energy. Force has units of
energy/length, temperature of energy, mobility of length^2/(energy*time),
and trap diffusivity of length^2/time. The readout has units of inverse time.
An experiment is a dictionary with:

- `density` in [0.1,1.0], in inverse length;
- `force` in [-4.0,4.0];
- `temperature` in [0.0,2.0];
- `trap_diffusivity` in [0.5,1.5].

`data/calibration.json` contains independent noisy measurements
`{ "input": experiment, "value": K, "sigma": sigma }`. Each sigma is the
known standard deviation of additive Gaussian readout noise. Calibration
uses T=0, with nonzero forces and independently diffusing traps.

Implement `Model.fit(records)` to estimate the shared target mobility,
store it as `model.mobility`, and return `self`. Implement
`Model.predict(experiments)` to return a finite NumPy array of shape
`(len(experiments),)`, in the same order as the supplied dictionaries.
Predictions must cover the full stated domain. The supplied response is a
starting approximation and may be changed. Run `pytest -q test_public.py`
for the visible interface and calibration check.
