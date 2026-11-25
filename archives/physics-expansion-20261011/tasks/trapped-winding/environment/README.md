# Winding of a trapped Brownian particle

An overdamped point particle moves on the full two-dimensional plane in
the isotropic potential U(x,y)=k*(x^2+y^2)/2. Its exact ideal dynamics are

    dx = -mu*k*x*dt + sqrt(2*mu*T)*dW_x,
    dy = -mu*k*y*dt + sqrt(2*mu*T)*dW_y,

with independent standard Wiener increments. Distance is in mm, time in
seconds, and energy in a fixed unit with k_B=1. Temperature T is in energy
units, stiffness k in energy/mm^2, and mobility mu in mm^2/(energy*second).
The common unknown mu lies in [0.7,1.4]. Each run begins in thermal
equilibrium for its stated k and T, independently of its future noise.
There is no inertia, excluded central disk, reflecting core, or spatial
or temporal observation cutoff. These continuum equations define the
apparatus at all positions and time scales.

A continuous ideal position record determines the real, unwrapped angular
change Theta_t about the origin. It includes every turn, with positive
sign counterclockwise, and starts at Theta_0=0. Equivalently,

    Theta_t = integral_0^t (x o dy - y o dx)/(x^2+y^2),

where o denotes Stratonovich integration. The trajectory almost surely
does not hit the origin. Theta_t is not reduced modulo 2*pi and is not the
principal angle between just the two endpoint positions.

For a known real dimensionless angular probe lambda, the apparatus reports

    rate = -lim_(t->infinity) log(abs(E[exp(i*lambda*Theta_t)]))/t.

The expectation is over independently prepared full trajectories. The
ensemble average is taken before the logarithm, and the output is a rate
in inverse seconds. It is this asymptotic rate, not a finite-time estimate,
that is measured. Each calibration value has separate additive readout noise.

An experiment is a dictionary with `stiffness` k in [0.6,1.8],
`temperature` T in [0.5,1.8], and `probe` lambda in [-2.5,2.5]. The same
unknown mobility applies to all settings. Calibration uses probes +1 and
-1 at several stiffnesses and temperatures. Predict over the full domain.

`data/calibration.json` contains independent noisy readings
`{ "input": experiment, "value": rate, "sigma": sigma }`, where sigma is
the known standard deviation of additive Gaussian readout noise in inverse
seconds. Implement `Model.fit(records)` to estimate mu, store it as
`model.mobility`, and return `self`. `Model.predict(experiments)` must
return a finite NumPy array of shape `(len(experiments),)` in input order.
The supplied response is a starting approximation and may be changed.
Run `pytest -q test_public.py` for the visible interface and calibration check.
