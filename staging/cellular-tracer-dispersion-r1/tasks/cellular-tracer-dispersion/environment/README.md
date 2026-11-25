# Tracer spreading in a periodic liquid flow

Dilute, noninteracting point tracers move in an unbounded planar liquid. Their inertia is negligible. The externally maintained, steady liquid velocity is

    u_x(x,y) = a sin(y/L0) + c sin(2y/L0),
    u_y(x,y) = b sin(x/L0) + d sin(2x/L0).

The four amplitudes are prescribed. This velocity is incompressible and periodic with period 2 pi L0 in each direction. External forcing maintains it independently of the tracers. There are no walls, capture, reactions, adsorption or forces on the tracers other than advection and molecular diffusion.

Use L0 as the length unit and t0 as the time unit. Input amplitudes a,b,c,d are in L0/t0. A tracer's laboratory position obeys

    dX = u(X) dt + sqrt(2 D) dW,

where the two components of W are independent standard Brownian motions. The unknown constant molecular diffusivity D is between 0.8 and 1.2 in L0^2/t0 and is common to every experiment. Different tracers have independent molecular noise and do not alter the flow. The stated advection-diffusion model defines the continuum experiment.

Each experiment seeds independent tracers with a phase uniformly distributed over one spatial period cell. Record each tracer's displacement from its own initial laboratory position. Positions and displacements are not reduced modulo the spatial period. All tracers remain in the sample and are counted with equal weight.

For the known detector direction e=(cos(angle),sin(angle)), the measured spreading coefficient is

    lim as t/t0 -> infinity of Var[e dot (X(t)-X(0))] / (2t).

The variance is over the initial phases and molecular trajectories. Return the coefficient in L0^2/t0. The four velocity amplitudes are fixed while this long-time limit is taken.

Each input has numeric keys `a`, `b`, `c`, `d`, `angle`. The first two amplitudes lie in [-8,8], the latter two in [-0.8,0.8], and angle lies in [0,pi] radians. Zero amplitudes are allowed. Calibration records in `data/calibration.json` give `input`, the measured `value`, and its known independent Gaussian standard deviation `sigma`. The fixed instrument uncertainty is 0.003 L0^2/t0.

Implement `Model.fit(records)` returning `self`, storing the fitted molecular diffusivity in `self.diffusivity`. `Model.predict(experiments)` must return a finite one-dimensional NumPy array with one spreading coefficient per input.
