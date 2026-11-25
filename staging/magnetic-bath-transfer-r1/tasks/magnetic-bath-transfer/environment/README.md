# A trapped particle with two thermal reservoirs

A classical particle moves in an unbounded two-dimensional plane. Its known positive mass is m. In reduced units with k_B=1, the potential is

    U(x,y) = spring*(x^2+y^2)/2 + coupling*x*y.

An independently maintained thermal reservoir acts on each Cartesian velocity. The full dynamics are

    dx = vx dt,                         dy = vy dt,
    m dvx = (-spring*x - coupling*y - gamma*vx + field*vy) dt
             + sqrt(2*gamma*T_x) dW_x,
    m dvy = (-coupling*x - spring*y - gamma*vy - field*vx) dt
             + sqrt(2*gamma*T_y) dW_y.

W_x and W_y are independent standard real Wiener processes. The signed coefficient `field` is charge times a static perpendicular magnetic field. The reservoirs, trap and field are fixed during each measurement. There are no other forces, interactions or energy exchanges. The harmonic potential and Markov thermal reservoirs are the exact ideal model over the stated parameter range.

Every preparation reaches the unique stationary state of the full position and velocity process. A separate ideal calorimeter measures the energy lost by each reservoir through its stated drag and random-force coupling to the particle. The readout is this stationary mean power for the selected reservoir, with positive sign for energy entering the particle. Reservoir heat along a stochastic trajectory uses the Stratonovich convention. No initial relaxation is included in the measurement.

The unknown common drag coefficient gamma lies in [.7,1.5], and is the same in every experiment. Known inputs are `mass` in [.15,.8], `spring` in [.8,1.8], `coupling` in [.15,.55], `field` in [-1.2,1.2], and `temperatures: [T_x,T_y]`, each in [.5,2]. The selected `bath` is 0 for x or 1 for y. These bounds make the trap strictly confining. All controls and units are exact.

`data/calibration.json` contains independent records with `input`, measured `value`, and Gaussian standard deviation `sigma`. Instrument uncertainty is fixed independently of gamma.

Implement `Model.fit(records)`, returning self and storing gamma as `friction`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order. Run `python -m pytest -q test_public.py`.
