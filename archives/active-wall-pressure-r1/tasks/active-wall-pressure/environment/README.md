# Wall-pressure measurements

A dilute gas of identical self-propelled particles occupies a two-dimensional channel. It is uniform and periodic along y, with two stationary soft wall coatings at x=−L and x=L. Particles do not interact. Each particle has position (x,y) and an orientation angle theta modulo 2*pi, with propulsion direction (cos(theta),sin(theta)). There is no imposed flow or other external force.

Length, time and energy are measured in fixed units L0,t0,E0. Set k_B=1 and translational mobility to L0^2/(E0*t0). The following equations are the exact dimensionless stochastic model:

    dx = [v*cos(theta) - partial_x V(x,theta)] dt + sqrt(2*T) dW_x
    dy = v*sin(theta) dt + sqrt(2*T) dW_y
    dtheta = -(D_r/T)*partial_theta V(x,theta) dt + sqrt(2*D_r) dW_theta

The three Wiener processes are independent. The known rotational diffusion coefficient is D_r=0.8. The propulsion speed v and thermal bath temperature T are held constant during each preparation. Inertia, solvent flow feedback and interactions between particles are absent. The thermal mobility and noise amplitudes in these equations define the ideal model throughout the stated range.

The wall-particle interaction energy is

    s = max(abs(x)-L, 0)
    V(x,theta) = k*s^4/4 + h*s^4/(ell^4+s^4)*sin(theta)^2
    k = 4
    ell = 0.6

Each coating translates rigidly with its support, carrying this entire interaction. The right coating acts at x>L and the left coating at x<−L. Space outside the nominal channel is included; there is no additional hard boundary. The quartic term confines every particle. The system reaches its stationary state before each reading.

The mean number of particles per unit y-length is the same unknown line loading `loading` in every preparation, with `0.8 <= loading <= 1.2`, in units L0^−1. A transducer measures the stationary mean normal force delivered by the particles to the right coating, divided by its y-length. Positive output is force toward +x, in units E0/L0^2. It records the reaction from the specified wall-particle interaction; direct solvent and support loads have been subtracted.

Each experiment supplies these known controls:

- `speed`: v, between 0.9 and 1.8;
- `temperature`: T, between 0.45 and 0.7;
- `half_width`: L, between 1 and 2;
- `alignment`: h, between −1 and 2.

`data/calibration.json` contains independent records of the form `{"input": {...}, "value": ..., "sigma": 0.001}`. The additive Gaussian measurement standard deviation is the fixed instrument value 0.001 in pressure units.

Implement `Model.fit(records)`, returning `self` and storing the fitted common line loading in `self.loading`. Implement `Model.predict(experiments)`, returning a one-dimensional finite NumPy array of pressures in input order. Keep this interface and the stated units.
