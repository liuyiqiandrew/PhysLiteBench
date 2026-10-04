# Vacuum transition rates on stationary trajectories

A pointlike two-level Unruh-DeWitt detector couples linearly to a real massless scalar field in the Minkowski vacuum of 3+1 dimensional spacetime. Use hbar=c=1 and the canonical scalar action (1/2) integral [(partial_t phi)^2-|grad phi|^2] d^4x. The detector's positive energy gap `gap` is measured in its proper-time rest frame. Its interaction Hamiltonian in proper time tau is lambda*mu(tau)*phi(x(tau)), with squared transition matrix element |<excited|mu(0)|ground>|^2=1. The only unknown is the dimensionless coupling lambda^2, stored as `coupling`, in [0.006,0.02]. It is the same in every preparation. No other field or decay channel couples to the detector.

The detector follows one of two externally maintained timelike trajectories. In the following coordinates, the first entry is Minkowski time and tau is proper time. The known positive `acceleration` a is the magnitude of proper acceleration.

* `trajectory="cusp"`: x(tau)=(tau+a^2*tau^3/6, a*tau^2/2, a^2*tau^3/6, 0).
* `trajectory="circle"`: x(tau)=(gamma*tau, R*cos(gamma*v*tau/R), R*sin(gamma*v*tau/R), 0), where v=`speed`, gamma=1/sqrt(1-v^2), and R=gamma^2*v^2/a.

Spacetime is unbounded, with no reflecting boundaries. The field is the same inertial vacuum for both trajectories, defined by positive-frequency modes with respect to Minkowski time. Neglect detector recoil and backreaction on the field or prescribed motion. The external apparatus supplies the force needed to maintain the trajectory. There is no additional thermal bath. The field and motion are stationary along each trajectory.

The readout is either `excitation`, the ground-to-excited transition rate, or `deexcitation`, the excited-to-ground transition rate, per unit proper time. Use the leading order in lambda^2, extracting the stationary rate from long, smoothly switched interaction windows. Transients from switching and changes in level occupation are excluded from this weak-coupling rate measurement. Each preparation starts the detector in the level corresponding to its requested readout. The rates are not laboratory-time rates.

Each input contains `trajectory`, `acceleration` in [0.6,1.8], `gap` in [0.08,2.5], and `readout`. Circle inputs additionally contain `speed` in [0.3,0.85]. Cusp inputs have no speed setting. Accelerations and gaps have inverse-time units. All controls are exact. Calibration uses the cusp trajectory at several accelerations and gaps, observing both transitions. Each record in `data/calibration.json` contains `input`, measured rate `value`, and independent Gaussian instrument standard deviation `sigma`, fixed independently of lambda.

Implement `Model.fit(records)`, storing the fitted scalar `coupling` and returning self. `predict(experiments)` must return a finite NumPy array of shape `(len(experiments),)` in input order. Any implementation satisfying this interface may be used.
