# Kinetic excursion aperture

A particle launches from a plane at position 0 and speed `launch` into the region X > 0. Its velocity satisfies dV = √(2D) dW and its position satisfies dX = V dt. W is standard Brownian motion and D is constant throughout the experiment. The plane records the first return after launch. All quantities below use the same laboratory units.

Fit D from the `free` calibration readings. With the plane removed, a reading gives the mean squared velocity change over the specified `lag`. Calibration data are rounded to eight decimal places. The fitted model must expose D as `diffusivity`.

Each camera shot specifies an outward launch speed u = `launch`, an inward arrival speed v = `incoming`, two durations [s, r] = `times`, and a measured position x = `position` at time s. The final time is T = s + r and the final state is (0, −v). Interpret continuous gates as densities, or equivalently as the limit of narrow bins in the quantities specified. First-return detections are binned in arrival time and inward speed. The camera has three operating modes.

- **clear** runs in the absence of a physical wall. It selects trajectories by the final position and velocity and by the state (x, `velocity`) at s. It reports the fraction which remained in X > 0 throughout 0 < t < T. There is no kick.
- **direction** runs with an absorbing plane. It selects arrivals whose first return is at T with speed v and whose position was x at s. It reports the probability that V(s) exceeded `cut` among the selected arrivals. There is no kick.
- **pulse** also runs with an absorbing plane. At time s an independent fair coin instantaneously adds either +`kick` or −`kick` to the velocity. Position is continuous. The camera keeps shots with position x and pre-kick velocity above `cut` at s, whose first return is at T with speed v. It reports the fraction of these selected shots that received the positive kick.

Inputs have 0.75 ≤ D ≤ 1.12, 0.37 ≤ u ≤ 0.95, 0.35 ≤ v ≤ 0.9, 0.50 ≤ s,r ≤ 1.15, and 0.11 ≤ x ≤ 0.31. The velocity setting is between −0.6 and 0.5, the cut between −0.22 and 0.24 and the kick magnitude between 0 and 1. Calibration lags lie between 0.1 and 1.3.

For example, take D = 0.83, u = 0.48, v = 0.55, [s,r] = [0.85,0.76] and x = 0.16. With velocity 0.20, `clear` reads 0.85410. With cut 0.04, `direction` reads 0.44341. Adding a kick of magnitude 0.85, with the same cut, gives `pulse` 0.12897. If the labels carry a zero kick, half of any selected arrivals have the positive label.

Implement `Model.fit(records)` and `Model.predict(inputs)` in `/app/model.py`. Fit receives dictionaries with `input` and `value` fields and returns the fitted instance. Predict receives a list of query dictionaries, including empty lists, and returns a finite real NumPy array of shape `(len(inputs),)`. The maximum absolute error must be below 0.000002 on calibration and below 0.006 on each camera mode. The supplied model fits calibration and implements a free Gaussian bridge approximation. You may use NumPy and SciPy.
