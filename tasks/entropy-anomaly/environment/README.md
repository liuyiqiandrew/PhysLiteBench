# Calorimetry of a charged Brownian particle

A particle moves on a two-dimensional periodic square: both x and y have period 2*pi. It has exactly two velocity components vx,vy, with the same positive mass m, and no other degrees of freedom. Set k_B=1 and use the following reduced length, energy and time units. The underdamped equations are

    dx = vx dt,                    dy = vy dt,
    m dvx = (force_x - friction*vx + magnetic*vy) dt
            + sqrt(2*friction*T(x)) dWx,
    m dvy = (force_y - drag_ratio*friction*vy - magnetic*vx) dt
            + sqrt(2*drag_ratio*friction*T(x)) dWy.

Wx and Wy are independent standard real Wiener processes. The constant friction coefficient is unknown and lies between 0.7 and 1.6. The same coefficient applies to every experiment. The known `drag_ratio` is between 0.25 and 8, and the known constant forces `force_x` and `force_y` are each between -1.5 and 1.5. The drag tensor is fixed along the stated laboratory axes. A static uniform perpendicular magnetic field produces the stated Lorentz force. Its signed known coefficient `magnetic` (charge times field, in the same units as friction) lies between -5 and 5. The magnetic field is externally fixed and does no work on the particle. There is no additional potential, force or noise. The forces may do work as the particle winds around the periodic square.

The externally maintained local bath temperature depends only on x:

    T(x) = temperature * (1 + contrast*cos(wavenumber*x)).

The known `temperature` is between 0.8 and 1.4, `contrast` is between 0 and 0.65, and `wavenumber` is an integer 1, 2 or 3. The bath is locally thermal, with the drag and noise specified above. It is not changed by the particle. No heat flows from maintaining the external temperature profile are included in the measurement. The stationary position distribution is uniform in y and normalized over the full periodic square.

The measured quantity is the stationary mean rate of entropy delivered by this particle to its local bath, obtained by calorimetry. At finite mass, write gamma_x=friction and gamma_y=drag_ratio*friction. Heat delivered to the bath along a trajectory is

    dQ_bath = sum over i=x,y of
              (gamma_i*vi dt - sqrt(2*gamma_i*T(x)) dWi) circ vi,
    dS_bath = dQ_bath / T(x),

where `circ` is the Stratonovich product. Heat entering the bath is positive. The measurement is defined first in the stationary state at fixed positive m, then as the long-time mean of S_bath divided by elapsed time, and finally in the limit m -> 0. Return this limiting rate per particle. The mass is not a fitted parameter or an input. The order of these limits is part of the apparatus definition. Each experiment is independently brought to its stationary state before measurement.

Every input contains `force_x`, `force_y`, `drag_ratio`, `magnetic`, `temperature`, `contrast`, and `wavenumber`. All settings are exact. Calibration uses zero magnetic field, known unequal as well as equal drag ratios, nonuniform temperature profiles, and known forces. These measurements identify the one friction coefficient. Their errors are independent Gaussian errors with the stated standard deviations.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Expose `Model.friction`. Records in `data/calibration.json` have keys `input`, `value`, and `sigma`.
