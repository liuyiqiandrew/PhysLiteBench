# Heat exchanged by a Brownian particle in a nonuniform bath

A particle has one position coordinate x, periodic with period 2*pi, and one velocity v. It has no other translational, rotational, or internal degrees of freedom. Set k_B=1 and use the following dimensionless length, energy, and time units. Its underdamped motion at positive mass m is

    dx = v dt,
    m dv = (force - friction*v) dt + sqrt(2*friction*T(x)) dW.

W is a standard real Wiener process. The constant friction coefficient is unknown and lies between 0.7 and 1.6. The same coefficient applies to every experiment. The known constant tangential `force` is between -1.5 and 1.5. It may do work as the particle winds around the periodic coordinate. There is no additional potential, force, or noise.

The externally maintained bath temperature is

    T(x) = temperature * (1 + contrast*cos(wavenumber*x)).

The known `temperature` is between 0.8 and 1.4, `contrast` is between 0 and 0.65, and `wavenumber` is an integer 1, 2, or 3. The bath is locally thermal and is not changed by the particle. No transverse velocity modes or heat flows from maintaining the temperature profile are included in the measurement.

The measured quantity is the stationary mean rate of entropy delivered by this particle to its local bath, obtained by calorimetry. For finite mass, heat delivered to the bath along a trajectory is

    dQ_bath = (friction*v dt - sqrt(2*friction*T(x)) dW) circ v,
    dS_bath = dQ_bath / T(x),

where `circ` denotes the Stratonovich product. Heat entering the bath is positive. The measurement is defined first in the stationary state at a fixed positive m, then as the long-time mean of S_bath divided by elapsed time, and finally in the limit m -> 0. Return this limiting rate per particle. The mass is not a fitted parameter or an input. The order of these limits is part of the apparatus definition. The position distribution is normalized over one period. Every experiment is independently brought to its own stationary state before measurement.

Each input contains `force`, `temperature`, `contrast`, and `wavenumber`. All are exact. Calibration measurements use spatially uniform baths (contrast=0), known nonzero forces, and independent Gaussian errors with standard deviations `sigma`. These measurements identify the friction coefficient.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Expose `Model.friction`. Records in `data/calibration.json` have keys `input`, `value`, and `sigma`.
