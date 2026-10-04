# Entropy records in a nonuniform bath

A dilute classical particle moves on a one-dimensional ring of circumference `2*pi`. Its unwrapped position `X` and velocity `V` obey

    dX = V dt
    m dV = -gamma V dt + sqrt(2 gamma T(X)) dW.

`W` is a standard Wiener process. There is one velocity degree of freedom, no applied force, no potential, and no other bath. The friction `gamma` is constant in space, time and mass. Boltzmann's constant is one. Length, time and energy use fixed apparatus units. The model is the stated classical Markov Langevin model at every positive mass.

The bath temperature is maintained at

    T(X) = temperature * (1 + contrast*cos(wavenumber*X)).

The particle does not alter this temperature profile. The external heat used to maintain it is excluded from the records. Each experiment is prepared in the stationary joint position-velocity state at positive mass.

A distributed calorimeter records the heat delivered by the particle to the local bath, divided by that bath's local temperature. The integrated entropy record over duration `t` is

    S_m(t) = -integral_0^t [m V_s/T(X_s)] circ dV_s,

where `circ` is the Stratonovich integral. Independent stationary repetitions determine the following rates:

    mean:      limit_(m -> 0+) limit_(t -> infinity) E[S_m(t)]/t
    variance:  limit_(m -> 0+) limit_(t -> infinity) Var[S_m(t)]/t.

The variance is centered. In each limit the temperature profile and friction are fixed. Predictions concern these rates, not a finite-duration entropy record.

Inputs contain `temperature` in `[.8,1.4]`, `contrast` in `[0,.65]`, integer `wavenumber` in `{1,2,3}`, and `statistic`, either `mean` or `variance`. The only unknown is `gamma` in `[.7,1.6]`.

`data/calibration.json` is a list of independent measurements with fields `input`, `value`, and `sigma`. The standard uncertainty is a fixed instrument value, `sigma=.0001`, independent of the unknown friction and the measured value.

Implement `Model.fit(records)` and `Model.predict(experiments)`. `fit` must return `self` and expose the fitted friction as `self.friction`. `predict` must return a finite one-dimensional NumPy array in input order. Preserve this interface; any numerical implementation may be used.
