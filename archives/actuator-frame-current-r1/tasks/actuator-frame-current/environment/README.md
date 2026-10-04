# Persistent actuator forces on a periodic plane

A dilute noninteracting tracer moves on a square periodic plane of side 2*pi. Its position is (x,y), with unwrapped coordinates retained for displacement measurements. There are no reflecting walls. Length, time and force units set the spatial period to 2*pi and the isotropic drag coefficient to 1. The following externally driven classical model is exact over the stated settings:

    dx = v_x dt,  dy = v_y dt
    m dv = [-v - grad U(x) + push*e_y + B(x) eta] dt
    tau d eta = -eta dt + sqrt(2D) dW
    tau = ratio*m,  U(x) = potential*cos(x).

Here v and eta each have two components. W_1 and W_2 are independent standard Wiener processes. D, stored as `noise_strength`, is the sole unknown common parameter and lies in [.55,1.1]. The drag is deterministic; there is no additional thermal or measurement back-action force in this ideal model. The apparatus continuously removes the dissipated energy without changing the controls.

There are two independent external actuator signals eta_1(t) and eta_2(t). Each signal keeps its identity and evolves by the equation above regardless of the tracer position. The force vectors to which these two signals couple are the respective columns of the specified matrix

    B(x) = amplitude(x) * [[cos(angle(x)), -sin(angle(x))],
                          [sin(angle(x)),  cos(angle(x))]]
    amplitude(x) = 1 + contrast*cos(x)
    angle(x) = winding*x + twist*sin(x).

Thus the force direction and strength of each fixed actuator channel are controlled by the tracer's current x coordinate. The position dependence is instantaneous and the actuator adds no further force or delay. `winding` is an integer, so B is periodic even when its angle winds around the ring.

For every positive m the entire tracer and actuator process first reaches its stationary state on the periodic plane. Two detector modes are available. `observable="cosine"` returns the stationary ensemble mean of cos(x). `observable="current"` returns the stationary mean unwrapped y displacement per unit time, equivalently the ensemble mean of v_y at finite m. The requested prediction is the limit of that measurement as m tends to zero while tau=ratio*m and all other controls stay fixed. The steady preparation precedes this joint limit. The uniform y marginal is included in the stationary preparation.

Every experiment contains `ratio` in [.25,4], `contrast` in [0,.65], `potential` in [0,1.5], integer `winding` in [-2,2], `twist` in [-1,1], `push` in [-.4,.4], and `observable` as defined above. All controls are exact. `data/calibration.json` contains stationary cosine measurements with winding=twist=0 at varied potential, amplitude contrast and time ratio. Each record has `input`, measured `value` and independent Gaussian uncertainty `sigma`. The fixed instrument uncertainty is .0006, independent of D and of the measured response.

Implement `Model.fit(records)`, returning self and storing `noise_strength`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order. Use the same fitted noise strength for all preparations.
