# Flow above a driven sheet

An infinite anchored sheet has prescribed material motion

    x(X,t) = X,
    y(X,t) = a*cos(k*X-omega*t).

The sheet is extensible: its material points have no horizontal motion. A homogeneous incompressible Newtonian liquid occupies the upper side. Its constant kinematic viscosity nu is unknown. There is no slip on the actual moving sheet. The liquid obeys the incompressible Navier–Stokes equations, with no body force or imposed pressure gradient. There is no distant imposed shear or counterflow. The periodic stationary response has decaying oscillatory velocity and bounded mean velocity.

All reported coefficients are extrapolated to a tending to zero at fixed k, omega and nu. The apparatus first reaches its periodic stationary response. Its pumping readout then takes the far-field limit in height. Spatial heights are measured from the flat reference plane y=0 in the laboratory frame.

At a specified positive height h, the complex first-harmonic coefficients are defined by

    u_x(x,h,t) = a*Re[U(h)*exp(i*(k*x-omega*t))] + O(a²),
    u_y(x,h,t) = a*Re[V(h)*exp(i*(k*x-omega*t))] + O(a²).

The signed real and imaginary parts are measured relative to this phase convention. The observable `pumping` is the far-field time-averaged horizontal fluid velocity divided by a² in the same amplitude limit. The order is stationary response, far-field mean, then small-amplitude coefficient. It is a laboratory fluid-velocity measurement; the sheet remains anchored.

## Inputs and units

Every experiment dictionary contains `wave_number` k in [.7,1.3] inverse millimeters, `frequency` omega in [.5,8] radians/second, and `observable`. The allowed observables are `u_real`, `u_imag`, `v_real`, `v_imag`, and `pumping`. Harmonic experiments additionally contain `height` h in [.2,2] millimeters. Pumping has no height field.

The unknown `viscosity` is nu in [.7,1.4] millimeters squared/second, common to all experiments. Harmonic outputs U and V have units inverse seconds. Pumping has units inverse millimeter per second. Wave number, frequency, height and sheet motion are controlled independently and known.

## Interface and calibration

`Model()` exposes the fitted scalar as `viscosity`. `fit(records)` returns self. Each record in `data/calibration.json` contains `input`, measured `value` and fixed instrument standard deviation `sigma`. `predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in the requested units. An empty list returns an empty array.

Run `pytest -q` for the public interface and calibration check.
