# Relaxation of a rotating shallow layer

An incompressible liquid layer has resting depth H=0.05 m and a free upper surface under gravity g=9.81 m/s². Use the linear, hydrostatic shallow-layer limit about its resting state, on a local Cartesian f-plane. The centrifugal equilibrium is included in the background state. The domain is periodic in x with length 2*pi m; perturbations are independent of y. The vertical rotation vector has Coriolis parameter `rotation`, in [-3,3] s^-1, positive upward. The coordinates x,y,z form a right-handed system.

The only dissipation is a uniform linear bottom force per unit mass equal to -drag_rate times the horizontal fluid velocity. The unknown positive `drag_rate` is in [0.08,0.24] s^-1 and is common to all preparations. There is no surface forcing, mass source, thermal effect, viscosity within the horizontal plane, or additional relaxation term. Retain only terms linear in free-surface displacement and horizontal velocity. The stated constant-f, shallow-layer and drag approximations define the model.

At release, a fresh preparation has

    surface_displacement(x)=height*cos(wave*x),
    velocity_x(x)=along_velocity*sin(wave*x),
    velocity_y(x)=across_velocity*sin(wave*x).

Here `wave` is an integer in [1,3], interpreted in m^-1, `height` is in [-0.0004,0.0004] m, and both velocity amplitudes are in [-0.01,0.01] m/s. All inputs are exact. The three initial amplitudes are independently prescribed. Measurements take place at `time` in [0,40] s after release. The detector reports the signed cosine Fourier amplitude of surface displacement, in meters:

    (1/pi)*integral_0^(2*pi) surface_displacement(x,time)*cos(wave*x) dx.

Every input contains `wave`, `rotation`, `height`, `along_velocity`, `across_velocity`, and `time`. Implement `Model.fit(records)`, returning self and setting `drag_rate`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order. Calibration records in `data/calibration.json` contain `input`, measured `value`, and fixed independent Gaussian standard deviation `sigma`. Use the same fitted drag rate for all queries.
