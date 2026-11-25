# Driven collisionless electron gas

An ideal classical electron gas is uniform in the transverse directions. Its longitudinal dynamics is the nonrelativistic collisionless electrostatic mean-field model. Electron mass, charge magnitude and vacuum permittivity are all one in the stated units. A fixed uniform positive background has the same number density as the electrons. There are no collisions, material boundaries, applied magnetic fields, or additional relaxation channels.

Before driving, the longitudinal velocity distribution per unit spatial volume is

    f0(v) = density * 15/(16*velocity_width)
            * [1 - (v/velocity_width)^2]^2,  |v| < velocity_width,

and zero otherwise. Transverse motion does not affect the experiment. The unknown `density` is common to all preparations and lies in `[0.6, 1.4]`. The positive `velocity_width` is known separately for each preparation. The spatially uniform state is prepared in the remote past.

External charges prescribe a longitudinal electric field which, in the absence of the electrons, would be

    E_ext(x,t) = amplitude * exp(eta*t)
                * cos(wavenumber*x - frequency*t),  t <= 0.

Here `eta > 0`. The electrons respond to the total electric field, including their own field. The electron charge is minus one. The system is spatially periodic with period `L = 2*pi/wavenumber`; the induced potential is periodic and has zero spatial mean.

At time zero the detector records the in-phase total electric field coefficient. The reported dimensionless transfer is

    lim_(eta -> 0+) lim_(amplitude -> 0)
        2/(L*amplitude) * integral_0^L E_total(x,0)*cos(wavenumber*x) dx.

The time unit, length unit and velocity unit are consistent with mass, charge magnitude and permittivity equal to one. `frequency` is an angular frequency in inverse time; `wavenumber` is in inverse length. Every experiment uses `wavenumber` in `[1.5,2.2]`, `velocity_width` in `[0.9,1.2]`, and a drive selected from these frequency windows:

    frequency/(wavenumber*velocity_width) = 0,
    or in [0.20,0.65], or in [1.5,2.5].

`data/calibration.json` is a list of independent measurements. Each record has `input` with `wavenumber`, `velocity_width`, and `frequency`, plus a measured `value` and its known Gaussian standard deviation `sigma`. The fixed instrument standard deviation is `0.0012` in transfer units.

Implement `Model.fit(records)`, returning `self` and setting the fitted scalar `density`. Implement `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)` containing the transfer for every input dictionary. Prediction inputs obey the same apparatus and ranges.
