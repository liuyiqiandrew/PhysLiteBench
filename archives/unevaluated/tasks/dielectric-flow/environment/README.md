# Electric circulation in a patterned dielectric liquid

An incompressible isotropic liquid occupies a two-dimensional periodic cell, 0<=x,y<2*pi. Use reduced units. Its density is fixed and its unknown dynamic viscosity is spatially constant in [.7,1.6]. Work in the steady creeping-flow limit; inertia is neglected. There are no walls or fluid interfaces. A uniform mechanical reaction enforces zero mean velocity; the pressure fluctuation and velocity are periodic. No other mechanical force is applied.

The liquid's local scalar permittivity and Ohmic conductivity are prescribed in the laboratory coordinates:

    epsilon(x,y)=1+sum(mode.amplitude*cos(mode.wave dot [x,y]+mode.phase)
                       for mode in permittivity_modes),
    sigma(x,y)=100*[1+sum(mode.amplitude*cos(mode.wave dot [x,y]+mode.phase)
                         for mode in conductivity_modes)].

External material control maintains these profiles. They are not advected by the flow. Each list has zero, one or two modes. A mode has integer wave components in [-2,2], not both zero, phase in [-pi,pi], and real amplitude. In each list, sum(abs(amplitude))<=.7, so both material properties are positive. Density and viscosity are independent of these controls. The dielectric is linear, isotropic and nondispersive. Electrostriction contributes only to pressure at the stated incompressible limit. Thermal, diffusion, thermoelectric and chemical body forces are excluded from this specified material model.

An external voltage source maintains the input mean electric field `field`, a real two-vector with Euclidean length at most .4. The local field has zero curl and this mean; its periodic potential correction has zero mean. The constitutive current is j=sigma*E. Electric charge relaxes before each measurement; steady charge conservation holds. Retain the electric force on the dielectric liquid. Charge convection, displacement current, magnetic forces and the flow's effect on the electrical problem are negligible. The externally controlled material profile removes Joule heat and does not introduce another mechanical force. Treat these continuum constitutive assumptions as the prescribed limit.

The apparatus reports one steady velocity Fourier coefficient:

    2*mean_over_cell[u[component](x,y)
                    *cos(detector_wave dot [x,y]+detector_phase)].

`component` is 0 for x or 1 for y. `detector_wave` has integer components in [-4,4], not both zero; `detector_phase` is in [-pi,pi]. Signed coefficients are allowed. Every experiment has the two mode lists, `field`, `component`, `detector_wave`, and `detector_phase`. All listed controls are exact.

Calibration uses proportional permittivity and conductivity profiles, several applied fields and several velocity projections. Measurement errors are independent Gaussian errors with the listed sigma. Fit the single shared viscosity. Implement `Model.fit(records)`, returning self and setting `Model.viscosity`, and `Model.predict(experiments)`, returning a finite NumPy array with shape `(len(experiments),)`. Calibration records in `data/calibration.json` contain `input`, `value` and `sigma`. Run `python -m pytest -q test_public.py`.
