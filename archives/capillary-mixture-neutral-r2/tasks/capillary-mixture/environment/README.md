# Instantaneous flow of a binary liquid

A two-dimensional, constant-density incompressible Newtonian liquid occupies a square periodic domain, 0<=x,y<2*pi. Its composition c is the volume fraction of component A minus 1/2; the other volume fraction is 1/2-c. The constituents share the bulk velocity. Mean velocity is zero. There are no walls, external mechanical forces, thermal changes, or fluctuating stresses.

In reduced units its Helmholtz free energy is

    F[c]=integral [c*c/2 + 0.2*|grad(c)|^2/2] dx dy.

The reversible mechanical stress is the thermodynamic stress of this free energy under material deformation. Other stresses are the incompressibility pressure and Newtonian viscous stress. The local dynamic viscosity is

    eta(c)=viscosity*exp(viscosity_contrast*c).

The single unknown material scale `viscosity` is in [0.002,0.008]. `viscosity_contrast` is a known preparation control in [0,7]. This is a local isotropic Newtonian material law, without stress memory or additional constitutive terms.

Each fresh preparation prescribes

    c(x,y)=sum(mode.amplitude*cos(mode.wave dot [x,y]+mode.phase)
               for mode in modes).

The detector reads the instantaneous quasistatic velocity: momentum has relaxed, while composition has not yet changed by advection or molecular diffusion. Take this controlled zero-inertia limit before the short-time composition limit. No external mechanism holds the pattern during the measurement. Every query starts with its own prescribed composition.

The reported signed coefficient is

    2*mean_over_cell[u[component](x,y)
                    *cos(detector_wave dot [x,y]+detector_phase)].

Each input has `modes`, a list of one to three dictionaries with integer `wave` components in [-3,3], not both zero, `phase` in [-pi,pi], and `amplitude` in [-0.3,0.3]. The sum of absolute amplitudes is at most 0.35. `component` is 0 or 1, `detector_wave` has integer components in [-6,6] and is not zero, and `detector_phase` is in [-pi,pi]. All controls are exact. All quantities use the reduced units above.

Calibration records in `data/calibration.json` contain `input`, measured `value`, and fixed independent Gaussian standard deviation `sigma`. Calibration includes spatially uniform viscosity and nonuniform composition. Implement `Model.fit(records)`, return self, and set `self.viscosity`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order.
