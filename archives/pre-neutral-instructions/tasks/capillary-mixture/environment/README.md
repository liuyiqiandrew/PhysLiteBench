# Relaxation of a binary liquid

A two-dimensional, constant-density incompressible Newtonian liquid occupies a square periodic domain, 0 <= x,y < 2*pi. Its composition variable c is the volume fraction of component A minus 1/2; component B has volume fraction 1/2-c. Both constituents move with the same bulk fluid velocity, apart from equal and opposite molecular diffusive fluxes. The spatial mean velocity is zero. There is no external forcing, no boundary flow, and no thermal evolution.

In the reduced units used here, the free energy is

    F[c] = integral [c^2/2 + 0.2*|grad(c)|^2/2] dx dy.

The chemical potential is the variation of F with respect to c. The molecular diffusive composition flux relative to the liquid is minus `mobility` times the gradient of this chemical potential. The constant unknown mobility is in [0.02, 0.09]. The shear viscosity is 0.004. Fluid inertia is negligible: momentum balance is the periodic quasistatic Stokes balance at each instant. In addition to Newtonian viscous stress and incompressibility pressure, the liquid has the reversible composition stress obtained from this free energy: its work under any volume-preserving material displacement is minus the corresponding change in F when the composition is carried with the liquid. There are no other stresses. Use this continuum model at the stated coefficients, without noise or additional constitutive terms.

Each fresh preparation starts with a specified composition

    c(x,y,0) = sum_j amplitude_j*cos(kx_j*x + ky_j*y + phase_j).

Each input contains `modes`, a list of one to three dictionaries with `wave: [kx, ky]`, `amplitude`, and `phase`. Wave components are integers in [-3, 3], not both zero. Each amplitude is in [-0.3, 0.3], and the sum of absolute amplitudes is at most 0.35. Phases are in [-pi, pi]. The mean composition is zero. Inputs also contain `time` in [0, 6], a nonzero integer `detector_wave` with components in [-6, 6], and `quadrature`, either `cosine` or `sine`.

The detector returns twice the spatial average of c at the requested time multiplied by cos(qx*x+qy*y) or sin(qx*x+qy*y), according to `quadrature`. It is a signed composition amplitude. It is not a squared intensity. All lengths, times, free energies, and material coefficients use the common reduced units above.

Calibration records in `data/calibration.json` measure the decay of single composition stripes. Records contain `input`, measured `value`, and independent Gaussian standard deviation `sigma`. Implement `Model.fit(records)`, return self, and set `self.mobility`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order. Each prediction refers to its own fresh initial preparation.
