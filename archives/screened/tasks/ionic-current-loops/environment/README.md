# Mixing in a periodic electrolyte film

An isothermal dilute ideal electrolyte occupies a two-dimensional periodic square of side 2*pi. A stationary porous matrix holds the solvent at rest; ignore fluid advection and solvent inertia. Two monovalent cation species have concentrations c0 and c1 and charges +1. A monovalent anion has concentration c2 and charge -1. The Debye length and charge relaxation time are negligible on the measured scales, so the bulk is locally electroneutral. There are no reactions, electrodes, externally imposed currents, or externally imposed electromotive forces. Magnetic induction is negligible. The electrostatic potential is periodic and determined up to an additive constant.

Use reduced units with k_B*T=1 and elementary charge one. The diffusion coefficients of species 0, 1, and 2 are respectively D, 0.08*D, and 3*D, with one shared unknown D in [0.5,3]. Their dilute Einstein mobilities equal these diffusion coefficients. Each species flux relative to the fixed solvent is the sum of ordinary Fick diffusion and migration in the electrostatic field. Ignore finite-size effects, activity corrections, and all cross-friction corrections beyond this ideal dilute model.

A preparation specifies means, amplitudes, waves, and phases, each containing two entries for the two cations. At preparation, c_i(x,y)=means[i]+amplitudes[i]*cos(waves[i][0]*x+waves[i][1]*y+phases[i]), for i=0,1, and c2=c0+c1. Each mean is in [0.6,1.4]; the absolute amplitude is at most 0.75 times its mean. Wave components are integers in [-2,2] with nonzero wavevector. Phases are between -pi and pi. All fields and fluxes have the square's periodic boundary conditions.

Fast charge relaxation has already occurred. Measure the initial slope of a concentration Fourier coefficient, before appreciable composition change. For species in {0,1,2}, detector wavevector [qx,qy] with components in [-4,4] and nonzero magnitude, and detector_phase between -pi and pi, return the time derivative at preparation of

    2*mean_over_square[c_species(x,y,t)*cos(qx*x+qy*y+detector_phase)].

The initial concentration profiles and settings are exact. Slopes are obtained from repeated preparations; no finite time step is part of the requested measurement. Every calibration preparation is independent of y and reflection-symmetric in x. Each record contains input, measured value, and independent Gaussian standard deviation sigma.

Implement Model.fit(records), returning self and setting Model.diffusivity, and Model.predict(experiments), returning one finite NumPy value per input. The shared diffusivity must be inferred from data/calibration.json. You may revise the supplied prediction model for all allowed preparations.
