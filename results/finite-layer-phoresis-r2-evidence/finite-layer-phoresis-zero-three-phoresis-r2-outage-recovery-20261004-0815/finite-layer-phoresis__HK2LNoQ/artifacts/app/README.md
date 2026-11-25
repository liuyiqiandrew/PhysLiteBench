# Solute-driven sphere motion

A dilute ideal neutral solute is dissolved in an incompressible Newtonian solvent at fixed temperature. The only unknown is the uniform dynamic viscosity `viscosity`, in [.8,1.6]. Length, energy and time units are ell, k_B*T and t0; viscosity is in k_B*T*t0/ell^3. Solute concentration is number per ell^3. Ignore fluid inertia, gravity and particle Brownian motion in the mean response.

A rigid sphere of known `radius` in [.7,1.3] moves freely in otherwise unbounded solvent. The solute interacts with a prescribed radial potential carried with the sphere. At distance z=r-radius from its surface,

    U(z) = strength*(1-z/width)^2  for 0 <= z <= width,
    U(z) = 0                     for z > width.

U is measured in k_B*T. `strength` lies in [-2,2] and `width` lies in [.05,3.5]. The solute is excluded from the sphere, with no normal solute flux at its surface. There is no adsorption reaction or surface transport. Solute interactions exert equal and opposite forces on the solute and sphere. The isotropic solution pressure includes the ideal-solute osmotic part.

The solvent cannot penetrate the sphere. Its known Navier slip length `slip` lies in [0,2]. Let n point from the sphere into the solvent, v_s be the local solid velocity, and sigma be the Newtonian solution stress. At the actual sphere surface, the tangential relative velocity satisfies

    (u-v_s)_tangent = (slip/viscosity)*(sigma*n)_tangent.

The surface has no additional tangential solute friction; the specified conservative potential is purely radial. The sphere has no external mechanical force. Its fluid traction and the reaction from this solute interaction determine its translation.

Reservoirs maintain a weak concentration gradient G along a chosen direction. The response is linear in G about a uniform positive bulk concentration c0: take G to zero with width and radius much smaller than c0/abs(G). Far from the sphere the concentration perturbation approaches G times the coordinate along this direction, and there is no imposed solvent motion or mechanical pressure gradient. Use the leading dilute, rapid-solute-diffusion limit, in which solute advection is negligible and the steady number flux is proportional to -(grad(c)+c*grad(U)). The solvent is in creeping flow and transfers the solute interaction force through its mechanical stress. The stated ideal continuum applies throughout the parameter ranges.

Each preparation reaches its steady response independently. Record the sphere velocity relative to the distant solvent, positive along G, divided by G. This signed response has units ell^5/t0. Every input contains `radius`, `width`, `strength` and `slip`.

Calibration records in `data/calibration.json` contain `input`, measured `value`, and independent Gaussian uncertainty `sigma=.00003`, fixed by the instrument.

Implement `Model.fit(records)`, returning self and storing the inferred `self.viscosity`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` with the requested responses in input order. Run `python -m pytest -q test_public.py`.
