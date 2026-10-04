# Solute-driven motion near solid surfaces

A dilute ideal neutral solute is dissolved in an incompressible Newtonian solvent at fixed temperature. The only unknown is the uniform dynamic viscosity `viscosity`, in [.8,1.6]. Length, energy and time units are ell, k_B*T and t0; viscosity is in k_B*T*t0/ell^3. Solute concentration is number per ell^3. Ignore fluid inertia, gravity and particle Brownian motion in the mean response.

The solute interacts with a solid through a prescribed potential, carried with the solid. At distance z from its surface,

    U(z) = strength*(1-z/width)^2  for 0 <= z <= width,
    U(z) = 0                     for z > width.

U is measured in k_B*T. The solute is excluded from the solid, with no normal solute flux at its surface. There is no adsorption reaction or surface transport. `strength` lies in [-2,2] and `width` lies in [.05,3.5]. The solvent obeys no slip at the actual solid surface. Solute interactions exert equal and opposite forces on the solute and solid. The isotropic solution pressure includes the ideal-solute osmotic part; there is no externally imposed mechanical pressure gradient.

Reservoirs maintain a weak solute concentration gradient G along a chosen direction. The response is linear in G about a uniform positive bulk concentration c0: take G to zero with width and the particle radius much smaller than c0/abs(G). Use the leading dilute, rapid-solute-diffusion limit, in which solute advection is negligible and its steady number flux is proportional to -(grad(c)+c*grad(U)). The solvent is in creeping flow and transfers the solute interaction force through its mechanical stress. All material properties remain constant. The stated ideal continuum applies at every listed width and radius.

There are two preparations:

- `kind="wall"`: A fixed infinite planar wall bounds the solvent half-space. The imposed concentration gradient is parallel to the wall. Far from the interaction region the shear rate vanishes and the mechanical pressure is uniform. Record the far-field tangential solvent velocity relative to the wall, positive along G.
- `kind="sphere"`: A rigid sphere of known `radius` in [.7,1.3] moves freely in otherwise unbounded solvent. There is no imposed fluid motion at infinity. Far from the sphere the concentration perturbation approaches G times the coordinate along the imposed gradient. The sphere has no external mechanical force: its fluid traction and the reaction from the specified solute interaction determine its translation. Record its velocity relative to the distant solvent, positive along G. The potential uses z=r-radius in this preparation.

Each reading reports velocity divided by G in the linear-response limit, in ell^5/t0. It is signed. Every input contains `kind`, `strength` and `width`; sphere inputs additionally contain `radius`. Each preparation reaches its steady response independently.

Calibration records in `data/calibration.json` contain `input`, measured `value`, and independent Gaussian uncertainty `sigma=.00003`, fixed by the instrument.

Implement `Model.fit(records)`, returning self and storing the inferred `self.viscosity`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` with the requested responses in input order. Run `python -m pytest -q test_public.py`.
