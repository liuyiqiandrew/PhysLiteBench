# Hydrodynamic heating: bounded wall-memory revision

Status: bounded author check complete, awaiting independent peer and root decision. No task harness, calibration dataset, model evaluation, canonical edit or Git operation was made.

The actual six prior fresh trials were reviewed before designing this change. Canonical r1 passed 1/3, and archived r2 passed 2/3. The r1 passer reconstructed the full hydrodynamic current before computing drag heat. Both r2 passers correctly distinguished local viscous strain heating from local mechanical work and its spatial energy flux. The failures retained the supplied physical closure after calibration agreement. Corrected temporary scratch errors are qualified in `prior-evidence-review.json`. Changing a coefficient or depth window would not invalidate those passing repairs.

## Proposed physical change

Keep the same incompressible Newtonian charged-fluid slab, exact Maxwell fields, known gamma=.06 and kinematic viscosity nu=.04, and unknown plasma frequency p in [.85,1.15]. Use normal incidence, so every point on a wall has the same phase. Oblique incidence followed by averaging an infinite wall would cancel a second-harmonic thermal signal; that is not this apparatus.

Replace the no-slip boundary by an explicit massless, optically inert tangential contact layer at each fixed face. Each layer consists of a temperature-independent energetic spring and a Newtonian dashpot in series, anchored to the stationary substrate. There is no layer inertia or extra electromagnetic response. Write rho=m*n0, k_s for spring stiffness per area, and kappa for dashpot drag per area. Known controls are alpha=kappa/rho and tau=kappa/k_s. Fluid slip velocity v is the sum of spring-extension rate and dashpot velocity d. The common traction magnitude satisfies T=k_s*e=kappa*d. Thus the contact law is tau*T_dot+T=kappa*v. Each face opposes the fluid motion. This is an explicit mechanical boundary constitutive model, not a thermal response model.

The measured quantity is heat removed by an ideal fast thermostat from the front contact dashpot, per area. The spring has no thermoelastic entropy, and its reversible mechanical energy is not counted as removed heat. Temperature excursions vanish in the weak-field limit; the thermostat's known response has no delay at either harmonic. Neither thermal feedback nor thermal transport changes the first-order optical/mechanical response. Equilibrium background heat is subtracted. The sensor is referenced to the incoming electric field at the front face, Re[E0 exp(-i omega t)]. It records either the mean heat or the two ordinary quadratures at 2 omega, divided by the mean normal incident optical power. A scalar output selector for mean/in-phase/quadrature is sufficient; no new dynamical unknown is needed.

The current variable J=q*n0*v gives a normalized traction a=T*q/(m), so tau*a_dot+a=alpha*J. Linear boundary conditions are nu*J_z=alpha*J/(1-i omega tau) at the front and nu*J_z=-alpha*J/(1-i omega tau) at the rear. The sign convention takes positive T as the traction delivered to a contact by the fluid. The contact layer can slip while the solid substrate and optical plane remain fixed.

## Completed source and physical prediction

Both predictors solve the full exact four-mode electromagnetic/viscous field problem with these complex contact impedances. Both retain the r2 passers' correct bulk drag plus symmetric-strain heating. The source calls the instantaneous power delivered by the fluid to a contact, T*v, its removed heat. The physical calorimeter measures kappa*d^2=T^2/kappa. Their difference is the time derivative of spring energy T^2/(2*k_s). No mean optical field or mechanical boundary correction is omitted.

If j is the complex wall current and a=alpha*j/(1-i omega tau), normalization by E0^2/2 gives

- shared mean q0=|a|^2/(alpha*p^2);
- physical complex second harmonic q2=a^2/(alpha*p^2);
- source second harmonic w2=a*j/p^2.

Here q(t)=q0+Re[q2 exp(-2i omega t)]. Therefore w2=(1-i omega tau)*q2. Mean calibration is exactly shared at finite relaxation, and the nonzero harmonic relative complex discrepancy is exactly omega*tau, independent of the fitted p. The second-harmonic readout tests reversible interfacial storage, whereas r2 tested spatial transport within the fluid.

This is a coherent power-attribution error. It is not a thermodynamically exact heat model: instantaneous mechanical power can become negative while dashpot heat cannot. This makes a positivity check or elementary series-contact energy balance a potentially easy repair. There is no evidence yet that the revision is harder for a model, and no recommendation to create a harness before root and peer assess this limitation.

## Bounded evidence

`check.py` and `report.json` preserve the first science run unchanged. Thirty-two endpoint combinations cover p=.85/1.15, omega=.7/1.5, thickness=.3/1.2, alpha=.08/.4, and tau=.2/1.2. The independent reference solves the eight-real first-order Maxwell/momentum boundary problem, then integrates the actual series-contact ODE in time. It determines the periodic initial contact traction from the affine one-period map and directly integrates dashpot heat and its Fourier quadratures. It does not insert the analytic heat-transfer factor into the readout.

Maximum modal/reference discrepancies are 1.34e-12 for mean heat, 5.84e-12 for the physical second harmonic, and 7.78e-12 for source work. Tightening the reference tolerance by ten changes the representative readout by 2.18e-12. Full optical mean energy balance, 1-|r|^2-|t|^2=bulk heating+both contact heats, closes within 1.59e-15. The run took 2.53 seconds on the author machine.

Mean and harmonic-magnitude signals range from .006898 to .106897 of incident power; they are not constructed by subtracting a large background or choosing a cancellation. Endpoint complex gaps range from 14% to 180%. Individual quadratures can cross zero and should not be graded by their individual relative errors. An ordinary multi-experiment RMS over the two documented quadratures is appropriate, with the existing .04 threshold if a package is eventually built. No data, noise or grading change exists now.

Six finite-memory front-wall mean-calibration settings vary frequency and thickness. On an 81-point full-p grid, all six mean curves increase; the minimum sampled secant derivative is .05718. Twenty-five noiseless fits, including endpoint truths, recover p within 2.36e-8. This is empirical global-range evidence, not an analytic injectivity proof or a 256-noise validation.

The preserved limits include tau=0 exactly, two small but nonzero relaxation times with gaps below 4%, weak contact friction tending to free slip, and large contact friction tending to no slip. Both slip limits suppress the contact signal; they are not proposed as scored diagnostics. The zero-memory limit makes work and heat identical. The reference uses the same continuum equations but independent field integration and time-domain heat evaluation; it is not an independent microscopic theory of a real surface material.

`execution-history.json` records one failed uv cache initialization before Python execution. Running an existing cached Python interpreter directly then completed the only science run. No source formula was changed between those invocations, and no failed numerical result was dropped.

## Remaining decision

A future task would need a final public domain and scalar readout interface, rigorous or otherwise sufficient global identification checks, fixed uncertainty chosen before model runs, full-domain and per-scored-case convergence, 256 noise draws and local controls. This bounded record makes no claim those are done. Root may reject the mechanism as too easy without spending a model batch. The canonical r1, archived r2 and all original outcomes remain unchanged.
