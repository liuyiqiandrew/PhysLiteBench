# Pressure surface waves, revision 1

The supplied predictor already contains the full finite-prestrain bulk response of the stated compressible neo-Hookean material. It solves the homogeneous static normal balance, the two decaying elastic modes and their exact nominal-traction matching problem. Its physical approximation is that the applied load remains fixed per reference area and direction during the wave. The apparatus instead applies constant pressure normal to the instantaneous surface. These loads have the same static force but different incremental work.

The fitted parameter is density per undeformed volume. The elastic moduli are known. This choice prevents the prescribed static pressure/deformation from revealing the fitted modulus without calibration. The reference density enters every speed as its inverse square root, and 240 zero-pressure readings across six finite tangential stretches identify it uniquely. Both loading laws coincide at zero pressure. Instrument uncertainty is fixed at 0.002 m/s and independent of the response or parameter. The public fit is unfinished; the separate completed shortcut contains the fitting implementation. The exact neutral instruction permits replacing all prediction helpers.

Let the base deformation be `F=diag(s,1,t)` and `J=s*t`. Measure pressure in units of the shear modulus. With `c=1-2*ln(J)`, static normal balance gives `t^2-c=-pJ`. The current density is `rho0/J`. Differentiating the full stored energy with respect to F, then pushing forward its Hessian, gives

```
A[i,j,k,l] = (mu/J) * (delta_ik*B[j,l]
                          + 2*delta_ij*delta_kl
                          + c*delta_il*delta_kj),
B = F*F^T.
```

Thus all material and geometric prestress terms in the bulk are already retained. Define `b=t^2`, `chi=b/(b+2+c)` and `x=(rho0*v^2/mu - s^2 + t^2)/b`. The two decay constants are `qP=sqrt(1-chi*x)` and `qS=sqrt(1-x)`. With tangential displacement represented as i times a real amplitude, their displacement columns are `(1,-qP)` and `(qS,-1)`. These exact bulk columns and their complete nominal tractions appear in both controls.

For an ideal pressure actuator the reference traction is `-p J F^-T N`. For a displacement gradient L, its increment is

```
-p J * [(tr L)*I - L^T] * F^-T N.
```

On the exposed plane the two wave traction equations therefore subtract `pJ*normal_amplitude` from the shear equation and `pJ*tangential_amplitude` from the normal equation, in the normalized variables. The source instead sets its nominal traction increments to zero. It is the exact dead-reference-load response, not a failed approximation to the bulk Hessian. All input pressure values still affect its base deformation, bulk modes and surface response.

The columns coalesce at `x=0`; dividing the determinant by x removes this representation degeneracy. The nonzero physical branch is isolated without squaring the secular equation. Both controls have one positive, subsonic root throughout the 187-point domain grid. Roots range from 0.174 to 0.892, minimum normalized squared speed is 0.488, and the slowest depth decay constant exceeds 0.328. These margins are well away from the removed degeneracy, a propagating bulk-wave threshold or a surface instability.

The independent verifier does not use those columns or a secular equation. It takes finite-difference second variations of the original scalar energy plus the pressure potential `p*det(F)`, in the current displacement gradient. The added determinant is a bulk null Lagrangian: it changes the natural boundary traction but leaves the bulk acoustic operator unchanged. A consistent Fourier finite-element discretization in depth then supplies the lowest surface eigenfrequency. The far boundary is clamped at 28 inverse wave numbers; 384/768 elements and Richardson extrapolation define the private reference. Extra 512/1024 and deeper-domain checks verify convergence. Scalar energy W alone independently verifies the completed source's dead-load result.

A direct finite displacement of the actual pressure traction checks its linearization within 2.1e-10 in normalized units. The independent energy-based physical speeds agree within 5.8e-7 m/s at corner and off-grid controls; mesh and depth changes are about 1e-6 m/s or less. The dead-load calculation agrees with its own scalar-energy reference within 1.5e-5 m/s. The unstressed Rayleigh cubic and the incompressible hydrostatic-pressure-independent limit also pass. The latter is an author control outside the graded finite-compressibility apparatus.

All 256 noisy fits pass calibration and density recovery. The correct hidden error is at most 1.14e-5 normalized; the shortcut's loaded-group errors are 0.163–0.171, against a 0.04 gate. Both pass independent zero-pressure anchors. Density recovery without noise is at roundoff throughout the allowed interval. Calibration uses finite strain and validates the exact bulk solver, while the pressure-dependent boundary law remains unobserved there.

This is a genuine revision of the broader prestressed-elasticity theme. The archived prestrained-solid candidate omitted geometric stress from bulk wave propagation. Here that entire bulk Hessian is correct; the distinction is the physical external loading and its boundary work. The public actuator description fixes constant actual-area pressure and negligible inertia. No real-fluid added mass is silently omitted. Only current-coordinate phase speed is graded; no polarization normalization or additional detector convention is required.

No model difficulty outcome is claimed before evaluation. A trial that retains fixed nominal traction at nonzero pressure is a physical failure. A trial that derives the proper pressure increment but makes a sign, density, mode-selection or implementation mistake must be classified separately.

Primary background: [Otténio, Destrade and Ogden, incremental solid-fluid boundary conditions](https://arxiv.org/abs/0711.4438), and [Murphy and Destrade, compressible prestressed surface waves](https://arxiv.org/abs/0812.1479). The particular massless ideal pressure actuator is defined in this task; the scalar-energy and boundary calculations above independently establish its response.

Run `python scripts/validate_pressure_surface_waves.py` from the stage root with the pinned task dependencies. Add `--generate` only to intentionally regenerate both calibration copies. Scientific and local-control reports are in `results/pressure-surface-waves-r1-validation.json` and `results/pressure-surface-waves-r1-local-controls.json`. Root owns Docker controls and all model evaluations.
