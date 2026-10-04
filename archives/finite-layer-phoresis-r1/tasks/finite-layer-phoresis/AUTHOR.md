# Finite-layer phoresis, revision 1

This task measures solute-driven flow at a flat wall and the translation of a force-free rigid sphere. The completed source solves the full curved solute transport problem but uses the locally planar hydrodynamic transfer kernel. That approximation is exact for the flat-wall calibration and approaches the physical spherical response when the interaction width is small compared with the radius. The hidden preparations use broader finite interaction ranges, so the curvature of momentum transfer matters. No difficulty outcome is claimed before model evaluation.

The common unknown is viscosity, between 0.8 and 1.6 in the stated units. Every response is proportional to its inverse. The 288 calibration readings span 18 flat-wall combinations of interaction strength and width, so weighted fitting uniquely identifies viscosity. The instrument standard uncertainty is the same fixed 0.00003 for every reading and does not encode a noiseless response. Both completed controls have identical calibration predictions. The public starter retains the complete planar-closure forward solver but leaves `fit` unfinished, matching the standard instruction. The separate shortcut control includes the fit.

The public dilute, rapid-diffusion limit removes advective solute feedback. In the particle frame the stationary ideal solute obeys

```
div(grad c + c grad U) = 0.
```

The isotropic mechanical solution pressure includes osmotic pressure. The fluid momentum equation is

```
-grad p + eta lap v - c grad U = 0,   div v = 0.
```

The rigid particle receives both the fluid traction and the opposite reaction of the smooth solute potential. No additional osmotic contact force is added to that total-pressure traction. This convention prevents double counting the hard-wall solute contact contribution. No slip applies at the actual sphere, and there is no external mechanical force or far-field fluid flow. The public small-gradient limit makes the far-field linear concentration profile a response condition, not a globally negative concentration preparation.

For the sphere write the first-order concentration as `G*f(r)*cos(theta)` and put `h=exp(U)*f`. The exact radial solute equation is

```
(r^2 exp(-U) h')' - 2 exp(-U) h = 0.
```

No normal flux gives `h'(a)=0`. The prescribed potential and its first derivative vanish at `R=a+width`. Outside that radius the concentration is `f=r+B/r^2`, so exact far-field matching gives `R*h'(R)+2*h(R)=3*R`. Both controls solve this same problem with adaptive collocation. They retain the finite interaction width, sphere radius and concentration polarization; neither substitutes a flat concentration profile.

The reciprocal translating-sphere Stokes field, combined with the direct solute reaction, gives the physical velocity per unit concentration gradient:

```
V/G = 2/(9 eta a) * integral_a^R
        [r^2 - 3*a*r/2 + a^3/(2*r)] * f(r)*U'(r) dr.
```

The combined kernel vanishes quadratically at the surface. Its leading expansion is `3*(r-a)^2/2`. The source uses exactly this locally planar expansion while retaining the exact curved concentration. This is a coherent interaction-layer approximation, not an incorrect integration, missing fitted parameter or broken Stokes solve. At broad interaction range, it assigns the wrong mechanical weighting to where solute momentum is transferred. The source need not satisfy the full curved momentum balance: that is the physical closure being tested.

For a fixed flat wall, normal mechanical balance and no far-field shear give its fluid velocity per gradient exactly:

```
u_wall/G = -1/eta * integral_0^width z*[exp(-U(z))-1] dz.
```

This relation holds for arbitrary width in a planar half-space. For a thin interaction layer the freely moving sphere has the opposite velocity to the wall-driven distant fluid. The sign distinction is explicit in the public readout. Repulsive positive U drives the sphere against the positive concentration gradient and the planar fluid along it. Attractive U reverses both signs. Velocity divided by concentration gradient has units ell^5/t0; the given viscosity unit supplies the remaining length and time factors.

The private reference does not use the reciprocal force kernel. It discretizes the self-adjoint solute equation with conservative radial piecewise-linear finite elements and four-point element quadrature. The stiffness terms are `r^2 exp(-U)` and `2 exp(-U)`; outer matching adds `2R` to the last matrix entry and `3R^2` to the right-hand side. It then independently solves the fourth-order Stokes streamfunction boundary-value problem. With

```
v_r = 2*psi*cos(theta)/r^2,
v_theta = -psi'*sin(theta)/r,
```

the equation is

```
eta*[psi'''' - 4*psi''/r^2 + 8*psi'/r^3 - 8*psi/r^4] = -G*f*U'.
```

No slip sets `psi(a)=psi'(a)=0`. Outside the interaction region a force-free decaying disturbance has `psi` in the span of `r^2` and `1/r`. Removing the Stokeslet and growing `r^4` term fixes the two outer conditions. The velocity is extracted as `-2*(psi'+psi/R)/(3R)`. This also imposes total force balance, rather than setting the fluid traction alone to zero. Reference solute meshes 256 and 512 use Richardson extrapolation;384/768 checks quantify convergence.

The validator independently integrates direct solute reaction and the mechanical wall traction from the Stokes solution. They balance to 7.3e-13 relatively. It also checks no-slip/no-Stokeslet boundary residuals, exact zero interaction, the thin-layer planar limit, quadratic scaling with simultaneous radius and width scaling, parameter recovery throughout the allowed interval, and exact calibration equivalence. The completed shortcut is independently compared with the planar-kernel prototype so its failure is not caused by an implementation defect.

Across 34 domain and off-grid cases, oracle and independent reference differ by at most 2.65e-10; mesh refinement changes predictions by at most 2.10e-10. Hidden spherical signals have absolute magnitudes 0.145–1.91. All 256 noisy calibration fits pass parameter and calibration checks; the physical oracle passes every hidden group. The greatest noisy oracle error is 7.21e-5, while the smallest noisy shortcut sphere-group error is 0.2348, against the unchanged 0.04 relative-RMS gate. Both controls pass flat-wall anchors. Local controls give 7/7 for the oracle and 4 passes plus 3 intended spherical failures for the shortcut, each in less than 0.7 seconds.

The archive scan found no prior diffusiophoresis or diffusio-osmosis task among 92 distinct AUTHOR families. Related electroviscous and ionic candidates test charge/current closure; this task concerns neutral-solute momentum transfer through a curved interaction region. The shared continuum background is acknowledged without claiming an entirely unrelated physical theory.

A trial that preserves the planar hydrodynamic kernel for broad spherical layers is a physical approximation failure. A trial that derives the correct curved force balance but makes a sign, derivative, normalization or numerical error must be classified separately. Full public/native trajectory review and a repair at fixed fitted viscosity should establish the cause after any failed model trial.

Primary background: [Marbach, Yoshida and Bocquet, Local and global force balance for diffusiophoretic transport, Journal of Fluid Mechanics 892, A6 (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7145454/). The specific compact potential, concentration matching, force kernel and streamfunction reference here are independently derived from the supplied apparatus.

Run `python scripts/validate_finite_layer_phoresis.py` from this stage with the pinned dependencies. `--generate` intentionally rewrites both calibration copies. Reports are `results/finite-layer-phoresis-r1-validation.json` and `results/finite-layer-phoresis-r1-local-controls.json`. Root owns Docker controls and all model evaluations; no model runs have been performed at this stage.
