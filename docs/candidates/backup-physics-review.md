# Physics review of six optional revision-1 candidates

Reviewed 2026-10-01 independently of the task authors. The review covered public
apparatus descriptions, starter and oracle predictors, independent references,
hidden metrics, hints, validators, and their local reports. No physics correction
is requested. Harbor and Luna outcomes are separate evidence.

The reviewer evaluated every hidden case at the true parameters using Python
3.13, NumPy 2.3.3, and SciPy 1.16.3. Both predictors are exactly equal throughout
calibration. The observed hidden differences are:

| Task | Shortcut grading error | Maximum oracle/reference absolute difference |
|---|---:|---:|
| Spatial diffusion | 0.158–0.245 absolute moment RMSE | 3.42e-6 |
| Hydrodynamic beads | 1.000 relative RMSE | 9.32e-13 micrometers |
| Poroelastic solid | 0.556–0.614 relative RMSE | 1.78e-15 MPa |
| Competitive adsorption | 0.260–0.403 relative occupancy RMSE; 1.000 displacement error | 1.12e-16 |
| Compensated conductor | 0.273–0.639 relative RMSE | 1.14e-13 A/m² |
| Fluid coexistence | 0.139–1.379 relative RMSE | 1.98e-16 reduced pressure |

Spatial diffusion uses an absolute threshold of 0.03. The other tasks normalize
each group's RMSE by its reference RMS and use a threshold of 0.04. All require
calibration reduced chi-square below 1.5 and each fitted parameter within 3%.
Adsorption fits two affinities, so its chi-square denominator subtracts two
parameters; the others subtract one.

## Spatial diffusion

The public specification provides uniform temperature, local mechanical drag,
the same local fluctuation-dissipation relation, the overdamped limit, constant
accessible volume, and no conservative force. This determines the physical
process; merely stating a spatial diffusion coefficient would not suffice.
The physical flux is `-D(x)*dp/dx`, equivalent to Ito drift `D'(x)`. The shortcut
instead uses zero-drift Ito motion and flux `-d(D(x)*p)/dx`.
[Physical basis](https://arxiv.org/abs/cond-mat/0005004).

For `D=D0*(1+a*cos(k*x))`, the physical Fourier generator has off-diagonal
coefficients `a*n*(n-1)/2` and `a*n*(n+1)/2`, including no source from the zeroth
moment to the first. The symmetric tridiagonal oracle is correct. The shortcut's
similarity transform correctly realizes `a*n^2/2` couplings and its nonuniform
stationary density. A separate direct 80-mode affine generator agrees with that
shortcut to 9.4e-15 on a contrast-0.8 check.

Zero-contrast calibration identifies diffusivity through decay of modes 1 and 2.
Hidden tests include nonuniform initial states and an initially uniform state,
so the task tests transients as well as equilibrium. The finite-volume reference
conserves mass and preserves a uniform equilibrium. The inspected convergence
report gives maximum errors 1.37e-5, 3.42e-6, and 8.54e-7 for 256, 512, and 1024
cells. Its separate Ito finite-volume check confirms the shortcut is a valid
implementation of its assumed process.

## Hydrodynamic beads

The public apparatus explicitly restricts the model to constant mobility,
isolated Stokes self drag, and leading far-field point-force response at the
nominal separation. This fixes the intended approximation; finite-size or
configuration-dependent corrections are not silently introduced in grading.
The longitudinal cross mobility is `1/(4*pi*eta*r)`, with self mobility
`1/(6*pi*eta*a)`. At the smallest separation, their ratio is 0.15, so the matrix
is positive definite throughout the domain.
[Hydrodynamic coupling reference](https://arxiv.org/abs/cond-mat/0307282).

Equilibrium covariance is `kBT*K^-1`, independent of mobility when thermal
noise obeys the same fluctuation-dissipation relation. Thus equilibrium variances
identify stiffness without distinguishing the shortcut. After a trap shift,
the mean is `shift-exp(-M*K*t)*shift`; elementwise multiplication by the stiffness
vector in the oracle correctly scales mobility columns. Hidden observations of
the unshifted bead expose the transient response that independent drag misses.
Initial means and shift signs are unambiguous.

The micrometer/pN viscosity conversion is correct: `1 Pa s = 1 pN s/um^2`.
The independently integrated SI reference agrees to 9.32e-13 micrometers.
The author's report also checks covariance balance, normal-mode decay,
positive mobility eigenvalues, zero initial displacement, and the limit of
large separation.

## Poroelastic solid

The strain, stress, pore-pressure, and tensor-shear signs are explicit. Sealed
uniform loading implies `0=alpha*tr(strain)+p/M`. Adding `-alpha*p*I` to the
drained skeleton stress gives `Ku=Kd+alpha^2*M`, with unchanged shear modulus.
The oracle uses this bulk modulus; the reference solves for pressure and all
six stresses without eliminating pressure first.
[Biot constitutive relations](https://osupdocs.forestry.oregonstate.edu/index.php?title=Poroelasticity_Calculations).

Every calibration has zero volumetric strain, including normal deviatoric
strains as well as shear, so pressure vanishes exactly and the common slope
identifies Young's modulus. Hidden compression, constrained extension, and mixed
strain remain within the declared small-strain single-phase model. Gauge pressure
is defined relative to the reference state. The local checks cover fluid-content
conservation, rotational covariance, positive added fluid energy, and decoupling
when the Biot coefficient is zero. The author's integer-zero-strain dtype
correction was inspected and independently checked.

## Competitive adsorption

The three site states are empty, occupied by A, and occupied by B. The public
reservoir and equal-site assumptions yield weights `1`, `KA*cA`, and `KB*cB`.
Normalizing them gives both occupancies. The independent reference solves the
two adsorption/desorption balances plus total site conservation.
[Shared-site balance derivation](https://sites.engineering.ucsb.edu/~jbraw/chemreacfun/ch5/slides-kinetics.pdf).

Separate pure-species curves identify both affinities while making independent
Langmuir predictions exact. Hidden mixtures test equilibrium occupancy and
loss after a competitor is added. The displacement protocol clearly defines its
denominator as all sites, with the original species' concentration held fixed.
The concentrations and affinity units make all binding weights dimensionless.
The shortcut's total hidden occupancy stays below 0.858: failure requires the
shared-site physics, not merely detecting occupancy above one.

## Compensated conductor

Two Drude populations share one transverse field. With common mobility `mu`,
total density `n`, imbalance `d=(n_positive-n_negative)/n`, and `b=mu*B`,
adding their tensors and imposing total `Jy=0` gives
`Jx=e*n*mu*(1+d^2*b^2)/(1+b^2)*Ex`. The independent five-variable solve
correctly applies both Lorentz-force balances and the total-current condition.
Equal densities cancel the total Hall term during calibration, including
nonzero magnetic fields; the zero-field calibration also identifies mobility.
Hidden imbalanced preparations require the shared transverse field.

Insulating walls alone would be insufficient if each carrier population were
separately conserved: both can have a neutral transverse particle flux.
The public specification addresses this by explicitly allowing charge-conserving
pair generation and recombination, fixing the bulk density difference by doping,
and measuring only the uniform central bulk of a wide bar. Recombination/diffusion
edge layers and their current contribution are explicitly excluded. Momentum
relaxation is faster than pair equilibration. These assumptions define the
local bulk approximation and avoid silently grading an unmodeled edge effect.
[Two-component transport and edge recombination](https://arxiv.org/abs/1612.02439).

The inspected validator checks positive Joule/friction dissipation, vanishing
total transverse current, reversal symmetries, and single-carrier, zero-field,
zero-drive, and compensated limits. Its calibration-loss scan has one minimum
on 1001 points. Both controls pass calibration and parameter checks, and the
oracle/shortcut hidden separation survives every tested noise realization.

## Fluid coexistence

The supplied homogeneous free energy `f=-T*ln(v-1)-a/v` has pressure
`P=T/(v-1)-a/v^2`. The public apparatus permits macroscopic density separation
and requires global equilibrium at fixed mean volume. The equilibrium free energy
is its lower convex envelope: a common tangent supplies equal pressure and
chemical potential, with a constant pressure between the liquid and vapor
volumes. The shortcut evaluates the homogeneous pressure at the mean volume.
[Maxwell construction and phase equilibrium](https://www.damtp.cam.ac.uk/user/tong/statphys/statmechhtml/S5.html).

All calibration temperatures exceed `Tc=8*a/27` for every allowed attraction,
so calibration identifies `a` without encountering phase coexistence. All hidden
mean volumes lie inside their binodal intervals. The oracle solves an equal-area
pressure equation with separate stable-branch volume roots; the independent
reference constructs a lower convex hull and refines common-tangent contacts.
The reference permits phase volumes outside the input mean-volume range, as
required physically.

The inspected report gives contact agreement below 3.47e-13 and numerical
equal-area residual below 3.80e-16. A tiny window within relative 1e-7 of `Tc`
uses the homogeneous critical limit to avoid an unresolved vanishing loop; the
reported near-critical pressure check differs from critical pressure by
4.45e-11. The reviewer also checked predictions at attraction bounds, domain
temperature/volume corners, and temperatures approaching criticality, with no
nonfinite output or root exception. Hidden tests are well outside this window.

## Local evidence and scope

The inspected reports are under `jobs/<task>-validation/summary.json`; the
standalone reproduction commands are `scripts/validate_<task_name>.py`.
Each report retains seeds, fitted controls, and 256 noise realizations.
Spatial diffusion, hydrodynamic beads, and fluid coexistence evaluate hidden
predictions at observed parameter extrema after their calibration Monte Carlo.
Poroelasticity, adsorption, and the compensated conductor fit both controls and
evaluate every hidden group on every noise
realization. Both procedures pass their declared checks; they are different
validation scopes and should not be described interchangeably.

All six images explicitly copy public files only. References, private
calibration copies, author notes, and hints remain outside the image. These
tasks are ready for fresh Harbor control runs and frozen unhinted/hinted batches.
