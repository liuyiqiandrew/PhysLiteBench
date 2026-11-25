# Thermal unsteady sphere, revision 1

The source retains the full deterministic unsteady Stokes response and particle
inertia. Its error is to drive that response with a white random force of the
steady Stokes strength. The measured fluid is at equilibrium: the same fluid
momentum diffusion that produces frequency-dependent dissipation also changes
the spectrum of its fluctuating force. Driven mean-response calibration cannot
distinguish these models, even when it validates fluid history and added mass.

This is a thermal revision of the archived unsteady-sphere family. That task's
source omitted deterministic Basset and added-mass terms; both are exact here.
Hot Brownian rotation instead retains equilibrium noise coloring and freezes a
spatially weighted temperature. Hydrodynamic-fluctuations and gyroscopic-noise
omit spatial cross-noise or imaginary cross spectra. The present measurement is
a scalar equilibrium velocity spectrum. The general fluctuation-dissipation
background is shared and is not presented as a new independent principle.
The actual author-file audit and hashes are in `results/archive-audit.json`
within this stage. No earlier source, score, or interpretation is replaced.

## Apparatus and equations

The task fixes an unbounded no-slip sphere, uniform-temperature liquid,
equilibrated stationary velocity, no additional stochastic forcing, and the
linear incompressible continuum. Particle position may diffuse; it needs no
normalizable stationary law. For exp(-i omega t), let

    s = a sqrt(-i omega rho/eta), Re(s)>=0
    Z = 6 pi eta a [1+s+s²/9]
    mu = 1/[Z-i omega m].

The last impedance term is the fluid added mass, not part of the separately
specified particle mass. Calibration observes Re(F 0*mu) and Im(F 0*mu) under a
known real force amplitude. The source and oracle use exactly the same mu.

The two-sided angular-frequency velocity spectra are

    S_source = 2 kBT (6 pi eta a) |mu|²
    S_physical = 2 kBT Re(Z) |mu|².

Their ratio is 1+a sqrt(rho*|omega|/(2 eta)). The source is a well-defined
positive spectrum for the stable causal hydrodynamic response driven by an
independent white point force. It is a valid stochastic model but not the
equilibrium contraction of the stated fluid. This is not a missing factor of
two, incorrect complex convention, or numerical approximation to the same noise.

Viscosity is stored in mPa s, radius in µm, and angular frequency in units of
10^6 rad/s. The implementation converts these to SI; mean velocity returns in
µm/s and spectrum in µm²/s. The public spectrum is explicitly the integral over
all signed time. It is not a one-sided spectrum per hertz. Temperature dependence
of material properties is excluded by the stated ideal-fluid model.

The prototype checks the finite-frequency continuum regime: omega*a/c<=.008
for c=1500 m/s, driven Reynolds number<1.09e-5, conservative thermal Reynolds
number<.00140, and hbar*omega/(kBT)<2.19e-7. These support the declared effective
model. Neither the public readout nor the grading claims that incompressibility
persists to arbitrarily high microscopic frequencies or establishes bare-mass
equipartition by integrating beyond this domain.

## Independent reference

The verifier does not insert the closed impedance or the final FDT formula. It
solves for the coefficients of the two decaying exterior radial flow modes.
For x=r/a and unit translation, ur=f(x) cos(theta) and
utheta=-[f+x f'/2] sin(theta), with

    f=A/x³+B exp[-s(x-1)](sx+1)/x³,
    f(1)=1, f'(1)=0.

Pressure follows from the irrotational mode:
p=-i omega rho a A cos(theta)/2. Surface integration of pressure and viscous
traction gives the full complex impedance. The regular steady flow is used at
zero frequency. This independently verifies the deterministic response too.

The auxiliary strain components are

    Drr=f' cos(theta)/a
    Dtheta,theta=Dphi,phi=-f' cos(theta)/(2a)
    Dr,theta=-[f'+x f''/2] sin(theta)/(2a).

Local fluid stress covariance is
2 kBT eta (delta_ik delta_jl+delta_il delta_jk) times spatial and temporal delta
functions. Its reciprocal projection onto translation gives force spectrum
4 kBT eta integral D:D* dV. The verifier integrates this positive volume
contraction directly, including angular quadrature, and propagates it through
the mobility derived from surface traction. The equality of volume dissipation
and real surface impedance is an independently checked identity. The radial
integral extends to infinity; tightened quadrature verifies its tail.

## Calibration and controls

Calibration has 224 independent readings: 14 scalar settings repeated 16 times.
Two radii and DC plus three nonzero frequencies exercise steady, in-phase and
quadrature response. The identically zero DC quadrature is omitted. All
noiseless data are shared at every viscosity. DC response scales strictly as
1/eta, so the unknown is globally identifiable; dynamic rows also constrain
the complete supplied impedance. The fixed instrument standard deviation is
.002 µm/s, independent of the unknown or noiseless response. Generation seed is
261017; the 256-noise validation seed is 261019. True viscosity 1.07 mPa s is
private. The public starter has `None` and an ordinary unfinished fit method.

Both completed controls fit viscosity 1.070062639 mPa s with reduced chi-square
.978192 and relative error 5.85e-5. Nominal oracle hidden errors are at most
4.30e-5. The source diagnostic errors are .36325,.42765,.40949. The response/DC
anchor group passes for both. The standard group relative-RMS gate is .04,
calibration chi-square limit 1.5, and parameter tolerance 3%.

All 256 independent fits pass calibration and parameter checks. Every oracle
passes, and every source fails all three thermal diagnostic groups. Maximum
parameter error is .0001703, maximum oracle prediction error .0001249, and
minimum source diagnostic error .36320. The smallest hidden thermal signal is
.008577 µm²/s, so failure is not a near-zero cancellation. Nine full-range
noiseless calibration profiles, including both endpoints, recover the unknown
within 5.33e-9 relative error; each interior 701-point profile has one minimum.

Fifty-six domain cases compare the oracle with the independent fluid reference;
agreement is below 1.1e-13 relative. Refined angular/radial quadrature changes the
reference by less than 3.0e-15. Positive spectra, positive dissipation, no-slip,
DC equivalence, frequency reversal, conjugate mobility and the exact source
ratio are checked. Cold hidden reference time is .036 seconds. The full scientific
validator takes about 8.05 seconds.

Fresh isolated pytest copies give oracle 7/7 in .44 seconds and source 4 passes
plus three intended thermal-spectrum failures in .41 seconds. Only the private
metrics destination is redirected; models and tests are unchanged. The standard
600-second agent and 60-second verifier limits remain. Scientific validation
and model difficulty are separate: no model evaluation has occurred for this
package. Root owns future frozen controls and three-trial evaluations.

## Evidence and classification

The prototype, its raw 56-case report, archive audit and original derivation are
preserved in `prototype/`. Stage results contain the full 256 validation, local
controls, source provenance and independent peer report. The neutral instruction
permits replacing every prediction helper; public prose supplies the apparatus
and PSD definition without an FDT relation, diagnosis, or solver requirement.

When reviewing trials, retaining the white-force spectrum after calibrating the
complete mean response is the intended physical mistake. A trial that identifies
equilibrium fluid coloring but mishandles units, Fourier factors, complex
branches or numerical integration should be classified separately. Familiarity
of scalar equilibrium FDT may make this task easy; no difficulty result is
inferred from the large control separation.

Primary background:

- Hauge and Martin-Löf, [Fluctuating hydrodynamics and Brownian motion](https://doi.org/10.1007/BF01030307),
  Journal of Statistical Physics 7,259–281(1973), derives generally nonwhite
  particle forcing from local fluctuating-fluid stress.
- Franosch et al., [The colour of thermal noise](https://arxiv.org/abs/1106.6161),
  reports the thermal signature of hydrodynamic memory. Its optical trap is
  experimental background, not a boundary condition in this free-sphere task.
- Fouxon et al., [Lamb-type solution and properties of unsteady Stokes equations](https://arxiv.org/abs/2110.00387),
  gives the exterior pressure and Helmholtz-mode construction.

From the repository root, run the pinned environment with
`python staging/thermal-unsteady-sphere-r1/scripts/validate_thermal_unsteady_sphere.py`.
Add `--generate` only to deliberately replace both calibration copies.
