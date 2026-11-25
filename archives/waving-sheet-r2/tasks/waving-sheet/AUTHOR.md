# Viscoelastic waving sheet, revision 2

This is a substantive revision of the waving-sheet family. R1 had two physical
passes and one physical failure among three completed trials. Three earlier
Docker interruptions were unscored. All six attempts, the fixed-viscosity
diagnostic and the original Newtonian task remain in `archives/waving-sheet-r1`.
Revision 2 has not been evaluated by a model.

The material is now Oldroyd-B, with known solvent fraction beta=.25 and polymer
relaxation time lambda=1 s. The common unknown is the total zero-frequency
kinematic viscosity nu0. The same anchored extensible sheet and laboratory
readouts are used. All coefficients refer to the stationary small-amplitude
problem; no finite-amplitude Hookean extension instability is invoked.

The source solves the full linear viscoelastic response and retains both the
moving-wall expansion and the mean fluid momentum convection omitted in r1.
Its polymer law is linear laboratory Jeffreys stress at both perturbation
orders. This is a coherent linear-response approximation extended to a
quadratic observable. It is not a generally objective nonlinear material law.
It should not be described as a complete nonlinear Oldroyd-B calculation.

With exp(-i omega t), the kinematic complex viscosity is
`nu_star=nu0*[beta+(1-beta)/(1-i*omega*lambda)]`. The first-harmonic streamfunction
has decaying modes k and `s=sqrt(k^2-i*omega/nu_star)`, with positive real part.
Its boundary values are psi(0)=omega/k and psi'(0)=0. Both controls use the same
exact harmonic coefficients for every calibration input.

For `G_ij=partial_j u_i`, the first-order Hookean conformation satisfies
`C1=(G1+G1^T)/(1/lambda-i*omega)`. The nonlinear mean polymer stress is
`N=eta_p*<G1 C1+C1 G1^T-u1.grad(C1)>`. With no far-field imposed shear, the mean
momentum balance is `eta0*U2'+Nxy=rho*<u1*v1>`. Pumping is therefore the wall
Taylor velocity plus the integral of `(rho*<u1*v1>-Nxy)/eta0`. The source keeps
the whole Reynolds momentum flux and omits only N. The signs and factors are
independently checked in the preserved prototype.

The oracle integrates N from exact harmonic tensors. The verifier solves the
harmonic vorticity boundary equations, constructs the conformation forcing
componentwise, and solves the locally forced mean momentum BVP. It uses no
exact mode function or integrated pumping formula. The shared tensor kinematics
represent the same constitutive definition; this is independent numerical and
momentum-balance validation, not two different microscopic derivations.

The reported liquid density is 1000 kg/m^3. Dividing all stresses by density
leaves the equations in kinematic units. Harmonic outputs divided by amplitude
are in 1/s; mean pumping divided by amplitude squared is in 1/(mm*s). The
solvent and polymer fractions multiply the same unknown total viscosity.

Calibration has 48 distinct real-valued settings, each repeated six times:
two wave numbers, three frequencies, two heights and all four signed harmonic
components. Sigma=.002 /s is a fixed instrument uncertainty independent of the
response and viscosity. Public and private files are identical. The fresh data
seed is 285021; the 256-noise seed is 285027. The fit is unfinished in the public
starter and viscosity starts as None. No true parameter appears in public files.

The identity `U+iV=omega*exp(-s*h)` makes the ratio at heights .5 and .6 mm
injective. The maximum possible phase difference is below .68851 radians, so
there is no logarithmic branch aliasing. Recovering s gives nu0 uniquely. The
25 noiseless calibration profiles checked over [.7,1.4] each have one minimum
and recover the parameter within 6.10e-9 relative, including endpoint recovery.
This does not claim arbitrary noisy objectives have a unique minimum.

At true nu0=1.05, the checked-in fit is 1.05027292, with chi-square .94344.
The nominal physical pumping errors are below 2.04e-5; the harmonic anchor error
is .000132. Source pumping errors are 5.3975, 4.0691 and 5.1903 at the unchanged
.04 group gate. All 256 noise draws pass calibration and parameter checks;
all oracle predictions pass and every source fails the three pumping groups.
Worst oracle error is .000301, and the smallest source diagnostic error is
4.06845. Over the 25 fitted-parameter controls, its smallest group error is
3.4366. Physical scored pumping is at least .59739, so the differences are not
created by normalization at a vanishing response.

The public frequency range remains [.5,8] /s. Scored pumping uses ordinary
controls in the elastic-response range 4–8 /s; no added readout or narrow
coordinate restriction is introduced. The prototype retains the broad-domain
near-equivalence at some other settings, so no universal separation is claimed.
Twenty-four package domain controls (12 structured and 12 random) give maximum
oracle/BVP error 4.50e-9 and harmonic error 3.94e-11. Refinement changes pumping
by 2.20e-9. Harmonic wall work equals volume dissipation within 8.9e-16 relative;
mean momentum residual is at most 1.05e-9 in the reported normalization.
Wave reversal also passes. Newtonian and creeping limits are checked in the
preserved generalized prototype; optional beta/lambda controls are author-only,
while the task fixes their stated values. Current task predictions match that
prototype within 8.3e-15 across the sampled public domain.

Local isolated pytest gives oracle 7/7 in 1.59 s and source 4 passes with the
three intended prediction failures in 1.41 s. Harmless local pytest cache-path
warnings are preserved in the raw report. No Docker/model control is claimed.
The scientific validator takes about 23 s; the cold private reference takes
about .81 s. Agent/verifier limits remain 600/60 seconds.

The first prototype's far-tail exponential overflow and NaN report are retained
under `prototype/history/first-tail-evaluation`. The final implementation uses
weighted exponentials and explicit finite-value checks. This development defect
is not an evaluated physical failure. The rejected corotational alternative is
also recorded: at quadratic order in planar incompressible flow its extra
stretching stress is isotropic, so it does not change pumping. Archived
dumbbell-stress tests affine Hookean transport; FENE r1/r2 instead test nonlinear
spring preaveraging. These are related polymer theories, not independent general
principles. R2 changes the constitutive material and failure from r1's Newtonian
momentum omission.

[Lauga, Propulsion in a viscoelastic fluid, Phys. Fluids 19, 083104 (2007)](https://arxiv.org/html/cond-mat/0703548)
provides the upper-convected perturbation analysis and creeping pumping ratio;
Appendix B explains the objective-derivative degeneracy. The finite-inertia
calculation here is independently derived and checked. A trial that retains the
linear constitutive closure can establish the intended physical error. A trial
that derives the mean polymer forcing but fails to implement it must instead be
classified from its full trajectory as mathematical or numerical when appropriate.

Reproduce scientific validation without data regeneration using the pinned
NumPy 2.3.3/SciPy 1.16.3 environment and
`python scripts/validate_waving_sheet.py` from this stage. Use `--generate` only
for an intentional new calibration revision. No empirical difficulty claim is
made before the frozen agent batch is reviewed.
