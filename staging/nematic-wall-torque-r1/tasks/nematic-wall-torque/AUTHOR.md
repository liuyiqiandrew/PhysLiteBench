# Nematic wall torque, revision 1

This task tests whether a correct equilibrium director field is enough to predict
the reversible work of its boundary anchoring actuator. The supplied prediction
solves that field correctly but evaluates work using only the bulk splay/bend
energy. The stated divergence contribution leaves the fixed-boundary field
unchanged and changes the boundary work. Agent difficulty has not been measured.

## Constitutive convention and measured work

The public equation is

`f = K[(div n)^2+|curl n|^2]/2 - K24 div[n div n+n cross curl n]`,

with `K24=K/2`. For a unit director this is exactly `K|grad n|^2/2`, so the
full energy is pointwise nonnegative. The explicit convention matters: some
papers include an additional factor of one half in their named saddle-splay
coefficient. The task does not rely on a symbol convention left unstated.

For the prepared annular branch,
`n=sin(theta(r))*e_r+cos(theta(r))*e_z`. At one azimuth the nonzero Cartesian
derivatives are `n_x,x=cos(theta)*theta_r`, `n_z,x=-sin(theta)*theta_r`, and
`n_y,y=sin(theta)/r`. Consequently

```
div[n div n+n cross curl n] = 2*sin(theta)*cos(theta)*theta_r/r,
F/axial_length = pi*K*integral(theta_s^2+sin(theta)^2, ds),
s = log(r/a).
```

The equilibrium equation is `theta_ss=sin(theta)*cos(theta)`. The source solves
this same equation. Its energy has the additional endpoint term
`pi*K*(sin(beta)^2-sin(alpha)^2)`, which does not affect that fixed-trace solve.
Differentiating the full relaxed energy in outer tilt and dividing by `2*pi*R`
gives the supplied holding work `K*theta_r(R)`. The source adds
`K*sin(beta)*cos(beta)/R`.

This is work supplied by a distributed tilt actuator, not net rigid-body torque
about the cylinder axis. Its local rotation vector is `e_phi*d beta`; the local
couple vectors integrate to zero but their work does not. The liquid's reaction
couple has the opposite sign. Ideal imposed anchoring and the subtraction of bare
actuator work leave no extra unspecified surface contribution.

Numerical `K` in pN divided by radius in micrometres is numerical work per area
in microjoules per square metre. The physical-unit package therefore uses
`K in [8,16]` pN; the preserved prototype used reduced moduli `[.8,1.6]`. This is
a factor-of-ten linear response rescaling, with all geometry, energy ratios and
relative separation unchanged. The unknown true modulus is 11.7 pN.

## Calibration, preparation and stability

For the flat slab the divergence contribution is zero, and both models predict
`K*(beta-alpha)/thickness`. The 24 positive calibration settings identify the
positive modulus uniquely: the weighted least-squares objective is a strictly
convex quadratic. There are six independent readings per setting, for 144 total.
The fixed instrument standard deviation is .003 µJ/m² for every reading and
does not encode the noiseless response or the fitted parameter.

The public preparation raises both tilts proportionally from axial alignment
while relaxing along the smooth stable defect-free state. The scalar energy is
strictly convex on the specified tilt interval since `cos(2*theta)>=cos(1.4)>0`.
For arbitrary zero-trace vector perturbations, the full second variation is
`K integral(|grad v|^2-|grad n|^2 |v|^2)`. In logarithmic radius, radial Poincare
gives `pi^2/log(4)^2`, while the director-gradient potential is bounded by
`(.65/log(2)+log(4)/2)^2+sin(.7)^2`. Their difference is greater than 2.06.
The maximum principle keeps the tilt between zero and .7; the derivative bound
uses `|theta_ss|<=1/2` and the mean endpoint slope. Along the preparation the
smaller tilts obey the same bounds.

This establishes local stability of the prepared branch to arbitrary azimuthal
and axial fixed-trace perturbations. It does not claim uniqueness of every
nonlinear three-dimensional or topological stationary state. No artificial
symmetry constraint is imposed on a solver.

## Independent reference and validation

The solution uses a collocation BVP on a rescaled unit interval. The private
reference instead minimizes the full Cartesian div/curl energy minus its
surface flux using a sine-Ritz basis. It re-minimizes after varying the outer
angle and computes the work with a fourth-order centered derivative. It does not
use the radial differential equation or the boundary torque formula. The source
is separately checked against virtual work from its own bulk-only energy, so
its failure is not an algebraic error implementing that approximation.

The report covers all 32 domain corners plus 32 random states, 32-to-64-mode
reference refinement, the full energy identity, modulus/length scaling and
full-range parameter recovery. Independent torque agreement is below `5e-7`
relative; reference refinement is below `3.6e-7`. The physical torque is positive
and ranges from .392 to 12.243 µJ/m² over those checks. Source relative errors
range from .460 to 1.107. The preserved prototype also verifies convergence to
the flat result at fixed gap and increasing cylinder radius.

All 256 independently noisy calibrations pass parameter/calibration gates for
both completed controls. The correct model passes every hidden group; the
source fails all three annular groups. The largest correct hidden error is
.000132 and the smallest incorrect annular error is .6117 at the unchanged .04
group normalized-RMS threshold. Nominal annular source errors are .612, .831 and
.667. Local isolated pytest controls give 7/7 for the oracle and four passes plus
three intended prediction failures for the source, both under one second.

The public fit remains unfinished and the unknown defaults to `None`; the
separate completed controls perform the same constrained weighted fit. The
instruction is the exact neutral implementation/replacement instruction.
Private true values, references, hints and this document are excluded from the
agent Docker image. All hidden experiments obey the public ranges and readout.

Reproduce scientific checks with:

```sh
uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/nematic-wall-torque-r1/scripts/validate_nematic_wall_torque.py
```

Use `--generate` only to intentionally replace both calibration copies. Seeds,
reports and source provenance are in the stage's `results/` directory. Docker
controls and Luna evaluations are scheduled separately by the root agent.

## Relation to earlier tasks and trial classification

A broad author/assessment archive search found no nematic, saddle-splay, Frank,
Helfrich or Gaussian-curvature family. The earlier pressure-surface-wave task
also involved a boundary observable, but its wrong external traction changed
the wave spectrum. Here the full bulk state is already identical and the
constitutive divergence contribution changes actuator work. This is a distinct
apparatus and approximation, not a renamed pressure-wave task.

A failure retaining the bulk-only anchoring work is a physical modeling failure.
An attempted full-energy variation with an incorrect factor, unit, angle sign,
or failed minimizer is an implementation or mathematical failure. Inspect the
complete public trajectory, including discarded diagnostics, before assigning
causality. Model scores cannot be inferred from scientific control separation.

Primary background: [Lassoing saddle splay and the geometrical control of
topological defects](https://pmc.ncbi.nlm.nih.gov/articles/PMC4932960/) gives the
explicit divergence convention and boundary dependence;
[Interpretation of saddle-splay and the Oseen-Frank free energy in liquid
crystals](https://arxiv.org/abs/1901.06306) discusses its energy decomposition.
