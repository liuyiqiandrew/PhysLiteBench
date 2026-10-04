# Nematic wall-torque prototype

Status: prototype only; no task package or model evaluation. The apparatus and
boundary-work convention passed conceptual review. Root reviewed the prototype source and authorized a full staged package; final
independent source review will accompany that package.

## Apparatus and readout

A nematic with a unit director has the explicitly defined elastic energy density

`f = K[(div n)^2 + |curl n|^2]/2 - K24 div[n div n + n cross curl n]`,

with positive unknown `K` and known `K24/K = 1/2`. The convention for the last
coefficient is part of the material definition; literature conventions vary.
There is no chirality, applied bulk field, appreciable director fluctuation, or
additional surface energy. Ideal anchoring actuators impose the director. Their
bare mechanical work is subtracted from the readout.

The calibration uses a flat slab of known thickness, with the director rotating
in its normal–tangent plane between two fixed tilts. The measured supplied work
per wall area per increase of the outer tilt is `K*(beta-alpha)/thickness`.
This identifies the one positive modulus over its full proposed range.

The distinguishing apparatus is a long cylindrical annulus, with inner radius
`a`, outer radius `R`, and end effects neglected. Both walls impose a tilt from
the positive axial direction toward the outward radial direction. The smooth,
defect-free state is prepared quasistatically from axial alignment. Its director
is `n = sin(theta(r))*e_r + cos(theta(r))*e_z`. Suggested controls are
`a in [.7,1.3]`, `R/a in [2,4]`, inner tilt `[.05,.15]`, outer tilt `[.3,.7]`
radians, and `K in [.8,1.6]`, in stated reference units.

A distributed anchoring actuator increments the outer tilt uniformly while
holding the inner tilt and geometry fixed. The measured supplied reversible work
is divided by outer-wall area and tilt increment. Equivalently this is the local
holding couple per area about each wall point's azimuthal tangent. Those local
torque vectors sum to zero around the cylinder; this is **not** the net torque
on a cylinder rotated about its axis. The liquid's torque on the actuator has the
opposite sign. Torque per area has units `K/length`.

## Physical approximation and calibration equivalence

The proposed completed source solves the exact nonlinear director profile and
computes the anchoring work from the ordinary bulk splay/bend energy. It drops
the divergence contribution from the mechanical readout because that contribution
does not affect the fixed-boundary bulk equation. This is a coherent bulk-energy
approximation: the prototype separately verifies its torque by differentiating
its own minimized energy. Every supplied geometry and tilt enters that solution.

The flat slab's divergence term is identically zero, so this source and the full
model have identical calibration at every allowed positive `K`. On the annulus,
both have the same profile, but they predict different boundary work. With
`s = log(r/a)`, both solve `theta_ss = sin(theta)*cos(theta)`. The full torque per
area is `K*theta_r(R)`; the source adds
`K*sin(beta)*cos(beta)/R`. This distinction is a physical boundary-work omission,
not a profile solver defect or a change in sign convention.

## Independent checks

The oracle prototype uses a collocation boundary-value solve. The reference
instead minimizes a sine-Ritz representation of the full energy. It constructs
the Cartesian divergence and curl, integrates their bulk contribution, subtracts
the divergence surface flux, and takes a centered fourth-order virtual-work
derivative after re-equilibrating the profile. It does not use the torque formula
or the radial differential equation. The source is checked by the same separate
energy calculation with its own bulk-only energy.

For this exact coefficient convention, the full density is
`K*|grad n|^2/2`, so it is pointwise nonnegative. The annular scalar energy is
strictly convex in the proposed tilt range because `cos(2*theta)>0`. The report
also records a sufficient stability bound for arbitrary azimuthal and axial
fixed-trace perturbations: radial Poincare in the logarithmic coordinate exceeds
the maximum director-gradient potential by at least `2.0607`. This establishes stability of the selected branch against these perturbations,
not uniqueness of every nonlinear or topological stationary state. The specified
quasistatic smooth preparation fixes the director branch without artificial
azimuthal confinement.

The 64 tested settings contain all 32 domain corners and 32 seeded interior
settings. The completed source differs by 46.0–110.7%; correct signals range
from .0392 to 1.2243. The independent reference agrees within `4.97e-7` relative,
and doubling the Ritz resolution changes the result by at most `4.34e-7`.
Calibration agrees within `1.4e-13`; the full-energy identity holds to `1.3e-15`.
At fixed gap and increasing radius, both predictions approach the flat-slab
answer and their relative difference tends to zero.

An initial supplementary planar-limit run exhausted the BVP mesh when the log
interval became very short. The final code solves on a unit interval with the
corresponding derivative scaling. This numerical change affected neither the
apparatus nor the source approximation; the complete prototype was rerun.

## Overlap and limitations

An archive/staging/task author-file search for nematic, liquid crystal,
saddle-splay, Frank energy, Helfrich, and Gaussian curvature found no existing
family. The closest conceptual analogy is the earlier pressure-wave boundary
traction task, but here the omitted constitutive term leaves the complete bulk
state unchanged and enters a director-anchoring work measurement. It is not a
renamed pressure loading problem. Model difficulty is unmeasured: a solver that
varies the full stated energy should recover the correct boundary work directly.

Primary background: [Lassoing saddle splay and the geometrical control of
topological defects](https://pmc.ncbi.nlm.nih.gov/articles/PMC4932960/) states the
same divergence convention and explains its boundary dependence;
[Interpretation of saddle-splay and the Oseen-Frank free energy in liquid
crystals](https://arxiv.org/abs/1901.06306) discusses the energy decomposition.
The proposed numbers and readout follow from the explicit apparatus above.

Reproduce with:

```sh
uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/nematic-wall-torque-prototype/prototype.py
```
