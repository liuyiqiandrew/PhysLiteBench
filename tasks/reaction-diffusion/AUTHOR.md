# Reaction-diffusion, revision 7

Revision 7 replaces the independent-salt starter with a completed coupled
linear-response approximation. The apparatus, calibration, physical oracle,
independent verifier, 5% prediction threshold, runtime limits, and approved
neutral instruction are unchanged. The old task, controls, validator, and all
three neutral-instruction trial reviews are preserved in
`archives/reaction-diffusion-neutral-v1/`. Those three trials passed. Revision 7
then failed its first unhinted Luna-high trial, MQyf6fe, under the approved
conditional protocol; no follow-ups were requested after that zero. The final
submission retained the mean-composition closure. At its fixed fitted D, the
local-conductivity repair passes all hidden groups. The review and causal
diagnostic are `results/neutrality-brownian-trial-reviews.json` and
`results/reaction-r7-MQyf6fe-fixed-parameter-repair.json`. This result is 0/1,
not evidence of 0/3.

For concentrations A and B, electroneutrality gives C=A+B. Enforcing zero
charge current in the three Nernst–Planck fluxes gives the dimensionless
potential gradient `(A_x-2 B_x)/(3 A+6 B)`. The correct local diffusion matrix is

```
D [[1+r/3, -2r/3], [4s/3, 4(1-2s/3)]],
r=A/(A+2B), s=B/(A+2B).
```

The starter evaluates r and s at the conserved spatial means, then propagates
both coupled concentration modes with the blocking-boundary cosine operator.
This is the exact transport Jacobian around a uniform background. It retains
cross-diffusion and both species conservation laws. Its eigenvalues are real
and positive. The ideal-ion free-energy Hessian at the background symmetrizes
the matrix, so the associated quadratic free energy decreases.

Every calibration preparation contains one salt only. On these preparations,
the closure is exact: the occupied channel diffuses with 4D/3 or 8D/3. These
curves identify D. In mixed finite-amplitude preparations, the evolving local
conductivity differs from the mean-composition conductivity; even an initially
uniform species develops a profile. The existing hidden cases distinguish this
nonlinear physical transport from the completed linear-response approximation.
The approximation is not asserted to preserve positivity for every arbitrary
finite-amplitude input. It stays nonnegative in all scored cases and 47
additional pure/mixed profiles checked by the validator.

`python scripts/validate_reaction.py` reads the existing data and performs 256
noise realizations with seed 1729. The report is
`results/reaction-r7-validation.json`; raw arrays are in
`jobs/reaction-validation-r7/noise-controls.npz`. Results:

- Both controls recover D to better than 0.08% in all 256 realizations and pass
  the unchanged calibration check.
- The physical oracle passes all 256. Its largest hidden error is 0.001904.
- The coupled shortcut fails all three groups in all 256. Its smallest group
  errors are 0.07536, 0.11130, and 0.08507, against the unchanged 0.05 limit.
- An independent finite-volume integration of the *linear approximation* agrees
  with its cosine propagator to 2.06e-9 in concentration. The separate physical
  verifier evolves all three ions with local zero-current fluxes on a finer
  grid; further refinement changes normalized predictions by at most 6.42e-5.
- The entropy symmetrizer is symmetric to 1.43e-14 at 361 positive backgrounds.
  The 47 profile checks conserve each species mean to 2.23e-15 and show no
  increase in the background quadratic free energy.

Local checks run the original public and private pytest files against each
completed model; results and hashes of all unchanged files are in
`results/reaction-r7-local-controls.json`. The parent coordinates fresh Docker
controls and agent evaluations. Peer source review is recorded separately in
`results/constraint-hardening-physics-review.json`.
