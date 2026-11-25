# Compensated waveguide revision 1: independent physics review

Reviewed 2026-10-11. **Approved for the frozen revision-1 evaluation. No blocking physics, calibration, reference, or tolerance issue found.** This approval concerns the strict thin-width observable in the public apparatus. It does not establish model difficulty or approve a finite-width variant. Docker controls are handled separately by the task author.

The review covers the current source and reports hashed below, including the corrected author-only bibliography. No task source, data, or grading file was changed during this review.

## Apparatus and limiting operator

The public specification fixes the physical Cartesian walls, centered transverse interval, Dirichlet end caps, kinetic coefficient, scalar applied compensation, ordering of the full spectrum, and order of the width limit. The unknown quantity is only a common dimensionless detector gain. The compensation therefore requires no knowledge of an unknown mass or coupling. All shape inputs stay fixed as epsilon tends to zero. Since the allowed shapes are smooth and bounded above and below, the transverse excited-state gap diverges uniformly.

There is no instability from the negative compensation. Transverse Dirichlet Poincare bounds the transverse kinetic quadratic form below by the exact term being canceled. The remaining form is bounded below independently of epsilon. The compensation is applied to every transverse mode, as a physical scalar potential should be.

For the normalized centered transverse ground state, set `u=y/(epsilon*a)` and `D=u*d_u+1/2`. Its real diagonal derivative connection vanishes. Direct integration gives

```
<chi, D chi> = 0,
<D chi, D chi> = pi**2/12 + 1/4.
```

Consequently the surviving longitudinal operator is

```
-d_xx + well*cos(2*pi*x) + (pi**2/12+1/4)*(a'/a)**2
```

with Dirichlet endpoints. The positive scalar term survives even as occupation of higher transverse modes vanishes. Its value depends on the centered-wall geometry, which is explicitly public. There is no unspecified curvature or holonomy term. The oracle implements this coefficient, logarithmic derivative, boundary condition, and gap indexing correctly.

## Independent reference

The reference starts with the full two-dimensional quadratic form in a normalized moving transverse basis. The derivative is `phi'*chi - (a'/a)*phi*Dchi`. Its two cross blocks have the correct transpose placement and signs; the quadratic dilation block integrates the full norm of `Dchi`, rather than replacing it by a projection onto the retained modes. The transverse block `pi**2*(n*n-1)/(epsilon*a)**2` applies the scalar compensation correctly. The reference does not insert the oracle's scalar coefficient.

The epsilon-squared extrapolation uses three positive widths and is checked with a smaller width window and larger longitudinal/transverse bases. This is an independent physical construction despite sharing sine functions and LAPACK with the effective oracle. Fixed-basis quadrature checks below also separate quadrature error from basis and width errors.

I reviewed the frozen validator report rather than regenerating the data. Its largest hidden-group change on reference refinement is `1.6851e-7` in relative norm. Oracle versus refined reference differs by at most `7.918e-9` in group relative norm. The 36 random/boundary cases have maximum absolute discrepancy `3.622e-6`; the oracle's 80-to-120-mode hidden refinement is below `8.140e-10`. These are ample checks for the 4% grading gate. Finite grids do not prove a uniform error bound over every continuous input, but there is no observed problematic region or small-gap normalization issue in the graded set.

## Additional reviewer checks

Using a separate numerical environment (NumPy 2.5.2, SciPy 1.18.0), I checked the following without changing any task files:

- Direct 160-point transverse integration gives coefficient `1.072467033424141`, versus analytic `1.0724670334241133`; the diagonal connection is `1.46e-15`.
- A second-order finite-difference tridiagonal longitudinal solver, independently Richardson-extrapolated from 4095 and 8191 interior points, agrees with the spectral oracle within `6.793e-8` absolute energy units across all 28 hidden preparations and 12 extreme boundary preparations.
- At fixed 64 longitudinal and 11 transverse modes, increasing only longitudinal quadrature from 220 to 400 points changes the challenging finite-width level-3 gap by at most `4.265e-9` over epsilon `.04,.02,.01`.
- The full finite-width reference at zero shape agrees with the separable oracle within `2.772e-11` for well strengths `-20,0,20` and level 3.

These supplement, rather than replace, the validator's pinned-environment and author-run Docker checks.

## Calibration, shortcut, and grading

Constant width gives exact separation, with independent Mathieu characteristic values for every calibration gap. Every such gap is positive; weighted least squares therefore identifies the single common gain uniquely. The complete shortcut uses the same correct local transverse energy and longitudinal confinement while dropping the kinetic cost of the changing normalized mode. It is exact throughout calibration, predicts nonzero gaps, and is a valid completed implementation of that approximation. The public starter is intentionally unfinished only in fitting; it does not expose the missing scalar term.

Both controls fit gain `1.07001426896534` and reduced chi-square `0.9893561857`. The 256 independent-noise checks recover gains between `1.06996347985` and `1.07004666782`; their maximum reduced chi-square is `1.30817`. The largest oracle hidden error under these noise draws is `4.377e-5` relative. The weakest shortcut diagnostic group remains above `0.13308` relative. This comfortably separates the 4% gate from both numerical and measurement uncertainty. The constant-width hidden group appropriately passes both controls. The smallest graded signal exceeds 25 energy units.

The verifier uses its own calibration copy and frozen reference values, checks the documented output shape and finiteness, and writes metrics and a numeric reward. Public/private calibration bytes match. The Dockerfile copies only public environment files; the physical solution, hidden preparations, reference, true gain, and author explanation are outside the agent image. Standard 600/60-second limits and submission artifact collection are preserved.

The earlier oracle refinement miss is preserved in the development report. Increasing the oracle basis before trials resolves that numerical check without changing experimental data, hidden cases, or grading. This is ordinary prerelease convergence work, not evidence of a model physics failure.

The intended failure is omission of the finite kinetic scalar term. An agent that derives that term but then implements its coefficient or eigensolver incorrectly must be classified separately as a mathematical or numerical failure. The task is related to the archived geometric-rotor family, but those controls retained the scalar term and missed transport; this apparatus isolates a different physical approximation. No empirical difficulty claim is made in this source review.

## Reviewed SHA-256 hashes

| File | SHA-256 |
|---|---|
| `tasks/compensated-waveguide/AUTHOR.md` | `b452938ad1a3ad1c23dc5770772e01678019a10d8eaabfdfea069b2f32f10c54` |
| `tasks/compensated-waveguide/instruction.md` | `6a5ef18a542d5fd42298ccab151355ecf224974f3b465d9a6299768988739b62` |
| `tasks/compensated-waveguide/task.toml` | `cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af` |
| `tasks/compensated-waveguide/environment/README.md` | `d60e67e501c1f04d8e04ddb1139cb27ef5cb66a3e932892e15e1100ad4c85b1a` |
| `tasks/compensated-waveguide/environment/model.py` | `202403bf583303aefddc2bce496b4477795f6bc0c814e3944892d94369a56f7b` |
| `tasks/compensated-waveguide/environment/Dockerfile` | `375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70` |
| `tasks/compensated-waveguide/environment/test_public.py` | `9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9` |
| `tasks/compensated-waveguide/environment/data/calibration.json` | `755d1f5e0e636b3013a4e9620139629222bac826864be5154c9ef2a7d5e3844f` |
| `tasks/compensated-waveguide/solution/model.py` | `52545ec4cbe1c0d156d43316c57bbe3ff38430135113d8401471372fae528ed4` |
| `tasks/compensated-waveguide/solution/solve.sh` | `bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644` |
| `tasks/compensated-waveguide/tests/reference.py` | `4ed7fbc1d4999ad102ac1fa7ccbe19538107b29d150975647811d8b34cbea52d` |
| `tasks/compensated-waveguide/tests/test_hidden.py` | `fbd292c680fef4ece788e3cd1f48071bd7063e997d26fd1015235bdc2d843ca1` |
| `tasks/compensated-waveguide/tests/test.sh` | `74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c` |
| `tasks/compensated-waveguide/tests/metadata.json` | `b9a20f300e1f93cb9193564b3af8439692e8525b3f207e92d4ba1644b7e7b661` |
| `tasks/compensated-waveguide/tests/data/calibration.json` | `755d1f5e0e636b3013a4e9620139629222bac826864be5154c9ef2a7d5e3844f` |
| `tasks/compensated-waveguide/tests/data/hidden_truth.json` | `46f1078f4b1cc328c2a3542b63308cf8ce7a8f246c75a235fe12d8c4fa3b5ae7` |
| `scripts/compensated_waveguide_baseline.py` | `cb38ff501aabaf6c53f4402a0c77c864c0ec069bc3945e5331b71003d36afe9d` |
| `scripts/validate_compensated_waveguide.py` | `ee47d91b746956e2d4186464c517167d1b8b2b17f2174606ee41ebb06c37a48b` |
| `results/expansion-compensated-waveguide-r1-validation.json` | `7e8ed167d12d0761e9df0f7700cc5b0e0b05b49b4c31e44e2a832d68195d4d46` |
| `results/expansion-compensated-waveguide-r1-development.json` | `e46a7acc30772813b659ee36921e3ee11e87448a210c2130a205d2ed8e7a8cb5` |
