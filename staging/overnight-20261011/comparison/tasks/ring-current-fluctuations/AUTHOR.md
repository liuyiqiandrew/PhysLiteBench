# Ring current fluctuations, revision 1

This task tests the assumption that a current-conditioned macroscopic density
profile stays homogeneous. That assumption gives the correct calibration law but
misses the nonuniform profile that dominates sufficiently suppressed currents in
a strongly driven, half-filled exclusion ring.

The finite-ring conditioned stationary law remains translation invariant. Its
unaligned one-point mean density is flat. The physical alternative is a
nonuniform macroscopic profile with an arbitrary translated center, not a claim
that the unaligned ensemble mean becomes nonuniform. At half filling the
dominant profile has zero propagation speed.

## Normalization and model

For a successful jump, the counted increment is `+1/L` or `-1/L`. The tilted
finite-ring generator therefore multiplies off-diagonal rates by
`exp(+/-lambda/L)` and leaves the escape diagonals unchanged. Its largest
eigenvalue, multiplied by `L` after taking the long-time limit, defines the
requested SCGF. The canonical particle number, successful-jump convention,
microscopic clock and ordered limits are public.

The hydrodynamic diffusivity is `D` and mobility is `2D rho(1-rho)`. At half
filling the homogeneous-profile prediction is

```
psi_flat = D lambda(lambda + 2E)/4.
```

The common rate prefactor multiplies the entire generator, so every SCGF scales
exactly linearly with `D`. There is no additional fitted effective temperature,
clock factor, particle density or measurement weight.

The homogeneous branch loses stability when `E > 2*pi` and
`abs(E+lambda) < sqrt(E**2-4*pi**2)`. In units with `D=1`, write
`z=2rho-1`, `eta=E+lambda` and `H=<1/(1-z*z)>`. The stationary functional is

```
psi = max_z [eta**2/(4H) - E**2 <1-z*z>/4
             - <z_prime**2/(1-z*z)>/4],  <z>=0.
```

The one-period solution obeys
`z_prime**2=E**2*(a*a-z*z)*(b*b-z*z)`. With `m=(a/b)**2`,
`K=ellipk(m)`, `b=4K/E` and `H=Pi(a*a,m)/K`, one scalar equation fixes the
profile:

```
abs(eta) = E*sqrt((1-a*a)*(1-b*b))*H
psi/D = 4*K*K*(1-m-2*ellipe(m)/K).
```

The oracle uses real quadrature for `Pi` and scalar bracketing. Its only change
from the reviewed prototype enforces the exact endpoint `eta=0` when `b=1`,
both in the bracket residual and the returned endpoint convention. This prevents
a possible floating-point bracket failure arbitrarily near the center of the
public bias interval. It does not change the physical branch or hidden cases.
Fixed checks cover seven fields and tiny positive/negative `eta`; the retained
domain study itself did not exhibit that bracket failure.

[Simon, arXiv:1011.3590](https://arxiv.org/abs/1011.3590), Section IV.D and
Appendix A, gives the exact half-filled correspondence between this stationary
profile and the selected thermodynamic Bethe branch. Its thermodynamic
resolvent assumption and discussion of further possible branches remain
limitations, not theorems established by this task.
[Espigares, Garrido and Hurtado, PRE 87, 032115](https://doi.org/10.1103/PhysRevE.87.032115)
supports the density instability and the half-filled zero-speed branch with
microscopic simulations. The task uses this published physical solution; its
numerical checks do not prove unrestricted global optimality.

## Calibration and controls

The public domain is `E in [0,12]`, `lambda in [-2E-2,2]`, and `D in [.8,1.2]`.
Calibration has fields `0,1,2,4`, with biases `-E-.75,.5,1.5` at each field and
16 independent measurements per setting: 192 records. Every calibration field
is below the instability threshold. The exact law is `value=D*f`, with
`f=lambda*(lambda+2E)/4`, so a weighted scalar fit through the origin identifies
`D` without fitting the prediction cases.

The fixed true value is `.937`, measurement sigma is `.005`, data seed is
`172101`, and the 256-realization noise seed is `172102`. Sigma is independent
of the hidden true parameter. These choices were recorded before generation.
Public and private calibration copies are byte-identical. The 192 records give
a stable number of degrees of freedom for the fixed reduced-chi-square cutoff;
the measured noise-validation result is reported separately.

The public starter contains the complete homogeneous predictor and an
unfinished fit. The completed shortcut has the exact weighted fit. It fits
calibration and recovers the parameter, then fails only the two distinguishing
prediction groups. An unfinished starter failure is not the physical control.

## Independent gold and grading

Two groups retain all six predeclared prediction points:

| Group | Field | Biases |
|---|---:|---|
| field10 | 10 | -5, -7.5, -10 |
| field12 | 12 | -6, -9, -12 |

Their gold consists of raw selected grid-256 SCGF values at `D=1`, scaled by the
true `D`. The independent solver maximizes the spatial functional using
`z=sin(phi)`, periodic edge gradients and the mean-density constraint. It uses
three fixed fresh seeds, followed by a fixed equality-constrained Newton polish.
It never calls the elliptic solver or starts from an elliptic profile. Both
grids, all seeds and selection evidence are retained outside the public image.
`tests/reference.py` contains the self-contained reproduction code; normal
verification reads frozen gold to stay within 60 seconds.

The exact-limit group uses `(E,lambda)=(0,1),(4,2),(2*pi,-2*pi)`. Each prediction
group is graded by `norm(prediction-truth)/norm(truth) < .04`. The groups have
nonzero norms. The parameter error must be below `.03`, and reduced calibration
chi-square is `sum(residual**2)/(192-1) < 1.5`.

The local oracle passes all seven tests. The completed shortcut passes five and
fails exactly `test_prediction[field10]` and `test_prediction[field12]`.
`../../results/validation.json` contains every noise-fit result and source
hashes. The corresponding timestamped directory preserves the reference
reproduction and complete pytest logs.

## Retained validation and reproduction

The first numerical screen remained unresolved because objective-based optimizer
stopping left some nonuniform profiles above the fixed stationarity tolerance.
Selection then fell back to an unstable flat stationary point. A separately
frozen polish of all 72 retained profiles resolved the residual without changing
the thresholds or materially changing SCGF values. The subsequent 37-case domain
study retained 222 raw and refined profiles, including three raw iteration-limit
failures excluded from selection. All domain gates passed. These studies and
reviews remain in `../../evidence/`; no failed result was replaced.

From the package root:

```bash
uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 \
  --with pytest==8.4.2 python scripts/validate_ring_current_fluctuations.py
```

Ordinary validation reads the fixed data. `--regenerate` creates the predeclared
calibration or verifies that existing bytes are identical; it refuses a change
to existing data. `--recompute-reference` also reproduces the six independent
references at grids 128/256. Every invocation has a new result directory.

The first bare-Python validation invocation lacked NumPy and stopped before
data generation or science. Its traceback and classification are retained.
Validation then passed with Python 3.13.15, NumPy 2.3.3 and SciPy 1.16.3.
No Docker controls or model trials have been run by the package author.
