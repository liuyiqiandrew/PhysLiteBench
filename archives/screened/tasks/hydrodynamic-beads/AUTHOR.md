# Hydrodynamically coupled beads, revision 1

Two separate harmonic traps give the equilibrium energy
`U = (k1*x1² + k2*x2²)/2`. At one bath temperature the equilibrium position
covariance is `C = k_B T K^-1`, with `K = diag(k1,k2)`, regardless of the constant
mobility matrix. Equilibrium positions therefore factorize even though the
fluid transfers dynamical disturbances between beads. This is the calibration
loophole.

For longitudinal motion in the stated leading far-field approximation,

    M_11 = M_22 = 1/(6 pi eta a),
    M_12 = M_21 = 1/(4 pi eta r).

The cross mobility is the longitudinal component of the point-force Stokes flow.
The public apparatus states this approximation and fixes the nominal separation
used throughout; finite-size hydrodynamic corrections, position-dependent
mobility, inertia, walls, and direct conservative interactions are excluded.
A primary-source discussion of hydrodynamically coupled trapped beads is
[Bérut, Petrosyan, and Ciliberto](https://arxiv.org/abs/1408.5319). This benchmark
uses the equal-temperature equilibrium bath and a deterministic center shift.

## Calibration and implementation

The measured equilibrium variance is `k_B T / k_i`, which identifies the single
base trap stiffness. Calibration contains two independent copies of 100
experiments, with seed 9211 and Gaussian sigma equal to 0.006 times the largest
noiseless calibration variance. Public and private copies are generated together
only by the validator's `--generate` option.

After the trap centers move by the vector `delta`, force balance gives

    d< x >/dt = M K (delta - < x >),
    < x(t) > = delta - exp(-M K t) delta.

The initial means vanish relative to the original centers. A shift of one trap
moves both beads initially; the unshifted bead subsequently returns to its
original center. The completed shortcut uses diagonal mobility and predicts
zero mean at the unshifted bead at every time, while fitting all equilibrium
variances exactly. The oracle uses a 2-by-2 matrix exponential. The independent
reference integrates the two force-balance equations in SI units using DOP853.

Calibration requires reduced chi-square < 1.5 and relative parameter error <
3%. Each hidden transient group requires RMSE divided by the reference RMS
signal < 0.04. Groups measure the bead whose own center has not moved, with
both signs of forcing, different separations, and unequal trap stiffnesses.
This normalization resolves the cross response, instead of allowing the larger
directly forced response to hide it. Four percent is above calibration and
numerical uncertainty and below the shortcut's complete absence of response.

## Validation

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 \
  --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 \
  scripts/validate_hydrodynamic_beads.py
```

The initial 256-noise validation used seed 19211: calibration passed for all
256, with maximum parameter error 0.223%. The checked-in calibration fits to
stiffness = 0.3001576091 pN/micrometer, chi-square 0.93087, against true stiffness
0.3. Oracle hidden NRMSE is at most 0.000368; shortcut NRMSE is exactly 1 in each
group. At fitted-parameter extrema, oracle NRMSE stays below 0.001557 and shortcut
NRMSE stays 1. This is a parameter-extrema sensitivity check, not an exhaustive
hidden-prediction Monte Carlo claim.

Independent SI integration agrees with the oracle to 9.32e-13 micrometers.
The covariance balance `M K C + C K M = 2 k_B T M` holds to 1.12e-16 in the
chosen units. The mobility is positive definite, the equal-trap normal-mode
solution agrees to 7.5e-16, initial means are zero, and the infinite-separation
limit agrees with independent dynamics to 5.83e-14. The local public/private
pytest harness gives 7 passed for the oracle, and 4 passed/3 hidden failures
for the completed shortcut.

Harbor confirms the oracle/shortcut control separation. GPT-5.6 Luna high
scored 3/3 unhinted and 3/3 with only `hint.md` appended to its instruction.
The target of 0/3 unhinted was not reached. See the
[reviewed evaluation results](../../DASHBOARD.md#candidate-expansion-matched-physics-hints).
