# Fluid coexistence, revision 1

The public homogeneous molar Helmholtz free energy is

    f(v,T) = -T ln(v-1) - a/v.

It gives the homogeneous pressure `P = -df/dv = T/(v-1) - a/v²`. The apparatus
fixes temperature, total amount, and total volume, permits macroscopic density
separation and material exchange, and neglects all interface or nucleation costs.
Its equilibrium free energy is therefore the lower convex envelope of the
homogeneous free energy. Inside the coexistence interval the mean molar volume
changes phase fractions, not the common equilibrium pressure. The two phases
have equal pressure and chemical potential. This is the
[Maxwell/common-tangent construction](https://arxiv.org/abs/1402.1205).

The completed shortcut differentiates the supplied homogeneous free energy
correctly and then uses that homogeneous state at every mean density. That is a
physical equilibrium-selection error, not a differentiation or root-finding
error. The lowest-temperature hidden case contains negative homogeneous
pressures, but the highest-temperature case has positive homogeneous pressures
throughout and still rejects the shortcut; clamping negative pressures cannot
repair the task.

## Calibration and controls

The one fitted parameter is attraction `a`; the true value is 3. The bounds are
2.6 to 3.4. The critical point is `T_c = 8a/27`, `v_c = 3`, and `P_c = a/27`.
All calibration temperatures are at least 1.4, above the largest possible
critical temperature (approximately 1.0074). Consequently every calibration
state is a single phase, both models coincide exactly, and the attraction is
identified by the linear coefficient of `1/v²` in pressure.

Calibration has two independent copies of 100 measurements, random seed 9212,
and Gaussian sigma equal to 0.003 times the largest absolute noiseless pressure.
The validator's `--generate` option explicitly writes both public and private
copies; normal validation reads the checked-in data.

The oracle obtains the spinodal volumes from `dP/dv = 0`. For each trial
coexistence pressure it brackets the outer liquid and vapor roots of the EOS.
It then solves the Maxwell equal-area condition using the analytical pressure
integral. For the extremely small critical window `1-T/T_c <= 1e-7`, the pressure
loop is below 1e-10 and the oracle uses the homogeneous pressure to avoid loss of
precision from nearly coincident roots. The reference is separate: it constructs
a lower convex hull of free energy on 30,000 logarithmically spaced volumes,
then refines the two common-tangent contacts. Its local phase volumes are not
artificially bounded by the allowed mean-volume range.

The hidden temperatures are 0.64, 0.72, and 0.80, with 36 mean volumes per group.
Every selected volume lies strictly between the reference coexistence contacts.
The independent reference confirms this fact. Public and private calibration
require reduced chi-square < 1.5, parameter recovery requires relative error <
3%, and each hidden group requires pressure RMSE divided by reference RMS
pressure < 0.04. This threshold is above the measured calibration sensitivity
and numerical uncertainty, and below the shortcut's error in every group.

## Validation

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 \
  --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 \
  scripts/validate_fluid_coexistence.py
```

The checked-in data fit to `a = 2.99960956425`, with reduced chi-square 0.87029.
The oracle's largest hidden NRMSE is 0.000520; shortcut NRMSEs are 1.37835,
0.51698, and 0.13928. Local public/private pytest gives 7 passed for the oracle;
the completed shortcut gives 4 passed and the three intended hidden failures.

All 256 noise realizations (seed 19212) pass calibration and parameter recovery.
Maximum relative parameter error is 0.4464%. Checking hidden predictions at the
fitted-parameter extrema gives maximum oracle NRMSE 0.01772 and minimum shortcut
NRMSE 0.13327. This is an extrema sensitivity check, not an exhaustive
hidden-prediction Monte Carlo claim.

The independent contact volumes and pressure agree with the oracle within
3.47e-13. Numerical quadrature confirms Maxwell area within 3.8e-16, endpoint
free-energy curvature is positive, and the common tangent lies below homogeneous
free energy over the checked domain. Stable single-phase branches retain the
original EOS exactly. Near the critical point at 0.9999 T_c, the coexistence
volumes approach 3 (2.94106 and 3.06110); at `(1-1e-10) T_c`, pressure differs
from critical pressure by less than 4.45e-11 without numerical failure.

Harbor confirms the oracle/shortcut control separation. GPT-5.6 Luna high
scored 3/3 unhinted and 3/3 with only `hint.md` appended to its instruction.
The target of 0/3 unhinted was not reached. See the
[reviewed evaluation results](../../DASHBOARD.md#candidate-expansion-matched-physics-hints).
