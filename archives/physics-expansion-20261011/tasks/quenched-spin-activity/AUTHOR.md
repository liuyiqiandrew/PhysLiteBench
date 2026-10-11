# Quenched spin activity, revision 1

The task measures long-time flip fluctuations in a fully connected Ising
magnet with two fixed field populations. The finite-size energy, exact
single-spin heat-bath rates, initial canonical ensemble, passive counters,
and order of limits are public. Thermal energy is one. The sole fitted
parameter is the common attempt rate, in inverse time units.

## Physical models

The completed approximation finds the equilibrium mean field and holds it
fixed while calculating two independent two-state counting processes. It
selects the global minimum of the equilibrium free energy and includes all
degenerate equilibrium phases. This is a well-defined auxiliary stochastic
process. It is exact for the uncoupled calibration preparations. Its flaw
is using the ordinary equilibrium collective field for rare trajectories
selected by activity bias. The field assignments themselves remain quenched
in both the approximation and the correct model.

For population magnetizations z_plus and z_minus, put M=(z_plus+z_minus)/2
and f_plus/minus=J*M+h+/-H. Detailed balance symmetrizes the finite-size
tilted generator. Its leading eigenvalue per spin tends to the global maximum

```
psi/kappa = max_(z_plus,z_minus in [-1,1])
  1/4 * sum_g [exp(-s_g)*sqrt(1-z_g**2)/cosh(f_g)
              - 1 + z_g*tanh(f_g)].
```

Off-diagonal flips receive the counting factor; physical escape rates do not.
The local diagonal term follows from the up and down rates. The symmetric
off-diagonal term is twice their geometric mean. Upper bounds from maximum
row sums and trial states localized around a magnetization give the same
thermodynamic limit. The oracle uses z=sin(theta), finds all grid maxima,
then refines them. Refining 49 to 97 grid points in each direction verifies
the selected maximum across hidden and domain checks.

The independent reference builds the exact finite-size count process for
the numbers of up spins in each population, including the finite-size
self-interaction exclusion in the stated flip law. It uses detailed balance
to form a symmetric sparse matrix and finds its largest eigenvalue. It
extrapolates sizes 80, 160, and 320 quadratically in inverse size. Refinement
uses 160, 320, and 640. This reference does not use the variational potential.
Canonical initial conditions give overlap with the positive leading state
at each finite size, including coexistence. The specified order of limits
therefore removes an ambiguity about metastable initial phases.

## Calibration and validation

At J=0, each field population is an independent two-state process. Its
closed-form tilted eigenvalue is linear in kappa, giving a unique scalar
weighted least-squares fit. Calibration contains 144 readings at 18 settings,
with Gaussian sigma 0.0003 and seed 611201. The true attempt rate is 1.13.
Both completed controls recover 1.1301069217, with reduced chi-square 1.00369.

The validator compares calibration to direct two-state matrix eigenvalues,
checks 256 independent noise draws (seed 611203), recomputes the frozen hidden
reference, refines finite size and oracle optimization, and checks 36 random
and boundary cases. Zero bias and reflection with population exchange are
additional invariants. Across the noise draws, the worst oracle group error
is 0.0844%; the smallest diagnostic shortcut error is 12.46%. Hidden reference
refinement changes a group by at most 0.000123%. Domain spot checks differ
from the oracle by at most 3.57e-5 in absolute rate units.

Calibration requires reduced chi-square below 1.5; parameter error must be
below 3%. Each hidden group uses RMSE divided by its true RMS, below 4%.
Groups cover uncoupled anchors, quiet populations, opposite activity biases,
applied field response, and moderate coupling. These are within the public
domain. Reference values are frozen to keep grading fast.

Run `scripts/validate_quenched_spin_activity.py` using the pinned NumPy,
SciPy, and pytest environment. Its `--generate` option intentionally replaces
both calibration copies and hidden truth. The report is
`results/expansion-quenched-spin-activity-r1-validation.json`.

## Evaluation

Evaluate exactly three frozen, unhinted gpt-5.6-luna High trials with standard
600/60-second agent/verifier limits, after independent review and Docker
controls. Preserve every outcome. Inspect final source and trajectories.
Using a fixed equilibrium field, discarding the distinction between field
populations, or selecting only an equilibrium saddle can be physical errors.
A missed numerical maximum, finite-size extrapolation error, wrong factor,
or unfinished implementation requires a separate diagnosis and cannot be
counted automatically as a physical failure.

Primary context is Guioth and Jack,
[Dynamical phase transitions in the random-field Ising model](https://arxiv.org/abs/2108.00063).
The task uses the explicitly stated energy convention and separately biases
the two populations. The exact finite-size generator verifies its factors.
