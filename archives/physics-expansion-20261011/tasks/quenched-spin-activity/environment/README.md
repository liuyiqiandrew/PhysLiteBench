# Switching activity of an interacting magnet

An even number `N` of classical Ising spins `sigma_i = +/-1` interact equally
with all other spins and remain attached to one thermal bath. Exactly half
the spins have a fixed local field `field + disorder`; the other half have
fixed local field `field - disorder`. Each spin keeps its field assignment
for the entire experiment. There is no averaging over new assignments during
a trajectory. Units have `k_B*T=1`.

For total spin `S=sum_i sigma_i`, the energy is

```
E = -coupling*S**2/(2*N) - sum_i local_field_i*sigma_i.
```

The exact continuous-time dynamics consists of single-spin flips. Spin `i`
flips at rate

```
attempt_rate/2 * (1 - sigma_i*tanh(coupling*(S-sigma_i)/N + local_field_i)).
```

There are no simultaneous flips. The common unknown `attempt_rate` is in
`[0.8, 1.4]` inverse time units. The energy and rates above define the ideal
apparatus at every finite `N`. Each run starts in its finite-`N` canonical
equilibrium distribution and then evolves uninterrupted at fixed controls.

Two passive counters record the actual number of flips, `K_plus(t)` and
`K_minus(t)`, in the two fixed field populations during time `t`. Both
directions of a flip count as one. The readout is

```
psi = lim_(N -> infinity through even N) lim_(t -> infinity)
      log(E[exp(-bias_plus*K_plus(t) - bias_minus*K_minus(t))]) / (N*t).
```

The expectation is over equilibrium initial states and the subsequent jump
trajectories. Take the long-time limit at each finite size before taking the
size limit. The logarithm is natural. The two biases weight the recorded
counts only; they do not change the applied fields, temperature or jump law.
The result is per spin per unit time, in inverse time units, and can be
positive or negative.

## Interface

Implement `Model` in `model.py`:

- `fit(records)` sets scalar attribute `attempt_rate` and returns `self`.
- `predict(experiments)` returns a finite NumPy array of shape
  `(len(experiments),)` in input order.

Each experiment has all five fields:

| Field | Allowed values |
|---|---|
| `coupling` | Ferromagnetic interaction strength, `[0, 1.6]` |
| `field` | Uniform applied field, `[-0.5, 0.5]` |
| `disorder` | Magnitude of the two fixed local-field offsets, `[0, 1.2]` |
| `bias_plus`, `bias_minus` | Dimensionless counting biases, `[-1, 1.5]` |

`data/calibration.json` is a list of records containing `input` (an
experiment), `value` (measured `psi`) and `sigma` (known independent Gaussian
measurement standard deviation). Calibration uses `coupling=0`. Prediction
experiments can use the full stated domain. The same attempt rate applies
to both field populations and all preparations.

Run `python -m pytest -q test_public.py` to check interface and calibration.
You may replace any implementation code while preserving the API.
