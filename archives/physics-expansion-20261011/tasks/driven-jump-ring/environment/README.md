# Driven molecular ring

A classical molecule has three conformations labelled `0, 1, 2` in clockwise order. Its state follows a continuous-time Markov jump process. A controller repeats a specified list of stages indefinitely. Each stage holds the energies `E_i`, barrier energies `B_i`, drive affinity `F`, and duration fixed. Energies and `F` are measured in units of the constant bath thermal energy `k_B*T=1`.

Bond `i` joins state `i` to state `j=(i+1)%3`. The exact transition rates during a stage are

```
i -> j: attempt_rate * exp(E_i - B_i + F/6)
j -> i: attempt_rate * exp(E_j - B_i - F/6).
```

The common unknown `attempt_rate`, in inverse time units, lies in `[0.8, 1.3]`. Every bond remains open. Stage switches change the rates instantaneously and do not cause a state transition. After many repetitions, the molecule reaches its unique periodic stationary preparation.

A passive counter adds `1/3` for every clockwise hop (`0->1`, `1->2`, `2->0`) and subtracts `1/3` for each reverse hop. It records fractional turns, not just completed revolutions. Starting at the beginning of the first stage, let `Q_N` be the total counter change during `N` consecutive complete periods. The dynamics runs continuously through every stage and period boundary, without resetting the molecule or its bath. There is no measurement feedback.

For a supplied real field `field=chi`, predict

```
psi(chi) = lim_(N -> infinity) log( E[exp(chi*Q_N)] ) / N.
```

The expectation is over the jump trajectories. The logarithm is natural. The result is dimensionless **per period**, without dividing by period duration. Changing the field changes only the statistic evaluated from the counter; it does not change transition rates. The constant-energy special case is permitted: a period then remains the stated observation duration.

## Interface

Implement `Model` in `model.py`:

- `fit(records)` sets scalar attribute `attempt_rate` and returns `self`.
- `predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order.

An experiment has `field` in `[-2, 2]` and `stages`, a list of one to four dictionaries in chronological order. Each stage contains:

| Field | Allowed values |
|---|---|
| `energies` | Three energies in `[0, 1.8]`, in state order |
| `barriers` | Three energies in `[2, 4]`, in bond order |
| `affinity` | `F` in `[-2, 2]`; the clockwise loop affinity |
| `duration` | Positive time in `[0.1, 1.5]` |

`data/calibration.json` contains records with `input` (an experiment), `value` (measured `psi`) and `sigma` (known independent Gaussian measurement standard deviation). Calibration uses stages with spatially uniform energies and barriers. Prediction experiments may use the full stated domain.

Run `python -m pytest -q test_public.py` to check interface and calibration. You may replace any implementation code while preserving the API.
