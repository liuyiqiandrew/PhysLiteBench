# Radiation transitions of moving detectors

Identical pointlike two-level detectors interact weakly with one real free
massless scalar field in unbounded 3+1-dimensional Minkowski spacetime. The
field is in the Minkowski vacuum. Use c=hbar=1 and a fixed time unit. The
canonical field normalization is

```
<Phi(t,r) Phi(t',r')> = -1/(4*pi²*((t-t'-i0)² - |r-r'|²)).
```

Each detector has proper-time Hamiltonian `gap*sigma_z/2`, with ground state
`g`, excited state `e`, and positive gap. A pair follows these prescribed
worldlines, written in their shared proper time tau:

```
t_1(tau) = t_2(tau) = sinh(acceleration*tau)/acceleration,
x_1(tau) = x_2(tau) = (cosh(acceleration*tau)-1)/acceleration,
y_1(tau) = 0,   y_2(tau) = separation,
z_1(tau) = z_2(tau) = 0.
```

At zero acceleration use their continuous limits, `t=tau`, `x=0`. The
separation is transverse to the acceleration and fixed. External agents
maintain these trajectories without recoil. They add no field radiation or
noise. There are no boundaries or direct detector-detector interactions.
The interaction Hamiltonian, generating evolution in tau, is

```
H_int(tau) = mu*(sigma_x^(1)*Phi(x_1(tau))
                + sigma_x^(2)*Phi(x_2(tau))).
```

Both couplings have the same real sign. In a single-detector preparation,
only detector 1 and its interaction term are present. The common unknown
dimensionless `coupling_squared=mu²` lies in `[0.00005, 0.0002]`.

The apparatus reports stationary transition rates per unit proper time at
leading order mu². Take the usual weak-coupling, long-observation Golden Rule
limit: discard switching transients and terms of higher order in mu, and
measure before population depletion. The field initially has no correlation
with the prepared detector state. Gaps are the stated free detector gaps;
higher-order frequency shifts are outside this readout.

The `preparation` selects one resolved downward transition:

| Preparation | Initial state | Counted final state |
|---|---|---|
| `single` | `e` | `g` |
| `symmetric` | `(eg+ge)/sqrt(2)` | `gg` |
| `antisymmetric` | `(eg-ge)/sqrt(2)` | `gg` |

For a pair, count only arrival in `gg`; transitions to `ee` are a different
outcome. The result is a nonnegative rate in inverse time units.

## Interface

Implement `Model` in `model.py`:

- `fit(records)` sets scalar attribute `coupling_squared` and returns `self`.
- `predict(experiments)` returns a finite NumPy array of shape
  `(len(experiments),)` in input order.

Every experiment contains:

| Field | Allowed values |
|---|---|
| `preparation` | `single`, `symmetric`, or `antisymmetric` |
| `gap` | `[0.5, 2]` in inverse time units |
| `acceleration` | `[0, 10]` in inverse time units |
| `separation` | `[0.1, 3]` in length units; ignored for `single` |

`data/calibration.json` contains records with `input`, measured `value`, and
known independent Gaussian standard deviation `sigma`. Calibration includes
single detectors at varied accelerations and pairs at zero acceleration.
Prediction experiments can use the full stated domain. All measurements have
the same coupling.

Run `python -m pytest -q test_public.py` to check interface and calibration.
You may replace any implementation code while preserving the API.
