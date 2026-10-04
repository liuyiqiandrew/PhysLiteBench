# Moving inductor

A coil has an externally moved core. Its reversible, linear flux linkage at
core coordinate x is L(x) I, with L(x) = 0.6 + 0.4 x henry. The coordinate x is
position divided by the known travel distance, hence dimensionless. The coil's
series resistance R is constant and unknown. Ignore hysteresis, saturation,
parasitic capacitance, and radiation. An ideal actuator prescribes x(t) regardless
of the electromagnetic force. A voltage source maintains constant terminal
voltage V during a run. At t=0 the current is I0. The actuator follows
x(t) = x0 + amplitude*sin(omega*t). The observable is current in amperes.

## Interface

Each experiment has `time` (s, 0 to 2), `initial_current` (A, -2 to 2),
`voltage` (V, -5 to 5), `x0`, `amplitude`, and `omega` (rad/s, 0 to 15).
Every specified trajectory stays within 0 <= x <= 1.
`resistance` is in ohms, between 0.2 and 8. Each record starts a fresh run.

Implement `Model` in `model.py`. `fit(records)` must return `self` and set
`resistance`. `predict(experiments)` must return a finite NumPy array of shape
`(len(experiments),)`. You may edit the entire implementation within this API.

`data/calibration.json` is a list of records. Each record has an `input`
experiment dictionary, measured `value`, and known Gaussian measurement standard
deviation `sigma`. Errors in different records are independent. Experimental
settings are exact. Use the fitted apparatus parameter for all predictions over
the stated input ranges.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
