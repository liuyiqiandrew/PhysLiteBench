# Fluorescence gate measurements

A fixed two-level emitter has states `|g>` and `|e>`. A classical monochromatic
field drives it continuously. In the rotating frame, with `hbar=1` and basis
`(|g>,|e>)`, its Hamiltonian is

```
H = [[0, rabi/2], [rabi/2, detuning]].
```

The rotating-wave and Markov approximations apply. The only dissipation is
spontaneous radiative decay from `|e>` to `|g>` at rate `decay`, into a vacuum
electromagnetic reservoir. There is no pure dephasing, nonradiative decay,
motion, or coupling to other emitters. The drive is maintained throughout each
measurement. Before every gate, the emitter has reached its stationary state
under these same controls.

A broadband photon counter collects the emitter's fluorescence. The incident
drive field is absent from the detected port. All emitted frequencies are
accepted, including the elastic fluorescence component. The counter records
each emitted photon independently with a common probability `efficiency`.
There are no dark counts, dead time, saturation, temporal smoothing, or spectral
filters. The gate opens for `duration`. Repeated gates use independently
prepared stationary ensembles. The requested reading is either the mean or the
variance of the actual number of recorded photons per gate. These are ensemble
count statistics, without finite-sample estimator corrections.

Time is in a fixed reference unit; `rabi`, `detuning`, and `decay` are in its
inverse. Recorded photon number and its variance are dimensionless. The only
unknown is `efficiency in [.5,.9]`.

Each experiment dictionary contains:

- `rabi` in `[.6,1.8]`;
- `detuning` in `[-.5,.5]`;
- `decay` in `[.8,1.2]`;
- `duration` in `[1,6]`;
- `observable`, either `"mean"` or `"variance"`.

## Interface

Implement `Model.fit(records)` and `Model.predict(experiments)` in `model.py`.
`fit` receives records from `data/calibration.json`, stores the fitted efficiency
in `self.efficiency`, and returns `self`. Each record contains `input`, measured
`value`, and independent Gaussian instrument standard deviation `sigma`.
`predict` receives a list of experiment dictionaries and returns a
one-dimensional NumPy array of the requested count statistics in the same order.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
