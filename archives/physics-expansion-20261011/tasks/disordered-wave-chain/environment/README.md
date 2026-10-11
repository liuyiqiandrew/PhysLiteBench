# Transmission through a disordered wave chain

A monochromatic scalar wave propagates on an infinite one-dimensional line.
Away from the scatterers, its complex amplitude obeys psi''+k^2*psi=0.
There are N identical point barriers at ordered positions x_j. At each
barrier psi is continuous and its derivative has the jump
psi'(x_j+)-psi'(x_j-)=2*b*psi(x_j). The common unknown positive barrier
strength b lies in [0.6,1.0] mm^-1. The wave number k is known.
The medium is identical on both sides and between all barriers.

A unit-amplitude wave is incident from the left, with no wave incident
from the right. The detector measures transmitted power divided by
incident power after stationary scattering is established. There is no
absorption, inelastic scattering, or phase noise during a measurement.
The incident wave has unlimited coherence length. All barriers remain
fixed throughout that measurement, including all repeated reflections.

For each repetition, the N-1 separations are independently redrawn as
L_j=(20*pi+u_j)/k, with each u_j uniform on [0,2*pi). They then remain
fixed. The apparatus reports the arithmetic ensemble mean of the power
transmission. It does not average complex amplitudes or logarithms of
power. For N=0 there are no barriers; for N=1 there are no separations.

An experiment is a dictionary with `count`=N, an integer in [0,12], and
`wave_number`=k in [0.8,1.6] mm^-1. `data/calibration.json` contains
independent noisy readings `{ "input": experiment, "value": mean_power,
"sigma": sigma }`. Each sigma is the known standard deviation of additive
Gaussian readout noise. Calibration uses N=1 and N=2 at several wave
numbers. Every experiment uses the same unknown b.

Implement `Model.fit(records)` to estimate b, store it as `model.strength`,
and return `self`. Implement `Model.predict(experiments)` to return a
finite NumPy array of shape `(len(experiments),)` in the same order.
Predictions must cover the full stated domain. The supplied response is a
starting approximation and may be changed. Run `pytest -q test_public.py`
for the visible interface and calibration check.
