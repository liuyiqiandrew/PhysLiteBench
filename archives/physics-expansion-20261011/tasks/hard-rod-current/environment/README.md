# Ballistic rod current

Identical rods of length `length` and equal mass move on a one-dimensional
line. Their centers cannot approach closer than `length`. There are no forces
between collisions. Elastic collisions exchange the two velocities. Each
initial velocity is independently +v with probability `right_fraction` and
-v otherwise. The common speed v is unknown and lies in [0.8, 1.3]. Length
and time use the apparatus's fixed units.

The initial positional ensemble is the homogeneous equilibrium hard-rod
ensemble at center density `density`, independently of the velocity marks.
One definition is a periodic ring of circumference L with N centers sampled
uniformly from all allowed nonoverlapping configurations, followed by
N,L -> infinity at fixed N/L. Take that thermodynamic limit before the
long-time limit below. Both positions and velocity marks are redrawn for each
realization; neither is resampled during a trajectory. A fixed spatial origin
is independent of the initial configuration. The preparations need not have
zero mean current.

Let J(t) count center crossings of the origin from left to right minus those
from right to left during [0,t]. The apparatus reports one of:

- `mean`: lim E[J(t)]/t.
- `variance`: lim Var[J(t)]/t, with the mean subtracted in the variance.
- `scgf`: lim log(E[exp(counting_field * J(t))])/t.

All limits are t -> infinity. A negative SCGF is allowed. The counting field
is dimensionless; every readout has units of inverse time. There is no finite
observation-time correction in the reported quantity.

An input dictionary has `density` in [0.1, 0.6], `length` in [0, 1.2],
`right_fraction` in [0.25, 0.75], `counting_field` in [-4, 4], and `statistic`
equal to `mean`, `variance`, or `scgf`. The field is ignored for the first
two statistics. Zero length means point particles, with the same velocity
exchange collision rule. All other apparatus parameters are known.

`data/calibration.json` is a list of independent Gaussian observations.
Each record has an `input` dictionary, measured `value`, and known standard
deviation `sigma`. The calibration covers means and variances for rods,
and full generating functions for point particles. The same speed is used
in every record and in subsequent predictions.

Implement `Model` in `model.py`. `fit(records)` must return `self` and store
the fitted speed as the float `self.speed`. `predict(experiments)` accepts a
list of input dictionaries and returns a finite NumPy array of shape `(n,)`
in the same order. The supplied response is a starting model that you may
change. Prediction is tested across the stated domain. Run
`python -m pytest -q test_public.py` to check calibration and the interface.
