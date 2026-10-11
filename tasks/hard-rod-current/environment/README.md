# Current through a mixture of ballistic rods

Equal-mass rods move on a one-dimensional line. Each rod has one of two
lengths: `length*(1-contrast)` or `length*(1+contrast)`, in equal proportions.
Here `length` is the mean rod length. There are no forces between collisions.
Rods cannot overlap or pass one another. Elastic collisions exchange their
velocities. A rod's length remains attached to that physical rod; the ordered
length sequence never changes during a trajectory.

Every initial velocity is independently +v or -v with probability one half,
independently of the rod lengths and positions. The common unknown speed v
lies in [0.8, 1.3]. Length and time use fixed apparatus units.

Prepare an even number N of rods on a periodic ring of circumference L,
with exactly N/2 of each length. Choose their cyclic order uniformly. For
that order, sample centers uniformly over all nonoverlapping configurations
and sample the independent velocity marks. Take N,L -> infinity at fixed
physical center density `density`=N/L before the time limit below. A fixed
spatial origin is independent of the configuration. Each realization redraws
the length order, positions, and velocities. None is resampled during a run.

Let J(t) be center crossings of the origin from left to right minus crossings
from right to left during [0,t]. The apparatus reports one of:

- `mean`: lim E[J(t)]/t.
- `variance`: lim Var[J(t)]/t, with the mean subtracted.
- `scgf`: lim log(E[exp(counting_field*J(t))])/t.

All limits are t -> infinity. The expectation averages over the freshly
drawn length order, positions, and velocity marks before taking the logarithm
and time limit. It is not conditioned on one particular length sequence.
The field is dimensionless and all readouts have units of inverse time.

An input dictionary contains `density` in [0.1, 0.6], `length` in [0.4, 1.2],
`contrast` in [0, 0.75], `counting_field` in [-4, 4], and `statistic` equal
to `mean`, `variance`, or `scgf`. The field is ignored for the first two
statistics. All other apparatus parameters are known.

`data/calibration.json` is a list of independent Gaussian observations, each
with an `input` dictionary, measured `value`, and known standard deviation
`sigma`. Calibration contains full generating functions for equal-length
rods and variance growth for length mixtures. The speed is shared by all
calibration and prediction preparations.

Implement `Model` in `model.py`. `fit(records)` must return `self` and store
the fitted float `self.speed`. `predict(experiments)` accepts a list of input
dictionaries and returns a finite NumPy array of shape `(n,)` in the same
order. You may change the supplied response. Prediction is tested across
the stated domain. Run `python -m pytest -q test_public.py` for public checks.
