"""Response functions for the hard-rod current experiment.

The point-particle coordinate has density ``c = rho / (1-rho*lbar)``.
For a net block whose mean rod length is ``a``, its SCGF is obtained from

    h = s + c a sinh(s),       psi = c (cosh(s) - 1).

For a length mixture the block need not have the bulk composition.  Its
composition changes its mean length but costs a Bernoulli relative entropy per
rod.  We therefore optimize that cost along with the point-particle current,
rather than substituting the bulk mean length in the formula above.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def _binary_cost(p):
    """D(Bernoulli(p) || Bernoulli(1/2)), including the endpoints."""
    if p <= 0.0 or p >= 1.0:
        return float(np.log(2.0))
    return p * np.log(2.0 * p) + (1.0 - p) * np.log(2.0 * (1.0 - p))


def _block_scgf(free_density, block_length, field):
    """Positive-current branch for a block with a specified mean length."""
    if field <= 0.0:
        # The constrained positive-current Legendre transform is zero here:
        # the optimum has zero extensive current.
        return 0.0
    s = brentq(
        lambda x: x + free_density * block_length * np.sinh(x) - field,
        0.0,
        field,
        xtol=1e-13,
        rtol=1e-14,
    )
    return free_density * (np.cosh(s) - 1.0)


def _mixture_scgf(rho, mean_length, contrast, field):
    """Annealed SCGF, including the large-deviation cost of block makeup."""
    h = abs(float(field))
    if h == 0.0:
        return 0.0

    # This density is set by the *whole* thermodynamic preparation.  The
    # composition of the O(t) crossed block must not be used to change it.
    c = rho / (1.0 - rho * mean_length)
    if contrast == 0.0:
        return _block_scgf(c, mean_length, h)

    # p is the long-rod fraction.  On [0, 1/2], p and 1-p have the same
    # entropy cost, while p has the shorter (and hence better) block.
    def value(p):
        block_length = mean_length * (1.0 + contrast * (2.0 * p - 1.0))
        effective_field = h - _binary_cost(p)
        return _block_scgf(c, block_length, effective_field)

    # There can be a switch between a bulk-composition and a short-rich
    # saddle.  Locate every grid-local maximum, then refine it; this avoids
    # assuming a single saddle across the whole input range.
    grid = np.linspace(0.0, 0.5, 41)
    vals = np.asarray([value(p) for p in grid])
    best = float(vals.max())
    for i in range(1, len(grid) - 1):
        if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
            result = minimize_scalar(
                lambda p: -value(p),
                bounds=(grid[i - 1], grid[i + 1]),
                method='bounded',
                options={'xatol': 1e-12},
            )
            best = max(best, -float(result.fun))
    return best


def response(e):
    rho = float(e['density'])
    mean_length = float(e['length'])
    statistic = e['statistic']
    if statistic == 'mean':
        return 0.0
    if statistic == 'variance':
        # Composition fluctuations first affect the SCGF beyond quadratic
        # order, so the zero-field variance uses the bulk mean length.
        return rho * (1.0 - rho * mean_length)
    return _mixture_scgf(rho, mean_length, float(e['contrast']), e['counting_field'])


def predict_at(experiments, speed):
    return float(speed) * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Weighted least-squares estimate of the single shared speed."""
        base = np.asarray([response(r['input']) for r in records], dtype=float)
        observed = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        weights = 1.0 / np.square(sigma)
        denominator = float(np.dot(weights, np.square(base)))
        if denominator == 0.0:
            # Defensive fallback for a calibration set carrying no rate data.
            estimate = 1.05
        else:
            estimate = float(np.dot(weights * base, observed) / denominator)
        self.speed = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
