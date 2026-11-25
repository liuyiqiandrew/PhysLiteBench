import numpy as np
from scipy.optimize import minimize_scalar


def _base_scgf(e):
    """Dimensionless current SCGF for equal length rods."""
    rho, length = e['density'], e['length']
    field = abs(e['counting_field'])
    # The exact equal-length result is the SCGF of the associated renewal
    # process, parameterized by its positive saddle point.
    c = rho / (1.0 - rho * length)
    # Solve s + c*length*sinh(s) = |lambda| by monotone bisection.
    lo, hi = 0.0, field
    for _ in range(60):
        mid = (lo + hi) * 0.5
        if mid + c * length * np.sinh(mid) < field:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) * 0.5
    return 2.0 * c * np.sinh(s / 2.0) ** 2


def response(e):
    rho, length = e['density'], e['length']
    stat = e['statistic']
    if stat == 'mean':
        return 0.0
    if stat == 'variance':
        # Mixture simulations show the variance is insensitive to contrast;
        # dependence on occupancy follows the free-gap fraction.
        return rho * (1.0 - rho * length)
    if stat == 'scgf':
        return _base_scgf(e)
    raise ValueError(f"unknown statistic: {stat}")


def predict_at(experiments, speed):
    return speed * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        # Weighted least squares for a single multiplicative speed parameter.
        x = np.asarray([response(r['input']) for r in records], dtype=float)
        y = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        w = 1.0 / sigma**2
        self.speed = float(np.sum(w * x * y) / np.sum(w * x * x))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.speed)
