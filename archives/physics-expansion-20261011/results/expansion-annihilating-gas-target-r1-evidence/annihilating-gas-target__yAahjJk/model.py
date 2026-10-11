"""Finite-time survival model for the tagged particle."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ive


_SPEED_BOUNDS = (0.8, 1.3)
_TAYLOR_ORDER = 96
_TALBOT_N = 20


def _one_sided_series(p, order=_TAYLOR_ORDER):
    """Derivatives of the one-sided survival probability at zero."""
    a = np.zeros(order + 1, dtype=float)
    a[1] = (p - 1.0) / 2.0
    # Substitute T(s)=sum(a[n]/s**n) into the quartic transform equation.
    for k in range(1, order):
        square = np.convolve(a, a)[:k + 1]
        fourth = np.convolve(square, square)[:k + 1]
        nonlinear = p * fourth[k] - (2.0 * p + 1.0) * square[k]
        a[k + 1] = -a[k] + 0.5 * nonlinear
    return a


def _physical_root(s, p):
    """The transform root analytic in the right half-plane."""
    s = np.asarray(s, dtype=complex)
    if p == 0.0:
        return -(s + 1.0) + np.sqrt(s * (s + 2.0))
    out = np.empty(s.shape, dtype=complex)
    for index, value in np.ndenumerate(s):
        roots = np.roots((p, 0.0, -(2.0 * p + 1.0),
                          -2.0 * (value + 1.0), p - 1.0))
        out[index] = roots[np.argmin(np.abs(roots))]
    return out.item() if out.ndim == 0 else out


def _one_sided_talbot(x, p, n=_TALBOT_N):
    """Invert (T(s)+1)/s at dimensionless time x."""
    r = 2.0 * n / 5.0
    k = np.arange(n, dtype=float)
    theta = np.pi * k / n
    cot = np.empty(n, dtype=float)
    csc2 = np.empty(n, dtype=float)
    cot[0] = csc2[0] = 0.0
    cot[1:] = 1.0 / np.tan(theta[1:])
    csc2[1:] = 1.0 / np.sin(theta[1:]) ** 2
    contour = np.empty(n, dtype=complex)
    derivative = np.empty(n, dtype=complex)
    contour[0] = r / x
    derivative[0] = 1j * r / x
    contour[1:] = (r / x) * (theta[1:] * cot[1:] + 1j * theta[1:])
    derivative[1:] = (r / x) * (
        cot[1:] - theta[1:] * csc2[1:] + 1j
    )
    transform = (_physical_root(contour, p) + 1.0) / contour
    terms = np.exp(contour * x) * transform * derivative
    return float((0.5 * terms[0].imag + np.sum(terms[1:].imag)) / n)


def _one_sided(x, p):
    if x <= 0.0:
        return 1.0
    if p == 0.0:
        return float(ive(0, x) + ive(1, x))
    if x <= 2.0:
        coefficients = _one_sided_series(p)
        value, term = 1.0, 1.0
        for n in range(1, len(coefficients)):
            term *= x / n
            value += coefficients[n] * term
        return float(np.clip(value, 0.0, 1.0))
    return float(np.clip(_one_sided_talbot(x, p), 0.0, 1.0))


def response(e, speed):
    x = float(e['density']) * float(speed) * float(e['time'])
    q = _one_sided(x, float(e['stationary_fraction']))
    return float(q * q)


def predict_at(experiments, speed):
    return np.asarray([response(e, speed) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError('fit requires at least one calibration record')

        def objective(speed):
            residuals = [
                (response(record['input'], speed) - float(record['value']))
                / float(record['sigma'])
                for record in records
            ]
            return float(np.dot(residuals, residuals))

        result = minimize_scalar(
            objective, bounds=_SPEED_BOUNDS, method='bounded',
            options={'xatol': 1e-10},
        )
        candidates = [(float(result.fun), float(result.x))]
        candidates.extend((objective(v), v) for v in _SPEED_BOUNDS)
        self.speed = min(candidates, key=lambda item: item[0])[1]
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.speed)
