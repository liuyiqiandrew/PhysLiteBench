import numpy as np


def _rod_field(rho, length, right_fraction, counting_field):
    """Return the hard-rod SCGF divided by the common speed.

    Removing the rod interiors maps the equilibrium configuration to a
    Poisson process of density ``c = rho / (1 - length*rho)``.  If ``s`` is
    the saddle point associated with the counting field, its equation is

        field = s + c*length*(p*(exp(s)-1) - (1-p)*(exp(-s)-1)).

    The current SCGF is then

        c * (p*(exp(s)-1) + (1-p)*(exp(-s)-1)).

    The equation is strictly increasing in s, so a safeguarded Newton solve
    is inexpensive and well behaved over the apparatus domain.
    """
    h = 1.0 - length * rho
    c = rho / h
    b = c * length

    if counting_field == 0.0:
        return 0.0

    def equation(s):
        return (s + b * (right_fraction * np.expm1(s)
                         - (1.0 - right_fraction) * np.expm1(-s)))

    def derivative(s):
        return (1.0 + b * (right_fraction * np.exp(s)
                            + (1.0 - right_fraction) * np.exp(-s)))

    # Bracket the unique root.  The zero-field point is always a convenient
    # endpoint, and the requested fields are small enough that one expansion
    # is normally sufficient.
    if counting_field > 0.0:
        lo, hi = 0.0, max(1.0, counting_field)
        while equation(hi) < counting_field:
            hi *= 2.0
    else:
        lo, hi = min(-1.0, counting_field), 0.0
        while equation(lo) > counting_field:
            lo *= 2.0

    s = min(max(counting_field, lo), hi)
    for _ in range(50):
        value = equation(s) - counting_field
        if abs(value) < 1e-14:
            break
        if value > 0.0:
            hi = s
        else:
            lo = s
        trial = s - value / derivative(s)
        s = trial if lo < trial < hi else 0.5 * (lo + hi)

    return c * (right_fraction * np.expm1(s)
                + (1.0 - right_fraction) * np.expm1(-s))


def response(e):
    rho = float(e['density'])
    a = float(e['length'])
    p = float(e['right_fraction'])
    h, d = 1.0 - a * rho, 2.0 * p - 1.0
    if e['statistic'] == 'mean':
        return rho*d
    if e['statistic'] == 'variance':
        return rho*h*(1-a*rho*d*d)
    if e['statistic'] == 'scgf':
        return _rod_field(rho, a, p, float(e['counting_field']))
    raise ValueError(f"unknown statistic: {e['statistic']}")


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        basis = np.asarray([response(record['input']) for record in records],
                           dtype=float)
        values = np.asarray([record['value'] for record in records],
                            dtype=float)
        sigma = np.asarray([record['sigma'] for record in records],
                           dtype=float)
        weights = 1.0 / (sigma * sigma)
        estimate = np.sum(weights * basis * values) / np.sum(weights * basis * basis)
        self.speed = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.speed)
