import numpy as np
from scipy.optimize import brentq


def response(experiment):
    """Rod-current readout for one input, per unit rod speed.

    The single unknown apparatus speed multiplies this response, so the
    returned values are the shape of ``mean``/``variance``/``scgf`` with the
    speed factored out.
    """
    rho, mean = experiment['density'], experiment['length']
    statistic = experiment['statistic']

    if statistic == 'mean':
        # Left/right symmetry of the velocity marks forces a vanishing drift.
        return 0.
    if statistic == 'variance':
        return rho * (1. - rho * mean)

    field = abs(experiment['counting_field'])
    if field == 0.:
        return 0.
    effective_density = rho / (1. - rho * mean)
    s = brentq(
        lambda s: s + effective_density * mean * np.sinh(s) - field,
        0., field + 1., xtol=1e-14,
    )
    return effective_density * 2. * np.sinh(s / 2.) ** 2


def predict_at(experiments, speed):
    values = [response(e) for e in experiments]
    return speed * np.asarray(values, dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        responses = np.asarray([response(r['input']) for r in records], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)

        weights = 1. / sigma ** 2
        denominator = float(np.sum(weights * responses * responses))
        if denominator > 0.:
            self.speed = float(np.sum(weights * responses * values) / denominator)
        else:
            # No informative (variance/scgf) record: the speed is unobservable.
            self.speed = 1.0
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
