import numpy as np


def response(e):
    """Heat rate per unit mobility for a stationary preparation.

    Time in the dimensionless OU process is mobility * stiffness * t.
    Its heat SCGF is 1 - sqrt(1 + 4*r**2*lambda*(1-lambda)).
    Integrating over the stationary initial state and the free final state
    restricts lambda; the resulting Legendre transform has linear tails.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])
    a = r * r

    if z <= -1.0:
        rate = -2.0 * a * z
    elif a < 1.0 / 3.0 and z >= (3.0 - a) / (1.0 - 3.0 * a):
        rate = 2.0 * a * ((1.0 - a) * z - 2.0) / (1.0 + a)
    else:
        rate = np.sqrt((1.0 + a) * (1.0 + a * z * z)) - 1.0 - a * z

    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        scale = np.asarray([response(row['input']) for row in records], dtype=float)
        value = np.asarray([row['value'] for row in records], dtype=float)
        sigma = np.asarray([row['sigma'] for row in records], dtype=float)
        weighted_scale = scale / sigma
        denominator = np.dot(weighted_scale, weighted_scale)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError('Calibration must contain information about mobility.')

        estimate = np.dot(weighted_scale, value / sigma) / denominator
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
