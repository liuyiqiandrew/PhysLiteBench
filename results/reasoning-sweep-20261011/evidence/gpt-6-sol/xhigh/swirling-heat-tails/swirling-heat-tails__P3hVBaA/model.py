import numpy as np


def response(e):
    """Heat rate per unit mobility for one preparation.

    Time in trap units is mobility * stiffness * t. In those units the
    stationary mean heat rate is 2 * drive_ratio**2. The central saddle
    gives the square-root expression below. Free, stationary endpoints
    limit its domain: final energy controls z <= -1, and for weak drive
    initial energy controls sufficiently large positive z.
    """
    a = e['drive_ratio'] ** 2
    z = e['relative_heat']

    if z <= -1:
        rate = -2 * a * z
    elif a < 1 / 3 and z >= (3 - a) / (1 - 3 * a):
        rate = 2 * a * ((1 - a) * z - 2) / (1 + a)
    else:
        root = np.sqrt((1 + a) * (1 + a * z * z))
        rate = a * (z - 1) ** 2 / (root + 1 + a * z)

    return e['stiffness'] * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        if not records:
            raise ValueError('At least one calibration record is required.')

        rates = np.asarray([response(record['input']) for record in records])
        values = np.asarray([record['value'] for record in records])
        sigmas = np.asarray([record['sigma'] for record in records])
        weights = 1 / sigmas**2
        mobility = np.sum(weights * rates * values) / np.sum(weights * rates**2)
        self.mobility = float(np.clip(mobility, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
