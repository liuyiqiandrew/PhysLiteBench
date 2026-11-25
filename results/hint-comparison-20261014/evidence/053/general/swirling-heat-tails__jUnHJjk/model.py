import numpy as np


def response(e):
    """Return the rate with the mobility factor removed.

    Rescaling time by ``mobility * stiffness`` reduces the dynamics to an
    isotropic OU process with rotational drift.  The Legendre transform of
    its heat SCGF gives

        sqrt((1 + r^2) * (1 + r^2 z^2)) - 1 - r^2 z.

    The algebraically equivalent rationalized form below avoids losing
    precision when z is close to its typical value, z=1.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])
    root = np.sqrt((1.0 + r * r) * (1.0 + (r * z) ** 2))
    rate = (r * r * (z - 1.0) ** 2) / (root + 1.0 + r * r * z)
    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        # The reported rate is linear in the common mobility.  Since the
        # readings have known Gaussian standard deviations, minimizing the
        # weighted squared residuals has a closed-form solution.
        x = np.asarray([response(record['input']) for record in records], dtype=float)
        y = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        weights = 1.0 / (sigma * sigma)

        estimate = np.sum(weights * x * y) / np.sum(weights * x * x)
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
