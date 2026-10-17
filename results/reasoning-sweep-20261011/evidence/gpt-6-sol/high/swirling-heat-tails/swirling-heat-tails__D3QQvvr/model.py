import numpy as np


def response(e):
    """Heat rate per unit mobility, including the free endpoint costs.

    In units of mobility * stiffness, the mean heat is 2*r**2. The
    stationary moment generating function has scaled logarithm
    1 - sqrt(1 - 4*r**2*lambda*(1 + lambda)). Its usable interval ends at
    lambda=-1 because the final position is unconditioned. For small r,
    the unconditioned initial position also limits the positive endpoint.
    """
    r = e['drive_ratio']
    z = e['relative_heat']
    s = r * r

    # Legendre transform on the interior. The rationalized expression
    # keeps the rate accurate near its zero at z=1.
    root = np.sqrt((1.0 + s) * (1.0 + s * z * z))
    rate = s * (z - 1.0) ** 2 / (root + 1.0 + s * z)

    if z < -1.0:
        rate = -2.0 * s * z
    elif s < 1.0 / 3.0:
        z_cut = (3.0 - s) / (1.0 - 3.0 * s)
        if z > z_cut:
            lambda_cut = (1.0 - s) / (1.0 + s)
            # At the cut, the generating function equals 4*s/(1+s).
            rate = 2.0 * s * lambda_cut * z - 4.0 * s / (1.0 + s)

    return e['stiffness'] * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        design = np.asarray([response(record['input']) for record in records])
        values = np.asarray([record['value'] for record in records])
        sigmas = np.asarray([record['sigma'] for record in records])
        scaled_design = design / sigmas
        scaled_values = values / sigmas
        estimate = np.dot(scaled_design, scaled_values) / np.dot(
            scaled_design, scaled_design
        )
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
