import numpy as np


def response(e):
    r, z = float(e['drive_ratio']), float(e['relative_heat'])
    r2 = r * r
    q = 2 * r2 * z  # Heat rate divided by mobility * stiffness.

    # The Legendre saddle for psi(s) = 1 - sqrt(1 - 4*r²*s*(1+s)).
    u = q * np.sqrt(1 + r2) / (r * np.sqrt(q * q + 4 * r2))
    s = 0.5 * (u - 1)

    # The stationary initial and free final positions restrict the heat
    # moment generating function. At a restriction the rate is linear.
    s = max(s, -1.0)
    if r2 < 1 / 3:
        s = min(s, (1 - r2) / (1 + r2))

    discriminant = max(0.0, 1 - 4 * r2 * s * (1 + s))
    rate = s * q - (1 - np.sqrt(discriminant))
    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        x = np.asarray([response(record['input']) for record in records])
        y = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        weights = 1 / sigma**2
        self.mobility = float(np.sum(weights * x * y) / np.sum(weights * x * x))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
