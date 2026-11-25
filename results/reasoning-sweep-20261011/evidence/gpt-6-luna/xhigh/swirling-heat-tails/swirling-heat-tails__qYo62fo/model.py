import numpy as np


def response(e):
    r, z = float(e['drive_ratio']), float(e['relative_heat'])
    r2 = r*r

    # Legendre transform of the heat SCGF gives
    # sqrt((1+r^2)(1+r^2 z^2)) - 1 - r^2 z.
    # Rationalizing avoids loss of precision close to the minimum at z=1.
    root = np.sqrt((1+r2)*(1+r2*z*z))
    denominator = root + 1 + r2*z
    rate = r2*(z-1)**2/denominator
    return float(e['stiffness'])*rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        base_rates = np.asarray([response(record['input']) for record in records], dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)

        # The mobility multiplies every rate, so the Gaussian maximum
        # likelihood estimate is the weighted least-squares slope through zero.
        weights = 1.0 / sigmas**2
        estimate = np.sum(weights*base_rates*values) / np.sum(weights*base_rates**2)
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
