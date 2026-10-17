import numpy as np


def response(e):
    """Heat rate per unit mobility for one preparation.

    In units of mobility * stiffness, the mean heat is 2*r**2 and the
    interior rate is the Legendre transform of

        psi(h) = 1 - sqrt(1 - 4*r**2*h*(1+h)).

    Stationary, unconstrained endpoints restrict the transform's domain.
    The final position gives a linear branch below z=-1. For weak drives,
    the initial position gives another branch at large positive z.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])
    r2 = r * r

    if z <= -1.0:
        rate = -2.0 * r2 * z
    elif r2 < 1.0 / 3.0 and z >= (3.0 - r2) / (1.0 - 3.0 * r2):
        rate = 2.0 * r2 * ((1.0 - r2) * z - 2.0) / (1.0 + r2)
    else:
        root = np.sqrt((1.0 + r2) * (1.0 + r2 * z * z))
        # Rationalization preserves I(1)=0 and avoids cancellation nearby.
        rate = r2 * (z - 1.0) ** 2 / (root + 1.0 + r2 * z)

    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        # Independent Gaussian readings give a weighted least-squares fit
        # because each predicted rate is linear in the common mobility.
        unit_rates = np.asarray([response(record['input']) for record in records])
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        scaled_rates = unit_rates / sigmas
        denominator = np.dot(scaled_rates, scaled_rates)
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError('Calibration records do not identify mobility.')
        estimate = np.dot(scaled_rates, values / sigmas) / denominator
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
