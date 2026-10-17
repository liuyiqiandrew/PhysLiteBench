import numpy as np


_MOBILITY_BOUNDS = (0.8, 1.3)


def response(e):
    """Dimensionless rate function for normalized heat fluctuations.

    For the stationary two-dimensional OU process the heat SCGF has a square
    root form.  Writing the heat in units of its stationary mean gives the
    expression below; the rate then scales with the relaxation rate
    mobility * stiffness.
    """
    r, z = e['drive_ratio'], e['relative_heat']
    q = 2.0 * r * r * z
    radicand = (1.0 + r*r) * (q*q + 4.0*r*r) / (4.0*r*r)
    # The direct subtraction loses precision close to the typical value z=1.
    root = np.sqrt(radicand)
    numerator = radicand - (q/2.0 + 1.0)**2
    denominator = root + q/2.0 + 1.0
    rate = numerator / denominator
    return e['stiffness'] * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        if not records:
            raise ValueError('records must contain at least one calibration reading')
        values = np.asarray([float(record['value']) for record in records])
        sigmas = np.asarray([float(record['sigma']) for record in records])
        if np.any(~np.isfinite(values)) or np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0):
            raise ValueError('calibration values and positive sigmas must be finite')
        base = np.asarray([response(record['input']) for record in records], dtype=float)
        weights = 1.0 / sigmas**2
        # Weighted least squares for y_i = mobility * response_i, constrained
        # to the apparatus' physically allowed common mobility interval.
        denominator = np.sum(weights * base**2)
        estimate = np.sum(weights * base * values) / denominator
        self.mobility = float(np.clip(estimate, *_MOBILITY_BOUNDS))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
