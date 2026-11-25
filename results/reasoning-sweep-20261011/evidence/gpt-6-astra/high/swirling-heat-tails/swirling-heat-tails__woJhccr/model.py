import numpy as np


def response(e):
    """Stationary heat large-deviation rate at unit mobility.

    In time units tau = mobility * stiffness * t, the mean heat rate is
    2*r**2. For the transform E[exp(-eta*Q)], the bulk scaled cumulant
    generator is 1 - omega, omega = sqrt(1 + 4*r**2*eta*(1-eta)).
    Since Q = circulating work + U(0) - U(t), integrating both stationary
    endpoints also requires 1 + omega > 2*abs(eta). The rate is therefore
    sup_eta [omega - 1 - 2*r**2*z*eta] over this restricted open domain.
    Its boundary limits give the linear tails below.
    """
    a = float(e['drive_ratio']) ** 2
    z = float(e['relative_heat'])

    if z < -1.0:
        # The upper transform boundary is eta = 1 for every drive.
        rate = -2.0 * a * z
    elif a < 1.0 / 3.0 and (1.0 - 3.0 * a) * z > 3.0 - a:
        # The lower endpoint boundary is eta = (a - 1)/(a + 1).
        # For a >= 1/3, the bulk branch point is reached first and there
        # is no finite positive-tail crossover.
        rate = 2.0 * a * ((1.0 - a) * z - 2.0) / (1.0 + a)
    else:
        root = np.sqrt((1.0 + a) * (1.0 + a * z * z))
        # Rationalize root - 1 - a*z to avoid cancellation near z = 1.
        rate = a * (z - 1.0) ** 2 / (root + 1.0 + a * z)

    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return mobility * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the common mobility by bounded Gaussian maximum likelihood."""
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        basis = np.asarray([response(record['input']) for record in records])
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if not (np.isfinite(sigma).all() and (sigma > 0.0).all()):
            raise ValueError('Calibration standard deviations must be finite and positive.')
        if not (np.isfinite(basis).all() and np.isfinite(values).all()):
            raise ValueError('Calibration responses and values must be finite.')

        weighted_basis = basis / sigma
        information = np.dot(weighted_basis, weighted_basis)
        if information == 0.0:
            raise ValueError('Calibration records do not identify the mobility.')
        estimate = np.dot(weighted_basis, values / sigma) / information
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise RuntimeError('Fit the model before predicting.')
        return predict_at(experiments, self.mobility)
