import numpy as np
from scipy.optimize import minimize_scalar


def predict_at(experiments, mobility):
    """Return the principal decay rate for each experiment.

    With ``a = mu*k`` and ``D = mu*T``, the radial Feynman--Kac
    eigenfunction is proportional to

        r**abs(lambda) * exp(-c*r**2/2),

    where ``c`` is the positive root of ``D*c**2 + a*c - s = 0``.
    Its eigenvalue gives the rate

        (abs(lambda) + 1) * sqrt(a**2 + 4*D*s) - a.

    In particular, the radial probe is not additive independently of the
    winding sector: it is multiplied by ``abs(lambda) + 1``.
    """
    k = np.asarray([e['stiffness'] for e in experiments], dtype=float)
    temperature = np.asarray([e['temperature'] for e in experiments], dtype=float)
    winding = np.abs(np.asarray([e['probe'] for e in experiments], dtype=float))
    radial = np.asarray([e['radial_probe'] for e in experiments], dtype=float)

    relaxation = mobility * k
    diffusion = mobility * temperature
    return (winding + 1.0) * np.sqrt(relaxation**2 + 4.0 * diffusion * radial) - relaxation


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        experiments = [record['input'] for record in records]
        observed = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)

        def objective(mobility):
            residual = (predict_at(experiments, mobility) - observed) / sigma
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            objective,
            bounds=(0.7, 1.4),
            method='bounded',
            options={'xatol': 1e-12},
        )
        self.mobility = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
