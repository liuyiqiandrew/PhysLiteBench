import numpy as np
from scipy.optimize import minimize_scalar


def predict_at(experiments, mobility):
    """Return the asymptotic joint winding/occupation rate.

    For an isotropic two-dimensional Ornstein--Uhlenbeck process, the
    radial part of the tilted generator has an inverse-square potential
    from the winding probe and a quadratic potential from the occupation
    probe.  Its principal eigenvalue is

        mu*k - (abs(lambda)+1)*sqrt((mu*k)**2 + 4*mu*T*s),

    so the reported rate is the negative of that eigenvalue.
    """
    if mobility is None:
        raise RuntimeError('Call fit before predict.')

    k = np.asarray([e['stiffness'] for e in experiments], dtype=float)
    temperature = np.asarray([e['temperature'] for e in experiments], dtype=float)
    winding = np.abs(np.asarray([e['probe'] for e in experiments], dtype=float))
    radial = np.asarray([e['radial_probe'] for e in experiments], dtype=float)

    relaxation = mobility * k
    diffusion = mobility * temperature
    shifted = np.sqrt(relaxation**2 + 4.0 * diffusion * radial)
    return (winding + 1.0) * shifted - relaxation


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        experiments = [record['input'] for record in records]
        observed = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)

        if observed.size == 0:
            raise ValueError('At least one calibration record is required.')
        if np.any(~np.isfinite(observed)) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError('Calibration values and sigmas must be finite, with positive sigma.')

        def objective(mobility):
            residual = (predict_at(experiments, mobility) - observed) / sigma
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            objective,
            bounds=(0.7, 1.4),
            method='bounded',
            options={'xatol': 1e-12},
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError('Mobility fit failed.')
        self.mobility = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
