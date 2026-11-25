import numpy as np
from scipy.optimize import minimize_scalar


def predict_at(experiments, mobility):
    """Return the exact long-time rate for the supplied experiments.

    After separating the angular coordinate, the tilted radial generator is

        D (d2 + r^-1 d - lambda^2 r^-2) - gamma*r*d - s*r^2,

    where ``D=mu*T`` and ``gamma=mu*k``.  Its lowest eigenvalue gives the
    measured rate, which is

        (abs(lambda) + 1)*sqrt(gamma^2 + 4*D*s) - gamma.

    The square-root factor multiplies the winding probe as well as the
    radial-occupation contribution.  This coupling matters when both probes
    are nonzero, even though the calibration data contains them separately.
    """
    experiments = list(experiments)
    k = np.asarray([e['stiffness'] for e in experiments], dtype=float)
    temperature = np.asarray([e['temperature'] for e in experiments], dtype=float)
    probe = np.abs(np.asarray([e['probe'] for e in experiments], dtype=float))
    radial = np.asarray([e['radial_probe'] for e in experiments], dtype=float)

    relaxation = mobility * k
    diffusion = mobility * temperature
    shifted = np.sqrt(relaxation**2 + 4.0 * diffusion * radial)

    # Write shifted-relaxation in a cancellation-free form.  This also makes
    # the s=0 limit exact, including for very small radial probes.
    radial_decay = (4.0 * diffusion * radial) / (shifted + relaxation)
    return probe * shifted + radial_decay


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
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(sigma)):
            raise ValueError('Calibration values and uncertainties must be finite.')
        if np.any(sigma <= 0.0):
            raise ValueError('Calibration uncertainties must be positive.')

        inverse_variance = 1.0 / sigma**2

        def objective(mobility):
            residual = predict_at(experiments, mobility) - observed
            return np.sum(inverse_variance * residual**2)

        result = minimize_scalar(
            objective,
            bounds=(0.7, 1.4),
            method='bounded',
            options={'xatol': 1e-12},
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError('Unable to fit the particle mobility.')
        self.mobility = float(result.x)
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise RuntimeError('Call fit before predict.')
        return predict_at(experiments, self.mobility)
