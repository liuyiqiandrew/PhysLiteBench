import numpy as np


def response(e, mobility=1.0):
    """Asymptotic annealed killing rate for one experiment.

    A target path with long-time velocity ``u`` pays
    ``(u - mobility * force)**2 / (4 * mobility * temperature)`` per unit
    time, while the Poisson traps contribute ``density * abs(u)``.  The
    minimum over ``u`` is the finite-temperature rate below.  The
    trap diffusivity affects only sublinear terms in this asymptotic limit.
    """
    density = float(e['density'])
    force = float(e['force'])
    temperature = float(e['temperature'])
    v = mobility * force

    if temperature <= 0.0:
        return density * abs(v)

    target_diffusivity = mobility * temperature
    speed = abs(v)
    threshold = 2.0 * density * target_diffusivity

    if speed <= threshold:
        return v * v / (4.0 * target_diffusivity)
    return density * speed - density * density * target_diffusivity


def predict_at(experiments, mobility):
    return np.asarray([response(e, mobility) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Estimate the shared mobility from the T=0 calibration records."""
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        # At T=0, K = mobility * density * abs(force).  Use the supplied
        # measurement uncertainties in a weighted least-squares fit through
        # the physically required zero intercept.
        design = np.asarray(
            [float(r['input']['density']) * abs(float(r['input']['force']))
             for r in records],
            dtype=float,
        )
        values = np.asarray([float(r['value']) for r in records], dtype=float)
        sigma = np.asarray([float(r['sigma']) for r in records], dtype=float)

        informative = design != 0.0
        if not np.any(informative):
            raise ValueError('Calibration records contain no nonzero force.')
        if (not np.isfinite(design[informative]).all()
                or not np.isfinite(values[informative]).all()
                or not np.isfinite(sigma[informative]).all()
                or np.any(sigma[informative] <= 0.0)):
            raise ValueError('Calibration records must contain finite values and positive sigma.')

        x = design[informative]
        y = values[informative]
        weights = 1.0 / sigma[informative] ** 2
        estimate = np.sum(weights * x * y) / np.sum(weights * x * x)

        # The apparatus states the physical mobility range.  Clipping also
        # keeps predictions physical if a very noisy calibration is supplied.
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
