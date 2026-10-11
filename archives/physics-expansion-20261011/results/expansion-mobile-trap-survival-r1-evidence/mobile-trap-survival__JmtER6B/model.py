import numpy as np


def response(e, trap_diffusivity):
    """Return the long-time survival readout for one experiment.

    In one dimension, averaging over the initial Poisson process before
    taking the logarithm gives the Pascal-principle asymptotic for a mobile
    target.  Its leading coefficient is the same as for a stationary
    target, so the target diffusivity does not enter this readout:

        -log(S(t)) ~ 4 * density * sqrt(D_B * t / pi).
    """
    return 4.0 * e['density'] * np.sqrt(trap_diffusivity / np.pi)


def predict_at(experiments, trap_diffusivity):
    return np.asarray([response(e, trap_diffusivity) for e in experiments])


class Model:
    def __init__(self):
        self.trap_diffusivity = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        density = np.asarray([r['input']['density'] for r in records], dtype=float)
        value = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r.get('sigma', 1.0) for r in records], dtype=float)

        if (not np.all(np.isfinite(density)) or
                not np.all(np.isfinite(value)) or
                not np.all(np.isfinite(sigma)) or
                np.any(density <= 0.0) or np.any(sigma <= 0.0)):
            raise ValueError('Calibration records must contain finite positive density and sigma.')

        # The readout is linear in density.  Fit its slope with the known
        # Gaussian-noise weights, then convert the slope to D_B.
        weights = 1.0 / sigma**2
        slope = np.sum(weights * density * value) / np.sum(weights * density**2)
        diffusivity = np.pi * (slope / 4.0)**2

        # The apparatus specifies the physically allowed parameter range.
        self.trap_diffusivity = float(np.clip(diffusivity, 0.5, 1.5))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.trap_diffusivity)
