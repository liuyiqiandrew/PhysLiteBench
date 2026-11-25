import numpy as np


def response(e, trap_diffusivity):
    """Return the leading long-time readout for one experiment.

    In one dimension the target's Brownian motion only changes subleading
    terms in ``-log(S(t))``.  The leading term is set by the range explored
    by the Poisson field of traps, and is consequently independent of the
    target diffusivity:

        -log(S(t)) ~ 4 * rho * sqrt(D_B * t / pi).
    """
    return 4.0 * e['density'] * np.sqrt(trap_diffusivity / np.pi)


def predict_at(experiments, trap_diffusivity):
    return np.asarray([response(e, trap_diffusivity) for e in experiments])


class Model:
    def __init__(self):
        self.trap_diffusivity = None

    def fit(self, records):
        # The calibration noise is additive with known standard deviation, so
        # fit the readout coefficient with a weighted least-squares estimate
        # through the physically required zero-density intercept.
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        density = np.asarray([r['input']['density'] for r in records], dtype=float)
        value = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)

        if (not np.all(np.isfinite(density)) or
                not np.all(np.isfinite(value)) or
                not np.all(np.isfinite(sigma)) or
                np.any(sigma <= 0.0) or
                np.any(density < 0.0)):
            raise ValueError('Calibration records must contain finite valid values.')

        weights = 1.0 / sigma**2
        coefficient = np.sum(weights * density * value) / np.sum(weights * density**2)
        diffusivity = np.pi * (coefficient / 4.0)**2
        # The apparatus specifies the admissible shared diffusivity range.
        self.trap_diffusivity = float(np.clip(diffusivity, 0.5, 1.5))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.trap_diffusivity)
