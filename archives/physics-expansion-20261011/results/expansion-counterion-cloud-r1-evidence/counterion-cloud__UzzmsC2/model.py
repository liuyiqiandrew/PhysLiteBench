import numpy as np


def predict_at(experiments,line_density):
    """Return the infinite-cell osmotic coefficient for each experiment.

    For a cylindrical counterion-only cell, the PB solution has a Manning
    crossover at ``xi = lB * u * tau0 = 1``.  The limiting wall pressure,
    divided by the mean counterion density, is

        1 - xi/2       (xi <= 1)
        1/(2*xi)       (xi > 1).

    The rod radius drops out of this limit, although it is part of the
    apparatus specification.
    """
    xi = np.asarray([
        e['bjerrum_length'] * e['charge_multiplier'] * line_density
        for e in experiments
    ], dtype=float)
    return np.where(xi <= 1.0, 1.0 - xi / 2.0, 1.0 / (2.0 * xi))


class Model:
    def __init__(self):
        self.line_density = None

    def fit(self,records):
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        # The calibration range is subcritical for every allowed tau0:
        # u <= 0.4, lB <= 1.25, and tau0 <= 1.4, so phi is linear in tau0.
        # This is the weighted maximum-likelihood estimate for the supplied
        # Gaussian errors, with the physical parameter interval enforced.
        x = np.asarray([
            r['input']['bjerrum_length'] * r['input']['charge_multiplier'] / 2.0
            for r in records
        ], dtype=float)
        y = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or
                not np.all(np.isfinite(sigma)) or np.any(sigma <= 0) or
                not np.any(x)):
            raise ValueError('Calibration records contain invalid values.')

        weights = 1.0 / sigma**2
        tau0 = np.sum(weights * x * (1.0 - y)) / np.sum(weights * x**2)
        self.line_density = float(np.clip(tau0, 0.7, 1.4))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.line_density)
