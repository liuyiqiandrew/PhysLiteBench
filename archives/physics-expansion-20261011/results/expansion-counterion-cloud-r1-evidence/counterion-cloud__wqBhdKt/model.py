import numpy as np


def _limiting_osmotic_coefficient(xi):
    """Return the salt-free cylindrical-cell limiting coefficient.

    The dimensionless Manning parameter is ``xi = lB * u * tau0``.  The
    infinite-cell Poisson--Boltzmann limit has two branches: the uncondensed
    branch below xi=1 and the condensed branch above it.
    """
    xi = np.asarray(xi, dtype=float)
    return np.where(xi <= 1.0, 1.0 - xi / 2.0, 1.0 / (2.0 * xi))


def predict_at(experiments, line_density):
    xi = np.asarray([
        e['bjerrum_length'] * e['charge_multiplier'] * line_density
        for e in experiments
    ], dtype=float)
    return _limiting_osmotic_coefficient(xi)


class Model:
    def __init__(self):
        self.line_density = None

    def fit(self,records):
        # The supplied calibration points have u <= 0.4, so the known
        # parameter range keeps xi below one for every calibration record.
        # Consequently phi = 1 - (lB*u)*tau0/2 is linear in tau0 there.
        x = np.asarray([
            r['input']['bjerrum_length'] * r['input']['charge_multiplier']
            for r in records
        ], dtype=float)
        y = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)

        weights = 1.0 / np.square(sigma)
        tau = 2.0 * np.sum(weights * x * (1.0 - y)) / np.sum(weights * x * x)
        self.line_density = float(np.clip(tau, 0.7, 1.4))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.line_density)
