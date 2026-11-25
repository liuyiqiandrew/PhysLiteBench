import numpy as np


def predict_at(experiments,line_density):
    """Return the infinite-dilution cylindrical-cell osmotic coefficient.

    In the salt-free cylindrical Poisson--Boltzmann cell model the result
    depends on the Manning parameter ``xi = lB*u*tau0`` only.  The two
    branches join continuously at xi=1.
    """
    xi = np.asarray(
        [e['bjerrum_length'] * e['charge_multiplier'] * line_density
         for e in experiments],
        dtype=float,
    )
    return np.where(xi <= 1.0, 1.0 - xi / 2.0, 1.0 / (2.0 * xi))


class Model:
    def __init__(self):
        self.line_density = None

    def fit(self,records):
        # All calibration records have u <= 0.4, so xi remains below the
        # Manning threshold throughout the permitted tau0 range.  On that
        # branch phi = 1 - (lB*u)*tau0/2, making the Gaussian maximum-
        # likelihood estimate a one-parameter weighted least-squares fit.
        x = np.asarray([
            r['input']['bjerrum_length'] * r['input']['charge_multiplier']
            for r in records
        ], dtype=float)
        y = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        weights = 1.0 / sigma**2

        tau0 = 2.0 * np.sum(weights * x * (1.0 - y)) / np.sum(weights * x**2)
        self.line_density = float(np.clip(tau0, 0.7, 1.4))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.line_density)
