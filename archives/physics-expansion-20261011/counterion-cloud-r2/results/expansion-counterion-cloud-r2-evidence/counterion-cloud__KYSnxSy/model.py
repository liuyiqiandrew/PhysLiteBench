import numpy as np
from scipy.optimize import minimize_scalar


def coefficient(xi,fraction):
    """Limiting osmotic coefficient for a counterion mixture.

    ``xi`` is the Bjerrum-weighted rod charge, before accounting for the
    number of counterions needed per unit rod charge.  The latter is
    ``1-f + f/3`` for monovalent and trivalent counterions, respectively.
    The result is the cylindrical salt-free PB limiting law.
    """
    number_per_charge = 1.0 - fraction + fraction / 3.0
    effective_xi = xi / number_per_charge
    if effective_xi <= 1.0:
        return 1.0 - effective_xi / 2.0
    return 1.0 / (2.0 * effective_xi)


def predict_at(experiments,line_density):
    return np.asarray([coefficient(e['bjerrum_length']*e['charge_multiplier']*
                                   line_density,e['trivalent_charge_fraction'])
                       for e in experiments])


class Model:
    def __init__(self):
        self.line_density = None

    def fit(self,records):
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        def objective(tau):
            experiments = [record['input'] for record in records]
            observed = np.asarray([record['value'] for record in records],
                                  dtype=float)
            sigma = np.asarray([record['sigma'] for record in records],
                               dtype=float)
            if np.any(~np.isfinite(observed)) or np.any(~np.isfinite(sigma)):
                raise ValueError('Calibration values and sigmas must be finite.')
            if np.any(sigma <= 0.0):
                raise ValueError('Calibration sigmas must be positive.')
            fitted = predict_at(experiments, tau)
            residual = (fitted - observed) / sigma
            return float(np.dot(residual, residual))

        result = minimize_scalar(objective, bounds=(0.7, 1.4), method='bounded',
                                 options={'xatol': 1e-12})
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError('Unable to fit the rod line-charge density.')
        self.line_density = float(np.clip(result.x, 0.7, 1.4))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.line_density)
