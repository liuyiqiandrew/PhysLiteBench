import numpy as np
from scipy.optimize import minimize_scalar


def coefficient(xi,fraction):
    """Limiting osmotic coefficient for the counterion mixture.

    ``xi`` is the bare line-charge Manning parameter, lB*u*tau0.  The
    trivalent fraction is a fraction of neutralizing charge, not a fraction
    of ions.  Below the first condensation threshold the number of ions per
    unit neutralizing charge is (1-f)+f/3, giving the usual mixture result.
    Above that threshold the trivalent ions are condensed in the infinite
    cell limit; if monovalent ions are present, they determine the pressure
    at the distant wall.
    """
    f = float(fraction)

    def single_valence(manning):
        return (1.0 - manning / 2.0 if manning <= 1.0
                else 1.0 / (2.0 * manning))

    # Pure preparations have no competing lower-valence species at the
    # distant wall.
    if f <= 0.0:
        return single_valence(xi)
    if f >= 1.0:
        return single_valence(3.0 * xi)

    number_per_charge = 1.0 - f + f / 3.0
    if xi <= 1.0 / 3.0:
        return 1.0 - xi / (2.0 * number_per_charge)

    # All trivalent charge is condensed.  The monovalent ions see the
    # residual line charge (1-f)*xi, while the normalization still includes
    # both species' fixed numbers.
    return ((1.0 - f) / number_per_charge *
            single_valence((1.0 - f) * xi))


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

        inputs = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(values).all() or
                not np.isfinite(sigmas).all() or (sigmas <= 0).any()):
            raise ValueError('Calibration values and sigmas must be finite; sigmas must be positive.')

        def objective(tau0):
            residual = (predict_at(inputs, tau0) - values) / sigmas
            return float(np.dot(residual, residual))

        result = minimize_scalar(objective, bounds=(0.7, 1.4), method='bounded')
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError('Unable to fit the rod line-charge density.')
        self.line_density = float(result.x)
        return self

    def predict(self,experiments):
        if self.line_density is None:
            raise RuntimeError('Model must be fit before prediction.')
        return predict_at(experiments,self.line_density)
