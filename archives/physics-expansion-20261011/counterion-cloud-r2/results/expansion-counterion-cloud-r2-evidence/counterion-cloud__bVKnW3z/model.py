import numpy as np
from scipy.optimize import minimize_scalar


def _single_species_coefficient(xi, valence):
    """Infinite-cell coefficient for one counterion valence."""
    valence_xi = valence * xi
    if valence_xi <= 1:
        return 1 - valence_xi / 2
    return 1 / (2 * valence_xi)


def coefficient(xi, fraction):
    """Limiting canonical PB osmotic coefficient.

    ``xi`` is lB times the positive rod charge per length and ``fraction``
    is the fraction of that charge carried by trivalent ions.  In the
    mixed case the counterion numbers are fixed separately.  Consequently,
    the trivalent species follows its own Manning threshold and cannot be
    replaced by an averaged valence.
    """
    xi = float(xi)
    fraction = float(fraction)

    number_per_charge = 1 - 2 * fraction / 3
    total_number = xi * number_per_charge
    beta = xi * (1 - fraction)

    # Below the largest-valence Manning threshold both species have a
    # finite wall concentration.  The charge-weighted result is exact.
    if xi < 1 / 3:
        return 1 - xi / (2 * number_per_charge)

    # If all trivalent charge were condensed, beta would be the residual
    # charge seen by the monovalent ions.  If beta is below 1/3, the
    # trivalent ions are sufficient to pin the diffuse charge at their
    # threshold.  The diffuse population then has charge 1/3, consisting
    # of beta monovalent charge and 1/3-beta trivalent charge.  Its wall
    # pressure numerator is N_diffuse - (1/3)^2/2.
    if beta <= 1 / 3:
        return (1 / 18 + 2 * beta / 3) / total_number

    # If beta is between 1/3 and 1, all trivalent charge is condensed and
    # the remaining monovalent population is subcritical.  Above beta=1
    # the monovalent population is itself in the Manning branch; its
    # limiting pressure numerator is 1/2.
    if beta <= 1:
        return (beta - beta * beta / 2) / total_number
    return 1 / (2 * total_number)


def predict_at(experiments,line_density):
    return np.asarray([coefficient(e['bjerrum_length']*e['charge_multiplier']*
                                   line_density,e['trivalent_charge_fraction'])
                       for e in experiments])


class Model:
    def __init__(self):
        self.line_density = None

    def fit(self,records):
        records = list(records)

        def objective(line_density):
            experiments = [record['input'] for record in records]
            observed = np.asarray([record['value'] for record in records],
                                  dtype=float)
            sigma = np.asarray([record['sigma'] for record in records],
                               dtype=float)
            residual = (predict_at(experiments, line_density) - observed) / sigma
            return float(np.dot(residual, residual))

        result = minimize_scalar(objective, bounds=(0.7, 1.4), method='bounded',
                                 options={'xatol': 1e-12})
        self.line_density = float(result.x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.line_density)
