import numpy as np
from scipy.optimize import minimize_scalar


def coefficient(gamma, count):
    """Limiting osmotic coefficient for a finite cloud.

    In logarithmic radial coordinates, a sector with ``m`` ions at the
    outer scale contributes the power

        s_m = 2*m - gamma*m*(m+1)/2

    to the large-outer-radius configurational integral.  The dominant
    sector is the one with the largest power, including the m=0 bound
    sector.  Dividing that power by 2*count converts it to P/(kT*nbar).
    """
    if count not in (1, 2, 3, 4):
        raise ValueError('ion count must be one of 1, 2, 3, or 4')

    m = np.arange(count + 1, dtype=float)
    powers = 2.0*m - gamma*m*(m + 1.0)/2.0
    return float(np.max(powers)/(2.0*count))


def predict_at(experiments,coupling):
    return np.asarray([coefficient(e['temperature_multiplier']*coupling,e['ion_count'])
                       for e in experiments])


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self,records):
        records = list(records)
        if not records:
            raise ValueError('at least one calibration record is required')

        def objective(coupling):
            experiments = [record['input'] for record in records]
            observed = np.asarray([record['value'] for record in records],
                                  dtype=float)
            sigma = np.asarray([record['sigma'] for record in records],
                               dtype=float)
            residual = (predict_at(experiments, coupling) - observed)/sigma
            return float(np.sum(residual*residual))

        result = minimize_scalar(objective, bounds=(0.7, 1.4),
                                 method='bounded',
                                 options={'xatol': 1e-12})
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError('unable to fit the reference coupling')
        self.coupling = float(np.clip(result.x, 0.7, 1.4))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)
