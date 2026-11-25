"""Equilibrium calibration and configuration coincidences."""
import math
from numbers import Real, Integral
import numpy as np

WIDTH_INTERVAL = (0.8, 1.2)
ENTROPIES = (0.3, 0.7)
VARIANCES = (0.7, 0.3)


def level_parameter(beta, width, level):
    critical = math.sqrt(2*ENTROPIES[level]/VARIANCES[level])/width
    return min(1.0, critical/beta)


def participation(m, replicas):
    result = 1.0
    for j in range(1, replicas):
        result *= (j-m)/j
    return result


def leaf_partition(alpha, blocks):
    n = sum(blocks)
    k = len(blocks)
    result = alpha**(k-1)*math.factorial(k-1)/math.factorial(n-1)
    for size in blocks:
        for j in range(1,size):
            result *= j-alpha
    return result


def coincidence(beta, replicas, observable, width, blocks=None):
    q = level_parameter(beta,width,0)
    alpha = level_parameter(beta,width,1)
    parent = participation(q,replicas)
    if observable == 'parent_collision':
        return parent
    if observable == 'state_collision':
        return parent*participation(alpha,replicas)
    return parent*leaf_partition(alpha,blocks)


class Model:
    def __init__(self):
        self.width = None

    def _parameter(self, width):
        if width is not None and (isinstance(width,bool) or not isinstance(width,Real)):
            raise ValueError('Supply a real material scale.')
        value = self.width if width is None else float(width)
        if value is None or not math.isfinite(value) or not .8 <= value <= 1.2:
            raise ValueError('Fit or supply width in [0.8,1.2].')
        return value

    def fit(self, records):
        """Fit the material scale from the calibration measurements.

        Every calibration beta is below the first freezing threshold even at
        the largest allowed width.  Consequently the thermodynamic mean
        energy density is ``-beta * width**2``.  Writing ``x = width**2``
        makes the weighted least-squares fit a one-parameter linear fit.
        """
        rows = list(records)
        if not rows:
            raise ValueError('At least one calibration record is required.')

        beta = np.asarray([float(row['input']['beta']) for row in rows],
                          dtype=float)
        measured = np.asarray([float(row['value']) for row in rows],
                              dtype=float)
        sigma = np.asarray([float(row['sigma']) for row in rows], dtype=float)

        if (not np.all(np.isfinite(beta)) or not np.all(np.isfinite(measured))
                or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0)
                or np.any(beta < .2) or np.any(beta > .6)):
            raise ValueError('Invalid calibration record.')

        weights = 1.0 / (sigma * sigma)
        denominator = np.sum(weights * beta * beta)
        if not math.isfinite(float(denominator)) or denominator <= 0:
            raise ValueError('Invalid calibration record.')

        width_squared = np.sum(weights * beta * (-measured)) / denominator
        width_squared = float(width_squared)
        if not math.isfinite(width_squared) or width_squared <= 0:
            raise ValueError('Calibration does not give a positive width.')

        width = math.sqrt(width_squared)
        self.width = min(WIDTH_INTERVAL[1], max(WIDTH_INTERVAL[0], width))
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        return np.asarray([-float(row['input']['beta'])*value*value for row in records],dtype=float)

    def predict(self, experiments, width=None):
        value = self._parameter(width)
        result = []
        for row in experiments:
            beta = float(row['beta'])
            replicas = row['replicas']
            observable = row['observable']
            if not math.isfinite(beta) or not .2 <= beta <= 6:
                raise ValueError('Outside the stated inverse-temperature range.')
            if isinstance(replicas,bool) or not isinstance(replicas,Integral) or not 2 <= replicas <= 6:
                raise ValueError('Use an integer number of replicas from 2 through 6.')
            if observable not in ('parent_collision','state_collision','leaf_partition'):
                raise ValueError('Unknown observable.')
            blocks = None
            if observable == 'leaf_partition':
                blocks = list(row['partition'])
                if (not blocks or any(isinstance(n,bool) or not isinstance(n,Integral) or n<1 for n in blocks)
                        or sum(blocks)!=replicas):
                    raise ValueError('Partition sizes must be positive integers summing to replicas.')
            result.append(coincidence(beta,replicas,observable,value,blocks))
        return np.asarray(result,dtype=float)
