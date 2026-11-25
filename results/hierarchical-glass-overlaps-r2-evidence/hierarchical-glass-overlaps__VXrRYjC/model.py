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
        """Fit the material scale from equilibrium mean-energy readings.

        Throughout the calibration range both levels are in their
        high-temperature phase, so the limiting disorder-averaged energy
        density is ``-beta * width**2``.  The readings have known, possibly
        unequal, standard deviations; minimizing their weighted squared
        residuals gives the unconstrained estimate of ``width**2`` directly.
        """
        rows = list(records)
        if not rows:
            raise ValueError('Calibration records are required to fit width.')

        beta = []
        measured = []
        sigma = []
        for row in rows:
            try:
                b = float(row['input']['beta'])
                y = float(row['value'])
                s = float(row['sigma'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Malformed calibration record.') from exc
            if (not math.isfinite(b) or not .2 <= b <= .6
                    or not math.isfinite(y) or not math.isfinite(s) or s <= 0):
                raise ValueError('Calibration records are outside the stated ranges.')
            beta.append(b)
            measured.append(y)
            sigma.append(s)

        beta = np.asarray(beta, dtype=float)
        measured = np.asarray(measured, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        weights = 1.0 / np.square(sigma)
        width_squared = -np.sum(weights * beta * measured) / np.sum(weights * beta * beta)
        width_squared = float(np.clip(width_squared, WIDTH_INTERVAL[0] ** 2,
                                      WIDTH_INTERVAL[1] ** 2))
        self.width = math.sqrt(width_squared)
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        result = []
        for row in records:
            beta = float(row['input']['beta'])
            if not math.isfinite(beta) or not .2 <= beta <= .6:
                raise ValueError('Outside the stated calibration inverse-temperature range.')
            result.append(-beta * value * value)
        return np.asarray(result, dtype=float)

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
