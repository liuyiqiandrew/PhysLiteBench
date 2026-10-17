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


def calibration_mean(beta, width):
    """Infinite-size equilibrium energy density in the calibration regime.

    The calibration temperatures are all below the first freezing
    temperature for every allowed value of ``width``.  Consequently the two
    Gaussian contributions are annealed and their variances add to one.
    """
    return -beta * width * width


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
        """Fit the material scale by Gaussian weighted least squares.

        Writing ``x = width**2``, every calibration mean is simply
        ``-beta*x``.  This makes the constrained likelihood a one-dimensional
        quadratic, so it can be minimized exactly rather than numerically.
        """
        numerator = 0.0
        denominator = 0.0
        count = 0
        for row in records:
            try:
                beta = float(row['input']['beta'])
                observed = float(row['value'])
                sigma = float(row['sigma'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Calibration records need finite beta, value, and sigma fields.') from exc
            if (not math.isfinite(beta) or not .2 <= beta <= .6
                    or not math.isfinite(observed) or not math.isfinite(sigma) or sigma <= 0.0):
                raise ValueError('Calibration records are outside the stated range.')
            weight = 1.0 / (sigma * sigma)
            numerator += weight * beta * observed
            denominator += weight * beta * beta
            count += 1
        if count == 0 or not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
            raise ValueError('At least one valid calibration record is required.')
        squared = -numerator / denominator
        squared = min(WIDTH_INTERVAL[1] ** 2, max(WIDTH_INTERVAL[0] ** 2, squared))
        self.width = math.sqrt(squared)
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        result = []
        for row in records:
            try:
                beta = float(row['input']['beta'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Calibration records need an input beta.') from exc
            if not math.isfinite(beta) or not .2 <= beta <= .6:
                raise ValueError('Outside the stated inverse-temperature range.')
            result.append(calibration_mean(beta, value))
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
