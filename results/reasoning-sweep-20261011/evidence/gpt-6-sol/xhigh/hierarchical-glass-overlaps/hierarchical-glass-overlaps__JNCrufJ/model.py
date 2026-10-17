"""Thermodynamic calibration and replica events for the two-level GREM."""
import math
from numbers import Real, Integral
import numpy as np

WIDTH_INTERVAL = (0.8, 1.2)
ENTROPIES = (0.3, 0.7)
VARIANCES = (0.7, 0.3)


def level_parameter(beta, width, level):
    """Poisson-Dirichlet parameter of a frozen level (one if unfrozen)."""
    critical = math.sqrt(2*ENTROPIES[level]/VARIANCES[level])/width
    return min(1.0, critical/beta)


def participation(m, replicas):
    result = 1.0
    for j in range(1, replicas):
        result *= (j-m)/j
    return result


def leaf_partition(parent_parameter, leaf_parameter, blocks):
    """Joint probability of a specified leaf partition and one shared parent.

    In the frozen cascade a parent selected by replicas has a tilted leaf
    law, PD(leaf_parameter, -parent_parameter). Multiplying its EPPF by
    the parent coincidence cancels the denominator of that tilt.
    The same expression extends continuously to either critical point.
    """
    n = sum(blocks)
    k = len(blocks)
    result = 1.0/math.factorial(n-1)
    for j in range(1, k):
        result *= j*leaf_parameter-parent_parameter
    for size in blocks:
        for j in range(1,size):
            result *= j-leaf_parameter
    return result


def coincidence(beta, replicas, observable, width, blocks=None):
    q = level_parameter(beta,width,0)
    alpha = level_parameter(beta,width,1)
    if observable == 'parent_collision':
        return participation(q,replicas)
    if observable == 'state_collision':
        return participation(alpha,replicas)
    return leaf_partition(q,alpha,blocks)


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
        # Every calibration beta is below the first freezing transition, so
        # the limiting energy density is -beta * width**2. Gaussian errors
        # make weighted least squares the maximum-likelihood fit in width**2.
        numerator = []
        denominator = []
        for row in records:
            beta = float(row['input']['beta'])
            measured = float(row['value'])
            sigma = float(row['sigma'])
            if (not math.isfinite(beta) or not .2 <= beta <= .6
                    or not math.isfinite(measured)
                    or not math.isfinite(sigma) or sigma <= 0):
                raise ValueError('Invalid calibration record.')
            weight = 1.0/(sigma*sigma)
            numerator.append(-beta*measured*weight)
            denominator.append(beta*beta*weight)
        if not denominator:
            raise ValueError('At least one calibration record is required.')
        squared = math.fsum(numerator)/math.fsum(denominator)
        squared = min(max(squared, WIDTH_INTERVAL[0]**2), WIDTH_INTERVAL[1]**2)
        self.width = math.sqrt(squared)
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        result = []
        for row in records:
            beta = float(row['input']['beta'])
            if not math.isfinite(beta) or not .2 <= beta <= .6:
                raise ValueError('Outside the stated calibration range.')
            result.append(-beta*value*value)
        return np.asarray(result,dtype=float)

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
