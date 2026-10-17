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


def leaf_partition(q, alpha, blocks):
    n = sum(blocks)
    k = len(blocks)
    # The leaf partition inside a selected parent has the cascade's
    # PD(alpha, -q) law.  Multiplication by the parent coincidence cancels
    # its rising-factorial denominator.
    result = 1.0/math.factorial(n-1)
    for i in range(1,k):
        result *= i*alpha-q
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
        # Throughout the calibration range both levels are unfrozen, so the
        # limiting energy density is -beta * width**2.  Fit width**2 by
        # inverse-variance weighted least squares, then respect its bounds.
        numerator = 0.0
        denominator = 0.0
        for row in records:
            beta = float(row['input']['beta'])
            measured = float(row['value'])
            sigma = float(row['sigma'])
            if (not math.isfinite(beta) or not .2 <= beta <= .6 or
                    not math.isfinite(measured) or
                    not math.isfinite(sigma) or sigma <= 0):
                raise ValueError('Invalid calibration record.')
            weight = 1.0/(sigma*sigma)
            numerator -= weight*beta*measured
            denominator += weight*beta*beta
        if denominator == 0:
            raise ValueError('At least one calibration record is required.')
        squared = min(WIDTH_INTERVAL[1]**2,
                      max(WIDTH_INTERVAL[0]**2,numerator/denominator))
        self.width = math.sqrt(squared)
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
