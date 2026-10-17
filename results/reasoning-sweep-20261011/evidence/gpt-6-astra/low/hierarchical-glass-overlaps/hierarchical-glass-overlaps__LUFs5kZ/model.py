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


def leaf_partition(alpha, blocks, q=0.0):
    """Joint probability of one parent and a specified labelled leaf partition.

    In the two-level cascade the parent masses have PD(q, 0) law.
    Their internal leaf masses have the tilted PD(alpha, -q) law,
    rather than an independent PD(alpha, 0) law. Multiplying its
    partition probability by the parent participation cancels the
    rising (1-q) factors, giving the finite product below. alpha=1
    also covers the unfrozen, diffuse leaf level.
    """
    n = sum(blocks)
    k = len(blocks)
    result = 1.0/math.factorial(n-1)
    for j in range(1, k):
        result *= j*alpha-q
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
    return leaf_partition(alpha,blocks,q)


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
        """Constrained Gaussian maximum likelihood fit of width squared.

        All calibration temperatures are below both freezing thresholds,
        so the limiting energy density is -beta * width**2.
        """
        rows = []
        for row in records:
            beta = float(row['input']['beta'])
            measured = float(row['value'])
            sigma = float(row['sigma'])
            if (not math.isfinite(beta) or not .2 <= beta <= .6
                    or not math.isfinite(measured)
                    or not math.isfinite(sigma) or sigma <= 0):
                raise ValueError('Use finite calibration data with beta in [0.2,0.6] and sigma > 0.')
            rows.append((beta, measured, sigma))
        if not rows:
            raise ValueError('Supply at least one calibration record.')
        # A common rescaling of inverse-variance weights leaves the fit
        # unchanged and avoids overflow for small measurement errors.
        scale = min(row[2] for row in rows)
        numerator = math.fsum(-(scale/sigma)**2*beta*y for beta,y,sigma in rows)
        denominator = math.fsum((scale/sigma)**2*beta*beta for beta,_,sigma in rows)
        squared = numerator/denominator
        self.width = math.sqrt(min(WIDTH_INTERVAL[1]**2,
                                   max(WIDTH_INTERVAL[0]**2, squared)))
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
