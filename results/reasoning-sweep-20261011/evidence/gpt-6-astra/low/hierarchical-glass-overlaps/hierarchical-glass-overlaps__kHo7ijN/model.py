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
    """Joint probability of one parent and the specified labelled leaf blocks.

    Frozen family masses have PD(q, 0) weights. Their internal weights
    have the total-mass tilted law PD(alpha, -q), not PD(alpha, 0).
    Multiplying that EPPF by parent participation cancels (1-q)_(n-1).
    The formula also covers diffuse leaves (alpha=1).
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
        # All leaf masses together have PD(alpha, 0) weights.
        return participation(alpha,replicas)
    if q == 1.0:
        return 0.0
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
        """Constrained Gaussian maximum likelihood, linear in width squared."""
        data = []
        for row in records:
            beta = float(row['input']['beta'])
            measured = float(row['value'])
            sigma = float(row['sigma'])
            if (not math.isfinite(beta) or not .2 <= beta <= .6
                    or not math.isfinite(measured)
                    or not math.isfinite(sigma) or sigma <= 0):
                raise ValueError('Use finite calibration data and positive sigma.')
            data.append((beta, measured, sigma))
        if not data:
            raise ValueError('Supply at least one calibration record.')
        beta, measured, sigma = np.asarray(data, dtype=float).T
        weights = (sigma.min()/sigma)**2
        squared = -np.dot(weights*beta, measured)/np.dot(weights, beta*beta)
        self.width = math.sqrt(float(np.clip(
            squared, WIDTH_INTERVAL[0]**2, WIDTH_INTERVAL[1]**2)))
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
