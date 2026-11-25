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
    """Joint probability of one family and a specified labelled partition.

    The two-level Gibbs cascade has family weights PD(q, 0). Within a
    Gibbs-selected family the leaf weights follow PD(alpha, -q): weighting
    a family by its partition function tilts its internal distribution.
    Multiplying its partition probability by the family participation
    cancels the factors (j-q) in the denominator, giving this product.
    Alpha=1 includes diffuse leaves, where only singleton blocks survive.
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
        # Flattening the cascade gives PD(alpha, 0) state weights.
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
        """Fit width by constrained Gaussian maximum likelihood.

        Every calibration beta lies below the first critical beta even
        at width=1.2, so the mean is -beta*width**2. Weighted least squares
        is quadratic in width**2 and has an exact constrained solution.
        """
        readings = []
        for row in records:
            beta = float(row['input']['beta'])
            measured = float(row['value'])
            sigma = float(row['sigma'])
            if not math.isfinite(beta) or not .2 <= beta <= .6:
                raise ValueError('Outside the stated calibration range.')
            if not math.isfinite(measured) or not math.isfinite(sigma) or sigma <= 0:
                raise ValueError('Use finite readings and positive finite sigma.')
            readings.append((beta, measured, sigma))
        if not readings:
            raise ValueError('Supply at least one calibration reading.')
        beta, measured, sigma = np.asarray(readings, dtype=float).T
        # A common weight factor cancels; rescaling avoids squaring 1/sigma.
        weights = (np.min(sigma)/sigma)**2
        width_squared = -np.sum(weights*beta*measured)/np.sum(weights*beta*beta)
        low, high = WIDTH_INTERVAL
        self.width = math.sqrt(float(np.clip(width_squared, low*low, high*high)))
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
