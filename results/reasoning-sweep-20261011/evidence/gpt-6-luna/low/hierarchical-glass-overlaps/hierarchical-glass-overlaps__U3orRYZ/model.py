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
    """Infinite-size disorder-averaged energy density of the two-level REM."""
    energy = 0.0
    for entropy, variance in zip(ENTROPIES, VARIANCES):
        critical = math.sqrt(2.0*entropy/variance)/width
        energy -= variance*width*width*min(beta, critical)
    return energy


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
        rows = list(records)
        if not rows:
            raise ValueError('At least one calibration record is required.')
        beta = np.asarray([float(row['input']['beta']) for row in rows], dtype=float)
        observed = np.asarray([float(row['value']) for row in rows], dtype=float)
        sigma = np.asarray([float(row['sigma']) for row in rows], dtype=float)
        if (not np.all(np.isfinite(beta)) or not np.all(np.isfinite(observed))
                or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0)):
            raise ValueError('Calibration records must be finite with positive uncertainties.')
        if np.any((beta < .2) | (beta > .6)):
            raise ValueError('Calibration beta must be in [0.2,0.6].')
        weights = 1.0 / sigma**2
        # The limiting mean is monotone in width.  Minimize the weighted
        # Gaussian negative log likelihood on the allowed material interval.
        lo, hi = WIDTH_INTERVAL
        for _ in range(80):
            x1 = lo + (hi-lo)/3.0
            x2 = hi - (hi-lo)/3.0
            p1 = np.asarray([calibration_mean(b, x1) for b in beta])
            p2 = np.asarray([calibration_mean(b, x2) for b in beta])
            e1 = np.sum(weights*(p1-observed)**2)
            e2 = np.sum(weights*(p2-observed)**2)
            if e1 <= e2:
                hi = x2
            else:
                lo = x1
        self.width = (lo+hi)/2.0
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        return np.asarray([calibration_mean(float(row['input']['beta']), value)
                           for row in records],dtype=float)

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
