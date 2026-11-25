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
    """Joint probability of one labelled leaf partition in a single family.

    The two-level Gibbs cascade has family weights PD(q, 0). Selecting
    a family by its Gibbs mass tilts its internal law to PD(alpha, -q).
    Multiplying that partition law by the family participation cancels
    the factors (j-q), leaving the expression below. In particular, a
    single leaf block has PD(alpha, 0) participation, without an extra
    family factor. At alpha=1 the leaves are diffuse.
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
        """Fit the Gaussian likelihood, linear in the parameter width**2."""
        data = np.asarray([
            (float(row['input']['beta']), float(row['value']), float(row['sigma']))
            for row in records
        ], dtype=float)
        if data.size == 0:
            raise ValueError('Supply at least one calibration record.')
        if (not np.all(np.isfinite(data))
                or np.any((data[:, 0] < .2) | (data[:, 0] > .6))
                or np.any(data[:, 2] <= 0)):
            raise ValueError('Use finite calibration data, beta in [0.2,0.6], and positive sigma.')
        beta, measured, sigma = data.T
        # Both levels are unfrozen throughout the calibration range, so
        # the limiting energy per N is -beta * width**2. Rescale inverse
        # uncertainties before squaring to avoid unnecessarily large weights.
        weights = np.square(np.min(sigma)/sigma)
        squared_width = -np.sum(weights*beta*measured)/np.sum(weights*beta*beta)
        self.width = math.sqrt(float(np.clip(
            squared_width, WIDTH_INTERVAL[0]**2, WIDTH_INTERVAL[1]**2)))
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
