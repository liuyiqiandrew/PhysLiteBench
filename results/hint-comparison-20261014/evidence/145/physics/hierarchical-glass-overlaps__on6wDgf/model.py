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


def leaf_partition(alpha, blocks, parent_parameter=0.0):
    """Probability of one labelled descendant partition under a tilted PD law.

    Selecting a parent from the full Gibbs measure biases its descendant
    partition sum by ``S**parent_parameter``.  The normalized descendant
    weights consequently have the Poisson--Dirichlet law PD(alpha,
    -parent_parameter), rather than the un-tilted PD(alpha, 0) law.
    ``parent_parameter=0`` retains the ordinary one-level expression.
    """
    n = sum(blocks)
    k = len(blocks)
    result = 1.0
    for j in range(1, k):
        result *= j * alpha - parent_parameter
    for j in range(1, n):
        result /= j - parent_parameter
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
    # Before the first transition, the probability that two replicas select
    # one of exponentially many parents vanishes.  It also avoids evaluating
    # the formally singular PD(alpha, -q) expression at q == 1.
    if q == 1.0:
        return 0.0
    if observable == 'state_collision':
        return parent * leaf_partition(alpha, [replicas], q)
    return parent * leaf_partition(alpha, blocks, q)


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
        """Fit the material scale by the Gaussian-error weighted MLE.

        All supplied calibration temperatures are in the unfrozen regime, in
        which the thermodynamic mean energy density is -beta * width**2.
        Fitting width squared makes this a one-parameter weighted regression.
        """
        numerator = 0.0
        denominator = 0.0
        count = 0
        for row in records:
            try:
                beta = float(row['input']['beta'])
                observed = float(row['value'])
                sigma = float(row['sigma'])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError('Calibration records need beta, value, and sigma.') from error
            if (not math.isfinite(beta) or not .2 <= beta <= .6
                    or not math.isfinite(observed)
                    or not math.isfinite(sigma) or sigma <= 0):
                raise ValueError('Invalid calibration record.')
            precision = 1.0 / (sigma * sigma)
            numerator -= beta * observed * precision
            denominator += beta * beta * precision
            count += 1
        if count == 0:
            raise ValueError('Supply at least one calibration record.')
        squared = numerator / denominator
        lower, upper = WIDTH_INTERVAL
        squared = min(upper * upper, max(lower * lower, squared))
        self.width = math.sqrt(squared)
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        result = []
        for row in records:
            try:
                beta = float(row['input']['beta'])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError('Calibration records need an input beta.') from error
            if not math.isfinite(beta) or not .2 <= beta <= .6:
                raise ValueError('Outside the stated calibration range.')
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
