"""Equilibrium calibration and configuration coincidences.

The two random-energy levels become independent Poisson--Dirichlet weight
sequences in the thermodynamic limit.  This module uses their sampling
formulae directly, rather than simulating a finite landscape.
"""
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
    """Expected ``sum_i p_i**replicas`` for PD(m, 0) weights.

    ``m == 1`` represents the unfrozen phase and correctly gives zero for
    every collision of two or more replicas.
    """
    result = 1.0
    for j in range(1, replicas):
        result *= (j-m)/j
    return result


def leaf_partition(alpha, blocks):
    """PD(alpha, 0) EPPF for this one, ordered replica partition."""
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
        """Fit ``width`` by constrained Gaussian maximum likelihood.

        In the complete calibration range both REM levels are unfrozen, so
        the limiting energy density is ``-beta * width**2``.  The supplied
        standard deviations make weighted least squares the likelihood
        maximizer for ``width**2``.
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
                raise ValueError('Calibration records need finite beta, value, and sigma fields.') from error
            if (not math.isfinite(beta) or not .2 <= beta <= .6
                    or not math.isfinite(observed) or not math.isfinite(sigma)
                    or sigma <= 0.0):
                raise ValueError('Invalid calibration record.')
            weight = 1.0 / (sigma * sigma)
            numerator += beta * observed * weight
            denominator += beta * beta * weight
            count += 1
        if count == 0:
            raise ValueError('At least one calibration record is required to fit width.')

        # Let t = width**2.  Minimize sum ((value + beta*t) / sigma)**2.
        unconstrained = -numerator / denominator
        lower, upper = WIDTH_INTERVAL
        self.width = math.sqrt(min(upper * upper, max(lower * lower, unconstrained)))
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
                raise ValueError('Outside the stated calibration inverse-temperature range.')
            result.append(-beta * value * value)
        return np.asarray(result, dtype=float)

    def predict(self, experiments, width=None):
        value = self._parameter(width)
        result = []
        for row in experiments:
            try:
                beta = float(row['beta'])
                replicas = row['replicas']
                observable = row['observable']
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError('Experiments need beta, replicas, and observable fields.') from error
            if not math.isfinite(beta) or not .2 <= beta <= 6:
                raise ValueError('Outside the stated inverse-temperature range.')
            if isinstance(replicas,bool) or not isinstance(replicas,Integral) or not 2 <= replicas <= 6:
                raise ValueError('Use an integer number of replicas from 2 through 6.')
            if observable not in ('parent_collision','state_collision','leaf_partition'):
                raise ValueError('Unknown observable.')
            blocks = None
            if observable == 'leaf_partition':
                try:
                    blocks = list(row['partition'])
                except (KeyError, TypeError) as error:
                    raise ValueError('Leaf partitions need block sizes.') from error
                if (not blocks or any(isinstance(n,bool) or not isinstance(n,Integral) or n<1 for n in blocks)
                        or sum(blocks)!=replicas):
                    raise ValueError('Partition sizes must be positive integers summing to replicas.')
            result.append(coincidence(beta,replicas,observable,value,blocks))
        return np.asarray(result,dtype=float)
